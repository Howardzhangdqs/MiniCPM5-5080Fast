"""Shared helpers for MiniCPM5-2B offline tooling (Lane D, P2).

Provides:
  * architecture constants (MiniCPM5-2B / LlamaForCausalLM)
  * safetensors self-parsing (u64 LE header length + JSON header, bf16 data area)
  * BF16 <-> FP32 numpy conversions (RNE rounding on the way down)
  * FP8 E4M3 <-> FP32 numpy conversions (RNE, saturation; P4 v1, field-exact
    with the native lane: 1 sign + 4 exp (bias 7) + 3 mantissa, max finite
    magnitude 448, no inf, NaN encodings 0x7F/0xFF never emitted)
  * FNV-1a 64-bit hashing of section names
  * WPK v1 binary format: header (64B) + TOC entries (132B, field-exact)
  * section layout spec (fusion/reorder map for MiniCPM5-2B)

Memory discipline: weights are accessed through np.memmap uint16 views; only
per-matrix (or per-chunk, for the big lm_head) fp32 temporaries are allocated.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct

import numpy as np

# ---------------------------------------------------------------------------
# Architecture constants (models/minicpm5-2b/config.json)
# ---------------------------------------------------------------------------
MODEL_NAME = "openbmb/MiniCPM5-2B"
MODEL_REVISION = "3497c460c89e00520c3cfa2e73f49ab7647f1177"
NUM_LAYERS = 42
HIDDEN = 2048
INTER = 6144
VOCAB = 130560
N_Q_HEADS = 16
N_KV_HEADS = 2
HEAD_DIM = 128
GQA_GROUP = N_Q_HEADS // N_KV_HEADS  # 8
ROPE_THETA = 5_000_000.0
RMS_EPS = 1e-6
BOS_ID = 0
EOS_IDS = (1, 130073)

# ---------------------------------------------------------------------------
# WPK v1 binary format (little-endian)
# ---------------------------------------------------------------------------
WPK_MAGIC = 0x31504B4D          # "MKP1" little-endian
WPK_ABI_VERSION = 0x0002
WPK_TARGET_SM = 120
WPK_ALIGN = 256                 # data sections 256B aligned

DTYPE_BF16 = 0                  # dtype enum
DTYPE_FP8_E4M3 = 2              # dtype enum: per-row-scaled FP8 E4M3
LAYOUT_ROW_MAJOR = 0            # layout enum: [out, in] row-major

# Header 64B:
#   magic u32 | abi_version u16 | target_sm u16 | model_hash u8[32]
#   | section_count u32 | reserved u32=0 | toc_offset u64=64 | data_start u64
WPK_HEADER_FMT = "<IHH32sIIQQ"
WPK_HEADER_SIZE = struct.calcsize(WPK_HEADER_FMT)
assert WPK_HEADER_SIZE == 64

# TOC entry, field-exact per spec:
#   name_hash u64 | dtype u32 | layout u32 | rank u32 | dims u32[8]
#   | data_offset u64 | data_bytes u64 | scale_offset u64=0 | scale_bytes u64=0
#   | reserved u64[6]
# NOTE: these named fields sum to 132 bytes (8+4+4+4+32+8+8+8+8+48).  The task
# brief also states "TOC entry 128B", which is arithmetically impossible with
# the mandated field list.  We keep every field at its mandated width (spec is
# "field-by-field consistent, do not modify"), so one entry is 132 B.  The
# header's explicit `data_start` keeps parsing position-independent; the entry
# size is also recorded in model_manifest.json as `toc_entry_size`.
WPK_TOC_FMT = "<QIII8IQQQQ6Q"
WPK_TOC_ENTRY_SIZE = struct.calcsize(WPK_TOC_FMT)
assert WPK_TOC_ENTRY_SIZE == 132

# ---------------------------------------------------------------------------
# FNV-1a 64-bit (over UTF-8 bytes of the section name)
# ---------------------------------------------------------------------------
FNV64_BASIS = 14695981039346656037
FNV64_PRIME = 1099511628211
_U64_MASK = 0xFFFFFFFFFFFFFFFF


def fnv1a64(data: bytes) -> int:
    h = FNV64_BASIS
    for b in data:
        h ^= b
        h = (h * FNV64_PRIME) & _U64_MASK
    return h


# Known-answer check (standard FNV-1a 64 test vectors).
assert fnv1a64(b"") == FNV64_BASIS
assert fnv1a64(b"a") == 0xAF63DC4C8601EC8C

# ---------------------------------------------------------------------------
# BF16 <-> FP32 (numpy, uint16 storage)
# ---------------------------------------------------------------------------


def bf16_to_f32(u: np.ndarray) -> np.ndarray:
    """uint16 bf16 bit pattern -> float32 (exact widening)."""
    u = np.ascontiguousarray(u, dtype=np.uint32)
    return (u << np.uint32(16)).view(np.float32)


def f32_to_bf16(f: np.ndarray) -> np.ndarray:
    """float32 -> uint16 bf16 with round-to-nearest-even truncation."""
    f = np.ascontiguousarray(f, dtype=np.float32)
    u = f.view(np.uint32)
    r = (u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))) >> np.uint32(16)
    return r.astype(np.uint16)


# ---------------------------------------------------------------------------
# FP8 E4M3 <-> FP32 (pure numpy bit-twiddling; P4 v1 spec, field-exact with
# the native lane -- do not modify):
#   * 1 sign | 4 exponent (bias 7) | 3 mantissa (4 significant bits)
#   * largest finite magnitude 448.0 (code 0x7E); no inf
#   * RNE rounding; magnitudes beyond 448 saturate to +/-448 so the NaN
#     encodings 0x7F / 0xFF are never produced
#   * subnormal grid: k * 2^-9 (k = 0..7), min normal 2^-6
# ---------------------------------------------------------------------------

FP8_MAX = 448.0                 # largest finite E4M3 magnitude
FP8_SCALE_FLOOR = 1e-12         # per-row scale lower bound


def f32_to_e4m3(f: np.ndarray) -> np.ndarray:
    """float32 -> uint8 E4M3 bit pattern (RNE, saturate to +/-448).

    Works on magnitude bits only (sign carried through untouched), so -0.0
    encodes to 0x80 and NaN/inf magnitudes saturate to +/-448.
    """
    f = np.ascontiguousarray(f, dtype=np.float32)
    u = f.view(np.uint32)
    sign = (u >> np.uint32(31)).astype(np.uint8)                  # 0/1
    a = u & np.uint32(0x7FFFFFFF)                                 # magnitude bits

    exp_f = (a >> np.uint32(23)) & np.uint32(0xFF)                # biased f32 exp
    m = (a & np.uint32(0x7FFFFF)) | np.uint32(0x800000)           # significand w/ implicit bit

    # --- normal path (value >= 2^-6): rebias 127 -> 7, round 24-bit
    # significand down to 4 bits (RNE). q+rnd-8 == 8 must CARRY into the
    # exponent field, so combine with + (| would silently drop the carry
    # whenever the exponent field is odd, e.g. 124 -> 128).
    q = m >> np.uint32(20)                                        # 8..15
    r = m & np.uint32(0xFFFFF)
    rnd = ((r > np.uint32(0x80000))
           | ((r == np.uint32(0x80000)) & ((q & np.uint32(1)) == np.uint32(1)))).astype(np.uint32)
    code_n = ((exp_f - np.uint32(120)) << np.uint32(3)) + (q + rnd - np.uint32(8))

    # --- subnormal path (0 <= value < 2^-6): round value/2^-9 to integer
    # k = 0..8 (k == 8 lands exactly on the min-normal code 0x08).
    # shift is clamped at 31 (numpy shifts are UB past the bit width);
    # every element clamped there is < 2^-17 and provably rounds to 0.
    shift = np.minimum(np.uint32(141) - exp_f, np.uint32(31))
    q2 = m >> shift
    r2 = m & ((np.uint32(1) << shift) - np.uint32(1))
    half = np.uint32(1) << (shift - np.uint32(1))
    rnd2 = ((r2 > half)
            | ((r2 == half) & ((q2 & np.uint32(1)) == np.uint32(1)))).astype(np.uint32)
    code_s = q2 + rnd2

    code = np.where(exp_f >= np.uint32(121), code_n, code_s)

    # anything at or above the would-be-NaN code saturates to 0x7E (+448);
    # this also swallows inf/NaN magnitudes and fp32 spill past 464.
    code = np.minimum(code, np.uint32(0x7E))
    return (sign << np.uint8(7)) | code.astype(np.uint8)


def _build_e4m3_decode_table() -> np.ndarray:
    """256-entry fp32 LUT built from pure integer bit manipulation."""
    bits = np.zeros(256, dtype=np.uint32)
    for code in range(256):
        s = (code >> 7) & 1
        e = (code >> 3) & 0xF
        m = code & 0x7
        if e == 0:
            if m == 0:
                b = 0                                   # +/-0
            else:
                p = m.bit_length() - 1                  # normalize m = 1.f * 2^p
                b = ((118 + p) << 23) | ((m - (1 << p)) << (23 - p))
        else:
            b = ((120 + e) << 23) | (m << 20)
        bits[code] = (s << 31) | b
    return bits.view(np.float32)


_E4M3_TO_F32 = _build_e4m3_decode_table()
# decode conventions: 0x7E/0xFE -> +/-448 (max finite); the NaN encodings
# 0x7F/0xFF decode to +/-480 by table construction but are never emitted
# by f32_to_e4m3 and never appear in a conforming pack.


def e4m3_to_f32(q: np.ndarray) -> np.ndarray:
    """uint8 E4M3 -> float32 (256-entry LUT gather)."""
    return _E4M3_TO_F32[np.asarray(q, dtype=np.uint8)]


def fp8_row_scales(w32: np.ndarray) -> np.ndarray:
    """Per-output-row scales for an [N,K] fp32 weight: max|row|/448 (>=1e-12)."""
    amax = np.max(np.abs(w32), axis=-1)
    return np.maximum(amax / np.float32(FP8_MAX), np.float32(FP8_SCALE_FLOOR)).astype(np.float32)


def quantize_fp8_rows(w32: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """w_q = e4m3(w / scale), row-wise. w32: [N,K] fp32, scales: [N] fp32."""
    return f32_to_e4m3(w32 / scales[..., np.newaxis])


def dequantize_fp8_rows(q_u8: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """w ~ e4m3_to_f32(w_q) * scale, row-wise."""
    return e4m3_to_f32(q_u8) * scales[..., np.newaxis]


# ---------------------------------------------------------------------------
# safetensors self-parsing
# ---------------------------------------------------------------------------


class SafeTensorFile:
    """Read-only access to a .safetensors file via np.memmap (uint16 views)."""

    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        self.file_size = os.path.getsize(self.path)
        with open(self.path, "rb") as fh:
            (hlen,) = struct.unpack("<Q", fh.read(8))
            if 8 + hlen > self.file_size:
                raise ValueError(f"{self.path}: header length {hlen} exceeds file")
            self.header = json.loads(fh.read(hlen))
        self.data_start = 8 + hlen
        self._mm = np.memmap(self.path, dtype=np.uint16, mode="r", shape=(self.file_size // 2,))

    def tensor(self, name: str) -> tuple[np.ndarray, tuple[int, ...], int, int]:
        """Return (uint16 view, shape, byte_offset, byte_len) for a tensor."""
        entry = self.header.get(name)
        if entry is None:
            raise KeyError(f"tensor not found: {name}")
        off, end = entry["data_offsets"]
        shape = tuple(entry["shape"])
        n_elem = 1
        for d in shape:
            n_elem *= d
        if end - off != n_elem * 2:
            raise ValueError(f"{name}: data_offsets span {end-off} != {n_elem*2} bytes (not bf16?)")
        if entry.get("dtype") not in ("BF16", "bfloat16"):
            raise ValueError(f"{name}: unexpected dtype {entry.get('dtype')}")
        view = self._mm[(self.data_start + off) // 2:(self.data_start + end) // 2].reshape(shape)
        return view, shape, self.data_start + off, end - off


def sha256_file(path: str, chunk_bytes: int = 1 << 23) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk_bytes)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Section layout spec: (section_name, shape, [(source_key, row0, row_count)])
# Row-concatenation order is exactly the task table order.
# ---------------------------------------------------------------------------


def build_section_spec(num_layers: int = NUM_LAYERS) -> list[tuple[str, tuple[int, ...], list[tuple[str, int, int]]]]:
    H, I, V = HIDDEN, INTER, VOCAB
    secs: list[tuple[str, tuple[int, ...], list[tuple[str, int, int]]]] = [
        ("embedding", (V, H), [("model.embed_tokens.weight", 0, V)]),
        ("lm_head", (V, H), [("lm_head.weight", 0, V)]),
        ("final_norm", (H,), [("model.norm.weight", 0, 1)]),
    ]
    for i in range(num_layers):
        p = f"model.layers.{i}."
        secs += [
            (f"layer{i}.norm1", (H,), [(p + "input_layernorm.weight", 0, 1)]),
            (f"layer{i}.qkv", (2560, H), [
                (p + "self_attn.q_proj.weight", 0, 2048),
                (p + "self_attn.k_proj.weight", 0, 256),
                (p + "self_attn.v_proj.weight", 0, 256),
            ]),
            (f"layer{i}.o", (H, H), [(p + "self_attn.o_proj.weight", 0, H)]),
            (f"layer{i}.norm2", (H,), [(p + "post_attention_layernorm.weight", 0, 1)]),
            (f"layer{i}.gate_up", (2 * I, H), [
                (p + "mlp.gate_proj.weight", 0, I),
                (p + "mlp.up_proj.weight", 0, I),
            ]),
            (f"layer{i}.down", (H, I), [(p + "mlp.down_proj.weight", 0, H)]),
        ]
    return secs


def expected_source_keys(spec: list) -> set[str]:
    keys: set[str] = set()
    for _name, _shape, srcs in spec:
        for k, _r0, _rn in srcs:
            keys.add(k)
    return keys


def align_up(x: int, a: int) -> int:
    return (x + a - 1) // a * a


# ---------------------------------------------------------------------------
# WPK header / TOC (de)serialization
# ---------------------------------------------------------------------------


def pack_header(model_hash: bytes, section_count: int, data_start: int) -> bytes:
    assert len(model_hash) == 32
    return struct.pack(
        WPK_HEADER_FMT,
        WPK_MAGIC,
        WPK_ABI_VERSION,
        WPK_TARGET_SM,
        model_hash,
        section_count,
        0,                      # reserved u32
        WPK_HEADER_SIZE,        # toc_offset = 64
        data_start,
    )


def pack_toc_entry(name: str, shape: tuple[int, ...], data_offset: int, data_bytes: int,
                   dtype: int = DTYPE_BF16, scale_offset: int = 0, scale_bytes: int = 0) -> bytes:
    dims = list(shape) + [0] * (8 - len(shape))
    return struct.pack(
        WPK_TOC_FMT,
        fnv1a64(name.encode("utf-8")),
        dtype,
        LAYOUT_ROW_MAJOR,
        len(shape),
        *dims,
        data_offset,
        data_bytes,
        scale_offset,
        scale_bytes,         # FP8 sections: fp32 [N] row scales live here
        0, 0, 0, 0, 0, 0,    # reserved u64[6]
    )


class WpkReader:
    """Parsed view of a model.wpk (header + TOC). Data accessed via memmap."""

    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        self.file_size = os.path.getsize(self.path)
        with open(self.path, "rb") as fh:
            raw = fh.read(WPK_HEADER_SIZE)
            (self.magic, self.abi_version, self.target_sm, self.model_hash,
             self.section_count, self.reserved, self.toc_offset,
             self.data_start) = struct.unpack(WPK_HEADER_FMT, raw)
            if self.magic != WPK_MAGIC:
                raise ValueError(f"{self.path}: bad magic 0x{self.magic:08X}")
            if self.abi_version != WPK_ABI_VERSION:
                raise ValueError(f"{self.path}: abi 0x{self.abi_version:04X} != 0x{WPK_ABI_VERSION:04X}")
            if self.target_sm != WPK_TARGET_SM:
                raise ValueError(f"{self.path}: target_sm {self.target_sm} != {WPK_TARGET_SM}")
            if self.toc_offset != WPK_HEADER_SIZE:
                raise ValueError(f"{self.path}: toc_offset {self.toc_offset} != {WPK_HEADER_SIZE}")
            toc_raw = fh.read(self.section_count * WPK_TOC_ENTRY_SIZE)
        self.sections: dict[str, dict] = {}
        self._by_hash: dict[int, dict] = {}
        for i in range(self.section_count):
            (name_hash, dtype, layout, rank, *rest) = struct.unpack_from(WPK_TOC_FMT, toc_raw, i * WPK_TOC_ENTRY_SIZE)
            dims = tuple(rest[:8])
            data_offset, data_bytes, scale_offset, scale_bytes = rest[8:12]
            sec = {
                "index": i,
                "name_hash": name_hash,
                "dtype": dtype,
                "layout": layout,
                "rank": rank,
                "dims": dims,
                "data_offset": data_offset,
                "data_bytes": data_bytes,
                "scale_offset": scale_offset,
                "scale_bytes": scale_bytes,
            }
            self._by_hash[name_hash] = sec
        self._mm = np.memmap(self.path, dtype=np.uint16, mode="r", shape=(self.file_size // 2,))
        self._mm8 = None       # lazy uint8 view (FP8 sections)
        # name -> section resolution happens lazily via spec names.

    def section(self, name: str) -> dict:
        sec = self._by_hash.get(fnv1a64(name.encode("utf-8")))
        if sec is None:
            raise KeyError(f"wpk section not found: {name}")
        return sec

    def array(self, name: str) -> np.ndarray:
        """uint16 view of a BF16 section, reshaped by its dims."""
        sec = self.section(name)
        if sec["dtype"] != DTYPE_BF16:
            raise ValueError(f"{self.path}: section {name} dtype {sec['dtype']} != BF16")
        shape = tuple(d for d in sec["dims"][: sec["rank"]])
        n_elem = sec["data_bytes"] // 2
        view = self._mm[sec["data_offset"] // 2: sec["data_offset"] // 2 + n_elem]
        return view.reshape(shape)

    def fp8(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """(q_uint8 [N,K], scales_f32 [N]) views for an FP8_E4M3 section."""
        sec = self.section(name)
        if sec["dtype"] != DTYPE_FP8_E4M3:
            raise ValueError(f"{self.path}: section {name} dtype {sec['dtype']} != FP8_E4M3")
        if sec["rank"] != 2:
            raise ValueError(f"{name}: fp8 section rank {sec['rank']} != 2")
        n, k = sec["dims"][0], sec["dims"][1]
        if sec["data_bytes"] != n * k or sec["scale_bytes"] != 4 * n:
            raise ValueError(f"{name}: fp8 byte accounting mismatch "
                             f"(data {sec['data_bytes']} != {n*k} or scales {sec['scale_bytes']} != {4*n})")
        if self._mm8 is None:
            self._mm8 = np.memmap(self.path, dtype=np.uint8, mode="r", shape=(self.file_size,))
        q = self._mm8[sec["data_offset"]: sec["data_offset"] + n * k].reshape(n, k)
        sc = self._mm8[sec["scale_offset"]: sec["scale_offset"] + 4 * n].view(np.float32)
        return q, sc
