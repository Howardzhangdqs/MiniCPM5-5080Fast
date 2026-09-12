#!/usr/bin/env python3
"""reference_forward.py -- pure-numpy BF16 reference forward for MiniCPM5-2B.

Golden generator for runtime numerics validation (no torch dependency).
Numerical contract (mirrors the GPU-side spec exactly):

  storage BF16, compute FP32
  * Linear       : fp32 matmul(upcast W, upcast x) -> round RNE to bf16
  * RMSNorm      : ms = mean(x^2) + 1e-6; y = x * rsqrt(ms) * upcast(w) -> bf16
  * residual add : fp32(a) + fp32(b) -> bf16
  * RoPE         : HF llama rotate_half (non-interleaved), theta = 5e6,
                   inv_freq = theta^(-2i/128), i = 0..63; angles in fp32;
                   q -> bf16, k -> bf16 (written to the all-bf16 KV cache)
  * Attention    : GQA 16q/2kv (q head h uses kv head h//8);
                   score = fp32(q) . fp32(k) / sqrt(128); softmax in fp32;
                   out = sum(p * fp32(v)) -> bf16; causal (prefill: j <= i)
  * SwiGLU       : silu(fp32 g) * fp32(u) -> bf16
  * lm_head      : logits -> bf16; greedy argmax on upcast fp32 (ties -> lower id)
  * embedding    : bf16 lookup, passed through as-is

Golden step numbering (native side uses the same definition):
  prefill(T)  -> last-position logits = step0.logits, argmax = step0.token
  step{t}     = output after consuming the t-th generated token (t >= 1)
  stop after step{t} if step{t}.token in {1, 130073} (meta.eos_step = t)

Output layout (tests/runtime/golden/<case>/):
  meta.json, input_ids.bin (i32 LE), step{t}.token (i32 LE file),
  step{t}.logits.f32 (130560 fp32 = bf16 logits upcast),
  step{t}.final_hidden.f32 (2048 fp32; after final rmsnorm, before lm_head),
  and for step0 only: layer{i}.hidden_in.f32 / layer{i}.attn_out.f32 /
  layer{i}.mlp_out.f32 (2048 fp32 each, last position; attn_out = after
  o-proj + residual, mlp_out = after down + residual).

Memory discipline: weights are mmap-backed uint16 views (model.wpk preferred,
safetensors fallback); fp32 upcasts happen per matrix per layer with a
temporary of at most ~100 MB (lm_head is upcast in vocab-row chunks).

Usage:
  python tools/reference_forward.py --model-dir models/minicpm5-2b --case all
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time

import numpy as np
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from minicpm_common import (  # noqa: E402
    BOS_ID,
    EOS_IDS,
    HEAD_DIM,
    HIDDEN,
    INTER,
    MODEL_NAME,
    N_KV_HEADS,
    N_Q_HEADS,
    NUM_LAYERS,
    RMS_EPS,
    ROPE_THETA,
    SafeTensorFile,
    VOCAB,
    bf16_to_f32,
    build_section_spec,
    e4m3_to_f32,
    f32_to_bf16,
)

# ---------------------------------------------------------------------------
# weight access: mmap uint16 views, wpk preferred, safetensors fallback
# ---------------------------------------------------------------------------


class WeightStore:
    def __init__(self, model_dir: str, prefer: str = "auto"):
        self.source = None
        self.r8 = None            # fp8 pack reader (mixed mode only)
        wpk = os.path.join(model_dir, "model.wpk")
        wpk8 = os.path.join(model_dir, "model_fp8.wpk")
        st = os.path.join(model_dir, "model-00000-of-00001.safetensors")
        if prefer == "fp8":
            # P4 closeout mixed mode: prefill reads BF16 from model.wpk, decode
            # steps read dequantized FP8 from model_fp8.wpk (168 layer sections
            # + lm_head); norms/embedding are BF16 in the pack. lm_head path
            # mirrors native forward.cpp: prefill (T>1) BF16, decode (T==1)
            # dequantized FP8 (pack carries the dtype=2 lm_head section).
            if not os.path.exists(wpk):
                raise FileNotFoundError(f"mixed fp8 mode needs {wpk}")
            if not os.path.exists(wpk8):
                raise FileNotFoundError(f"mixed fp8 mode needs {wpk8}")
            from minicpm_common import WpkReader

            self.r = WpkReader(wpk)
            self.r8 = WpkReader(wpk8)
            self.source = "wpk+fp8"
        elif prefer in ("auto", "wpk") and os.path.exists(wpk):
            from minicpm_common import WpkReader

            self.r = WpkReader(wpk)
            self.source = "wpk"
        elif prefer in ("auto", "safetensors") and os.path.exists(st):
            self.stf = SafeTensorFile(st)
            self.source = "safetensors"
        else:
            raise FileNotFoundError(f"no model.wpk nor safetensors under {model_dir}")

    # -- canonical section accessors -----------------------------------------
    def matrix(self, name: str) -> np.ndarray:
        """[out, in] uint16 view (fused sections come back as one matrix)."""
        if self.source in ("wpk", "wpk+fp8"):
            return self.r.array(name)
        parts = {
            "embedding": ["model.embed_tokens.weight"],
            "lm_head": ["lm_head.weight"],
        }
        p = name.split(".", 1)
        if p[0].startswith("layer"):
            i = p[0][5:]
            pre = f"model.layers.{i}."
            parts.update({
                f"layer{i}.qkv": [pre + "self_attn.q_proj.weight",
                                  pre + "self_attn.k_proj.weight",
                                  pre + "self_attn.v_proj.weight"],
                f"layer{i}.o": [pre + "self_attn.o_proj.weight"],
                f"layer{i}.gate_up": [pre + "mlp.gate_proj.weight", pre + "mlp.up_proj.weight"],
                f"layer{i}.down": [pre + "mlp.down_proj.weight"],
            })
        views = [self.stf.tensor(k)[0] for k in parts[name]]
        return views[0] if len(views) == 1 else np.concatenate(views, axis=0)

    def vector(self, name: str) -> np.ndarray:
        """rank-1 uint16 view (norms)."""
        if self.source in ("wpk", "wpk+fp8"):
            return self.r.array(name)
        keys = {"final_norm": "model.norm.weight"}
        if name.startswith("layer"):
            i, kind = name[5:].split(".")
            keys[name] = (f"model.layers.{i}.input_layernorm.weight" if kind == "norm1"
                          else f"model.layers.{i}.post_attention_layernorm.weight")
        return self.stf.tensor(keys[name])[0]

    def matrix_fp8(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """(q_uint8 [N,K], scales_f32 [N]) views from the fp8 pack (mixed mode)."""
        assert self.source == "wpk+fp8", "fp8 weights require --weights fp8"
        return self.r8.fp8(name)


# ---------------------------------------------------------------------------
# numeric kernels (bf16 storage / fp32 compute)
# ---------------------------------------------------------------------------


def linear_bf16(w_u16: np.ndarray, x_u16: np.ndarray) -> np.ndarray:
    """y = x @ W^T in fp32, stored back to bf16. x: [T,in] or [in]."""
    w32 = bf16_to_f32(w_u16)          # [out, in] (temp <= ~100 MB for gate_up)
    x32 = bf16_to_f32(x_u16)
    return f32_to_bf16(x32 @ w32.T)


def linear_fp8(q_u8: np.ndarray, scales: np.ndarray, x_u16: np.ndarray) -> np.ndarray:
    """Decode step linear on dequantized FP8 weights, mirroring the native
    gemv_fp8 kernel exactly: acc = sum_k e4m3(w[n,k]) * f32(x[k]) in fp32,
    then y[n] = bf16(acc * scale[n]) -- the row scale multiplies once AFTER
    accumulation (gemv_fp8.cu: y = __float2bfloat16(acc * scale[warp]))."""
    x32 = bf16_to_f32(x_u16)
    acc = x32 @ e4m3_to_f32(q_u8).T                           # fp32 [T,N]
    return f32_to_bf16(acc * scales)                          # broadcast per row


def rmsnorm_bf16(w_u16: np.ndarray, x_u16: np.ndarray) -> np.ndarray:
    x32 = bf16_to_f32(x_u16)
    ms = np.mean(x32 * x32, axis=-1, keepdims=True) + np.float32(RMS_EPS)
    y = x32 * (np.float32(1.0) / np.sqrt(ms)) * bf16_to_f32(w_u16)
    return f32_to_bf16(y)


def add_bf16(a_u16: np.ndarray, b_u16: np.ndarray) -> np.ndarray:
    return f32_to_bf16(bf16_to_f32(a_u16) + bf16_to_f32(b_u16))


# RoPE (HF llama rotate_half, non-interleaved), fp32 angles
_INV_FREQ = (ROPE_THETA ** (-np.arange(0, HEAD_DIM, 2, dtype=np.float64) / HEAD_DIM)).astype(np.float32)


def rope_cos_sin(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ang = np.outer(positions.astype(np.float32), _INV_FREQ)      # [T, 64] fp32
    return (np.concatenate([np.cos(ang), np.cos(ang)], axis=-1),   # [T, 128]
            np.concatenate([np.sin(ang), np.sin(ang)], axis=-1))


def rotate_half(x: np.ndarray) -> np.ndarray:
    h = x.shape[-1] // 2
    return np.concatenate([-x[..., h:], x[..., :h]], axis=-1)


def rope_bf16(x_u16: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """x: [T, n_head, 128] bf16 -> roped bf16 (fp32 math)."""
    cos, sin = rope_cos_sin(positions)
    x32 = bf16_to_f32(x_u16)
    out = x32 * cos[:, None, :] + rotate_half(x32) * sin[:, None, :]
    return f32_to_bf16(out)


def softmax_f32(s: np.ndarray) -> np.ndarray:
    m = np.max(s, axis=-1, keepdims=True)
    e = np.exp(s - m)
    return e / np.sum(e, axis=-1, keepdims=True)


def attention_bf16(q_u16: np.ndarray, k_cache: np.ndarray, v_cache: np.ndarray,
                   layer: int, kv_len: int, pos_start: int, causal: bool) -> np.ndarray:
    """q: [T, 16, 128] bf16; k/v caches: [cap, 256] bf16 (2 kv heads)."""
    T = q_u16.shape[0]
    q32 = bf16_to_f32(q_u16)                                     # [T,16,128]
    k32 = bf16_to_f32(k_cache[layer][:kv_len]).reshape(kv_len, N_KV_HEADS, HEAD_DIM)
    v32 = bf16_to_f32(v_cache[layer][:kv_len]).reshape(kv_len, N_KV_HEADS, HEAD_DIM)
    kE = np.repeat(k32, N_Q_HEADS // N_KV_HEADS, axis=1)         # [n,16,128]
    vE = np.repeat(v32, N_Q_HEADS // N_KV_HEADS, axis=1)
    scale = np.float32(1.0 / np.sqrt(np.float32(HEAD_DIM)))
    s = np.einsum("thd,nhd->htn", q32, kE) * scale               # fp32 scores
    if causal:  # prefill: allow key j iff j <= pos_start + t
        j = np.arange(kv_len)[None, :]
        t = (pos_start + np.arange(T))[:, None]
        s = np.where(j <= t, s, np.float32(-np.inf))
    p = softmax_f32(s)                                           # fp32 softmax
    o = np.einsum("htn,nhd->thd", p, vE)                         # fp32 weighted sum
    return f32_to_bf16(o.reshape(T, N_Q_HEADS * HEAD_DIM))


def swiglu_bf16(g_u16: np.ndarray, u_u16: np.ndarray) -> np.ndarray:
    g32 = bf16_to_f32(g_u16)
    return f32_to_bf16(g32 * (np.float32(1.0) / (np.float32(1.0) + np.exp(-g32))) * bf16_to_f32(u_u16))


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


class MiniCPM5Ref:
    def __init__(self, store: WeightStore):
        self.store = store
        self.Wemb = None  # lazy memmap view

    def embedding(self, ids: list[int]) -> np.ndarray:
        emb = self.store.matrix("embedding")
        return emb[np.asarray(ids, dtype=np.int64)]              # bf16 rows as-is

    def forward(self, ids: list[int], kv_k, kv_v, kv_len, pos_start: int,
                collect_layers: bool = False, use_fp8: bool = False):
        """One forward over `ids` placed at absolute positions
        pos_start..pos_start+T-1 with a KV cache already holding kv_len entries.
        use_fp8: decode-path weights come dequantized from the fp8 pack
        (P4 closeout mixed mode): qkv/o/gate_up/down AND lm_head; norms and
        embedding stay BF16 (native mirrors: gemv_fp8 for all M=1 fp8
        sections; prefill always BF16).
        Returns (logits_u16[V], final_hidden_u16[H], layer_records)."""
        T = len(ids)
        causal = T > 1
        x = self.embedding(ids)                                  # [T,H] bf16
        recs: list[tuple[int, str, np.ndarray]] = []

        def lin(name: str, h: np.ndarray) -> np.ndarray:
            if use_fp8:
                q, sc = self.store.matrix_fp8(name)
                return linear_fp8(q, sc, h)
            return linear_bf16(self.store.matrix(name), h)

        for i in range(NUM_LAYERS):
            if collect_layers:
                recs.append((i, "hidden_in", np.array(x[-1])))
            h = rmsnorm_bf16(self.store.vector(f"layer{i}.norm1"), x)
            qkv = lin(f"layer{i}.qkv", h)                        # [T,2560] bf16
            q = qkv[:, :N_Q_HEADS * HEAD_DIM].reshape(T, N_Q_HEADS, HEAD_DIM)
            k = qkv[:, N_Q_HEADS * HEAD_DIM:N_Q_HEADS * HEAD_DIM + N_KV_HEADS * HEAD_DIM]
            v = qkv[:, -N_KV_HEADS * HEAD_DIM:]
            positions = np.arange(pos_start, pos_start + T, dtype=np.int64)
            q = rope_bf16(q, positions)                          # bf16
            k = rope_bf16(k.reshape(T, N_KV_HEADS, HEAD_DIM), positions).reshape(T, -1)
            kv_k[i][kv_len:kv_len + T] = k                       # bf16 into cache
            kv_v[i][kv_len:kv_len + T] = v
            attn = attention_bf16(q, kv_k, kv_v, i, kv_len + T, pos_start, causal)
            ao = lin(f"layer{i}.o", attn)                        # O proj
            x = add_bf16(x, ao)                                  # residual
            if collect_layers:
                recs.append((i, "attn_out", np.array(x[-1])))
            h2 = rmsnorm_bf16(self.store.vector(f"layer{i}.norm2"), x)
            gu = lin(f"layer{i}.gate_up", h2)                    # [T,12288]
            act = swiglu_bf16(gu[:, :INTER], gu[:, INTER:])
            d = lin(f"layer{i}.down", act)
            x = add_bf16(x, d)
            if collect_layers:
                recs.append((i, "mlp_out", np.array(x[-1])))

        hf = rmsnorm_bf16(self.store.vector("final_norm"), x)    # [T,H] bf16
        logits = self.lm_head(hf[-1], use_fp8=use_fp8)           # bf16 [V]
        return logits, np.array(hf[-1]), recs

    def lm_head(self, h_u16: np.ndarray, use_fp8: bool = False) -> np.ndarray:
        """Chunked fp32 matmul over vocab rows (keeps fp32 temp ~67 MB).

        use_fp8 mirrors the native decode path (forward.cpp): when the fp8 pack
        carries an lm_head dtype=2 section, decode (T==1) lm_head runs
        launch_gemv_fp8 -- per-vocab-row acc = sum_k e4m3(w) * f32(h[k]) in
        fp32, then bf16(acc * scale[row]). Prefill (step0, T>1) always uses
        the BF16 original weights (gemm on w.lm_head)."""
        h32 = bf16_to_f32(h_u16)
        out = np.empty(VOCAB, dtype=np.uint16)
        chunk = 8192
        if use_fp8:
            q, sc = self.store.matrix_fp8("lm_head")
            for s in range(0, VOCAB, chunk):
                acc = e4m3_to_f32(q[s:s + chunk]) @ h32      # fp32 [chunk]
                out[s:s + chunk] = f32_to_bf16(acc * sc[s:s + chunk])
            return out
        w = self.store.matrix("lm_head")
        for s in range(0, VOCAB, chunk):
            w32 = bf16_to_f32(w[s:s + chunk])
            out[s:s + chunk] = f32_to_bf16(w32 @ h32)
        return out

    @staticmethod
    def greedy(logits_u16: np.ndarray) -> int:
        """argmax over upcast fp32; ties resolve to the smaller id (argmax
        returns the first occurrence)."""
        return int(np.argmax(bf16_to_f32(logits_u16)))


# ---------------------------------------------------------------------------
# golden generation
# ---------------------------------------------------------------------------

CASES = {
    "case_zh": {"prompt": "你好，请介绍一下你自己。", "steps": 32},
    "case_en": {"prompt": "Write a Python function to compute fibonacci numbers.", "steps": 32},
    "case_mixed": {"prompt": "用中文解释什么是 CUDA graph，then list 3 pros in English.", "steps": 16},
}


def write_f32(path: str, u16: np.ndarray) -> None:
    bf16_to_f32(u16).astype("<f4").tofile(path)


def generate_case(model: MiniCPM5Ref, tokenizer: Tokenizer, case: str, prompt: str,
                  steps: int, out_root: str, ids_override: list | None = None,
                  mixed_fp8: bool = False) -> dict:
    out_dir = os.path.join(out_root, case)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.perf_counter()

    if ids_override is not None:
        ids = list(ids_override)
    else:
        ids = [BOS_ID] + tokenizer.encode(prompt, add_special_tokens=False).ids
    T = len(ids)
    max_pos = T + steps + 8
    kv_k = [np.zeros((max_pos, N_KV_HEADS * HEAD_DIM), dtype=np.uint16) for _ in range(NUM_LAYERS)]
    kv_v = [np.zeros((max_pos, N_KV_HEADS * HEAD_DIM), dtype=np.uint16) for _ in range(NUM_LAYERS)]

    np.asarray(ids, dtype="<i4").tofile(os.path.join(out_dir, "input_ids.bin"))

    tokens: list[int] = []
    eos_step = None

    # step0: prefill -> next-token logits at the last position.
    # Mixed mode: prefill always uses the original BF16 weights -- including
    # lm_head (native forward.cpp routes the T>1 lm_head through the bf16
    # gemm on the last row even when the fp8 lm_head section is loaded).
    logits, hidden, recs = model.forward(ids, kv_k, kv_v, kv_len=0, pos_start=0,
                                         collect_layers=True, use_fp8=False)
    tok = model.greedy(logits)
    write_f32(os.path.join(out_dir, "step0.logits.f32"), logits)
    write_f32(os.path.join(out_dir, "step0.final_hidden.f32"), hidden)
    with open(os.path.join(out_dir, "step0.token"), "wb") as fh:
        fh.write(struct.pack("<i", tok))
    for i, kind, vec in recs:
        write_f32(os.path.join(out_dir, f"layer{i}.{kind}.f32"), vec)
    tokens.append(tok)
    if tok in EOS_IDS:
        eos_step = 0
        steps_done = 1
    else:
        steps_done = 1
        for t in range(1, steps):
            logits, hidden, _ = model.forward([tok], kv_k, kv_v, kv_len=T + t - 1,
                                              pos_start=T + t - 1, collect_layers=False,
                                              use_fp8=mixed_fp8)
            tok = model.greedy(logits)
            write_f32(os.path.join(out_dir, f"step{t}.logits.f32"), logits)
            write_f32(os.path.join(out_dir, f"step{t}.final_hidden.f32"), hidden)
            with open(os.path.join(out_dir, f"step{t}.token"), "wb") as fh:
                fh.write(struct.pack("<i", tok))
            tokens.append(tok)
            steps_done = t + 1
            if tok in EOS_IDS:
                eos_step = t
                break

    meta = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "input_ids": ids,
        "steps": steps_done,
        "eos_step": eos_step,
        "dtype": ("bf16-prefill + fp8-e4m3-decode (incl. lm_head) / fp32acc "
                  "(P4 closeout mixed)" if mixed_fp8 else "bf16-fp32acc"),
        "weights": "fp8-mixed" if mixed_fp8 else "bf16",
        "lm_head": {
            "prefill_step0": "bf16",
            "decode_steps": "fp8-dequant (e4m3->f32 dot, fp32 acc, scale "
                            "applied post-accumulation, single RNE to bf16)"
                            if mixed_fp8 else "bf16",
            "mirror_of": "native forward.cpp lm_head tail: T>1 -> gemm(1, V, H, "
                         "w.lm_head, normed_last_row) [BF16]; T==1 && "
                         "fp8_lm_head present -> launch_gemv_fp8 (fused "
                         "prologue variant bit-identical); MC_FP8_LMHEAD_OFF=1 "
                         "or absent dtype=2 section -> BF16",
        },
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    dt = time.perf_counter() - t0
    text = tokenizer.decode(tokens)
    return {"case": case, "dir": out_dir, "input_len": T, "steps": steps_done,
            "eos_step": eos_step, "tokens": tokens, "text": text, "seconds": dt}


def main() -> None:
    ap = argparse.ArgumentParser(description="MiniCPM5-2B numpy BF16 golden generator")
    ap.add_argument("--model-dir", default="models/minicpm5-2b")
    ap.add_argument("--case", default="all", choices=["all"] + list(CASES))
    ap.add_argument("--out", default="tests/runtime/golden")
    ap.add_argument("--weights", default="auto",
                    choices=["auto", "wpk", "safetensors", "fp8"],
                    help="fp8 = P4 closeout mixed mode: prefill from model.wpk "
                         "(BF16), decode steps dequantized from model_fp8.wpk "
                         "(168 layer sections + lm_head)")
    ap.add_argument("--ids-file", default=None,
                    help="直接使用文件中的 i32 token id（跳过 prompt 分词；用于交叉验证）")
    ap.add_argument("--steps", type=int, default=None, help="覆盖用例步数（配合 --ids-file）")
    args = ap.parse_args()

    store = WeightStore(args.model_dir, prefer=args.weights)
    print(f"[ref] weights source: {store.source}")
    if store.source == "wpk+fp8":
        from minicpm_common import DTYPE_BF16, DTYPE_FP8_E4M3
        spec = build_section_spec()
        assert store.r.section_count == len(spec) == 255
        assert store.r8.section_count == len(spec) == 255
        quant = {f"layer{i}.{k}" for i in range(NUM_LAYERS)
                 for k in ("qkv", "o", "gate_up", "down")} | {"lm_head"}
        n8 = n16 = 0
        for name, shape, _srcs in spec:
            arr = store.r.array(name)             # bf16 pack: shape check
            assert arr.shape == shape, f"{name}: {arr.shape} != {shape}"
            sec = store.r8.section(name)
            if name in quant:
                q, sc = store.r8.fp8(name)
                assert q.shape == shape and sc.shape == (shape[0],)
                assert sec["dtype"] == DTYPE_FP8_E4M3
                n8 += 1
            else:
                assert sec["dtype"] == DTYPE_BF16 and sec["scale_bytes"] == 0
                n16 += 1
        assert n8 == 169 and n16 == 86
        lm_sec = store.r8.section("lm_head")
        assert lm_sec["dtype"] == DTYPE_FP8_E4M3, \
            "mixed fp8 mode requires the lm_head fp8 section (decode lm_head mirror)"
        print(f"[ref] fp8 pack verified: 255 sections ({n8} fp8 incl. lm_head + "
              f"{n16} bf16), shapes OK; prefill=bf16 weights, decode=dequantized "
              f"fp8 weights (layers + lm_head)")
    elif store.source == "wpk":
        spec = build_section_spec()
        assert store.r.section_count == len(spec)
        # sanity: every expected section resolvable and byte-sized correctly
        for name, shape, _srcs in ((n, s, x) for n, s, x in spec):
            arr = store.r.array(name)
            assert arr.shape == shape, f"{name}: {arr.shape} != {shape}"
        print(f"[ref] wpk sections verified: {len(spec)} x shape OK")

    tokenizer = Tokenizer.from_file(os.path.join(args.model_dir, "tokenizer.json"))
    model = MiniCPM5Ref(store)

    if args.ids_file:
        ids = np.fromfile(args.ids_file, dtype="<i4").tolist()
        steps = args.steps or 8
        r = generate_case(model, tokenizer, "case_ids", "", steps, args.out, ids_override=ids)
        print(f"[ref] case_ids: input_len={r['input_len']} steps={r['steps']} "
              f"eos_step={r['eos_step']} ({r['seconds']:.1f}s)")
        print(f"[ref] case_ids tokens: {r['tokens']}")
        return

    cases = list(CASES) if args.case == "all" else [args.case]
    mixed = store.source == "wpk+fp8"
    for case in cases:
        r = generate_case(model, tokenizer, case, CASES[case]["prompt"], CASES[case]["steps"],
                          args.out, mixed_fp8=mixed)
        print(f"[ref] {r['case']}: input_len={r['input_len']} steps={r['steps']} "
              f"eos_step={r['eos_step']} ({r['seconds']:.1f}s)")
        print(f"[ref] {r['case']} tokens: {r['tokens']}")
        print(f"[ref] {r['case']} text: {r['text']}")


if __name__ == "__main__":
    main()
