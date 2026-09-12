#!/usr/bin/env python3
"""quantize.py -- WPK BF16 -> WPK mixed BF16/FP8-E4M3 quantizer (plan P4 v1).

Reads  <model-dir>/model.wpk                    (BF16 pack; left untouched)
Writes <model-dir>/model_fp8.wpk                (mixed BF16/FP8 pack)
       <model-dir>/model_manifest_fp8.json       (provenance manifest)

Spec (field-exact with the native lane; do not modify):
  * format FP8 E4M3: RNE, saturate |x| > 448 to +/-448 (never NaN encodings)
  * granularity: per output row of the [N,K] weight: scale = max|row|/448
    (floor 1e-12); w_q = e4m3(w/scale); w ~ e4m3_to_f32(w_q) * scale
  * quantized sections: 42 layers x {qkv, o, gate_up, down} + lm_head = 169,
    dtype=2, data = uint8 [N,K], scales = fp32 [N] in the existing
    scale_offset / scale_bytes TOC fields. lm_head fp8 (P4 closeout): the
    native loader consumes an lm_head dtype=2 section -> decode lm_head runs
    gemv_fp8; prefill lm_head stays on the BF16 original weights.
  * kept BF16 (byte-for-byte copy, dtype=0): embedding, final_norm,
    layer{i}.norm1, layer{i}.norm2 = 86 sections
  * container: WPK v1 -- header 64B, TOC 132B stride, data 256B aligned,
    abi 0x0002, target_sm 120

Self-verification (after writing): re-parse the fp8 pack, assert 169+86=255
sections with correct dtypes, byte-compare every BF16 section against the
source pack, and dequantize 7 quantized sections x 4 rows vs the BF16
originals recording per-row relative L2 error (incl. lm_head).

Usage:
  /workspace/.venv/bin/python tools/quantize.py --model-dir models/minicpm5-2b
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from minicpm_common import (  # noqa: E402
    DTYPE_BF16,
    DTYPE_FP8_E4M3,
    LAYOUT_ROW_MAJOR,
    MODEL_NAME,
    MODEL_REVISION,
    NUM_LAYERS,
    WPK_ABI_VERSION,
    WPK_ALIGN,
    WPK_HEADER_SIZE,
    WPK_TARGET_SM,
    WPK_TOC_ENTRY_SIZE,
    WpkReader,
    align_up,
    bf16_to_f32,
    build_section_spec,
    dequantize_fp8_rows,
    fp8_row_scales,
    fnv1a64,
    pack_header,
    pack_toc_entry,
    quantize_fp8_rows,
    sha256_file,
)

QUANT_KINDS = ("qkv", "o", "gate_up", "down")
EXTRA_QUANT_SECTIONS = ("lm_head",)   # P4 closeout: 169th fp8 section
CHUNK_COPY = 1 << 24            # 16 MiB raw-copy chunks (bf16 sections)
QUANT_ROW_CHUNK = 2048          # rows per encode chunk (bounds fp32 temporaries;
                                # lm_head [130560,2048] streams in 16MB chunks)

SPOT_SECTIONS = ["layer0.qkv", "layer10.o", "layer20.gate_up", "layer30.down",
                 "layer41.qkv", "layer41.gate_up", "lm_head"]


def quantized_section_names(num_layers: int = NUM_LAYERS) -> set[str]:
    return {f"layer{i}.{k}" for i in range(num_layers) for k in QUANT_KINDS} \
        | set(EXTRA_QUANT_SECTIONS)


def is_quantized(name: str, quant_set: set[str]) -> bool:
    return name in quant_set


def build_fp8_layout(spec, quant_set: set[str]):
    """Compute (data_start, entries, total_file). entries: list of dicts."""
    toc_end = WPK_HEADER_SIZE + len(spec) * WPK_TOC_ENTRY_SIZE
    data_start = align_up(toc_end, WPK_ALIGN)
    cur = data_start
    entries = []
    for name, shape, _srcs in spec:
        if is_quantized(name, quant_set):
            assert len(shape) == 2, f"{name}: quantized section must be rank 2"
            n, k = shape
            assert k % 4 == 0, f"{name}: K={k} not 4B-divisible (scale alignment)"
            dtype = DTYPE_FP8_E4M3
            data_bytes = n * k
            scale_offset = cur + data_bytes       # 4B aligned (k % 4 == 0)
            scale_bytes = 4 * n
            end = scale_offset + scale_bytes
        else:
            dtype = DTYPE_BF16
            data_bytes = 2 * int(np.prod(shape))
            scale_offset = 0
            scale_bytes = 0
            end = cur + data_bytes
        entries.append({
            "name": name, "shape": tuple(shape), "dtype": dtype,
            "data_offset": cur, "data_bytes": data_bytes,
            "scale_offset": scale_offset, "scale_bytes": scale_bytes,
        })
        cur = align_up(end, WPK_ALIGN)
    return data_start, entries, cur


def write_fp8_wpk(out_path: str, src: WpkReader, entries, model_hash: bytes,
                  data_start: int, quant_set: set[str]) -> tuple[float, dict]:
    """Stream the mixed pack out; returns (elapsed_s, byte accounting)."""
    t0 = time.perf_counter()
    tmp_path = out_path + ".tmp"
    acc = {"bf16": 0, "fp8": 0, "scales": 0, "pad": 0}
    scale_stats: list[dict] = []
    with open(tmp_path, "wb") as out, open(src.path, "rb") as sf:
        out.write(pack_header(model_hash, len(entries), data_start))
        for e in entries:
            out.write(pack_toc_entry(e["name"], e["shape"], e["data_offset"],
                                     e["data_bytes"], dtype=e["dtype"],
                                     scale_offset=e["scale_offset"],
                                     scale_bytes=e["scale_bytes"]))
        pos = WPK_HEADER_SIZE + len(entries) * WPK_TOC_ENTRY_SIZE
        if pos < data_start:
            out.write(b"\x00" * (data_start - pos))
            acc["pad"] += data_start - pos

        pos = data_start
        for e in entries:
            assert pos == e["data_offset"], f"layout desync at {e['name']}"
            name = e["name"]
            if e["dtype"] == DTYPE_BF16:
                # byte-for-byte搬运 from the source pack, chunked
                sf.seek(e["data_offset"])
                left = e["data_bytes"]
                while left:
                    n = min(left, CHUNK_COPY)
                    out.write(sf.read(n))
                    left -= n
                acc["bf16"] += e["data_bytes"]
            else:
                n, k = e["shape"]
                src_rows = src.array(name)                   # uint16 view [N,K]
                scales = np.empty(n, dtype="<f4")            # [N] row scales
                for s in range(0, n, QUANT_ROW_CHUNK):
                    ee = min(s + QUANT_ROW_CHUNK, n)
                    w32 = bf16_to_f32(src_rows[s:ee])        # fp32 chunk (<= ~50 MB)
                    scales[s:ee] = fp8_row_scales(w32)
                    q = quantize_fp8_rows(w32, scales[s:ee])
                    out.write(memoryview(q))                 # all q rows first ...
                    del w32, q
                out.write(scales)                            # ... scales after data
                acc["fp8"] += n * k
                acc["scales"] += scales.nbytes
                scale_stats.append({
                    "name": name, "shape": [n, k],
                    "scale_min": float(scales.min()),
                    "scale_max": float(scales.max()),
                    "scale_mean": float(scales.mean()),
                })
                del scales
            pos = e["scale_offset"] + e["scale_bytes"] if e["scale_bytes"] else \
                e["data_offset"] + e["data_bytes"]
            aligned = align_up(pos, WPK_ALIGN)
            if aligned > pos:
                out.write(b"\x00" * (aligned - pos))
                acc["pad"] += aligned - pos
                pos = aligned
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp_path, out_path)
    return time.perf_counter() - t0, {"bytes": acc, "scale_stats": scale_stats}


def verify_fp8_wpk(fp8_path: str, src: WpkReader, spec, quant_set: set[str]) -> dict:
    """Re-parse + self-verify; returns recorded stats (asserts on hard errors)."""
    r = WpkReader(fp8_path)
    assert r.magic == 0x31504B4D and r.abi_version == WPK_ABI_VERSION
    assert r.target_sm == WPK_TARGET_SM and r.toc_offset == WPK_HEADER_SIZE
    assert r.model_hash == src.model_hash, "model_hash differs from source pack"
    assert r.section_count == len(spec) == 255, \
        f"section_count {r.section_count} != 255 (169 fp8 + 86 bf16)"

    n_fp8 = n_bf16 = 0
    for i, (name, shape, _srcs) in enumerate(spec):
        sec = r.section(name)
        assert sec["index"] == i, f"{name}: TOC index {sec['index']} != {i}"
        assert sec["name_hash"] == fnv1a64(name.encode("utf-8"))
        assert sec["layout"] == LAYOUT_ROW_MAJOR and sec["rank"] == len(shape)
        assert sec["dims"][: len(shape)] == tuple(shape), f"{name}: dims mismatch"
        assert sec["data_offset"] % WPK_ALIGN == 0, f"{name}: not 256B aligned"
        if is_quantized(name, quant_set):
            n_fp8 += 1
            n, k = shape
            assert sec["dtype"] == DTYPE_FP8_E4M3, f"{name}: dtype != FP8_E4M3"
            assert sec["data_bytes"] == n * k, f"{name}: data_bytes != N*K"
            assert sec["scale_offset"] == sec["data_offset"] + n * k
            assert sec["scale_bytes"] == 4 * n, f"{name}: scale_bytes != 4N"
        else:
            n_bf16 += 1
            assert sec["dtype"] == DTYPE_BF16, f"{name}: dtype != BF16"
            assert sec["data_bytes"] == 2 * int(np.prod(shape))
            assert sec["scale_offset"] == 0 and sec["scale_bytes"] == 0
    assert n_fp8 == 169 and n_bf16 == 86, f"{n_fp8}+{n_bf16} != 169+86"

    # byte-compare EVERY bf16 section against the source pack
    bf16_checked = 0
    with open(fp8_path, "rb") as af, open(src.path, "rb") as bf:
        for name, shape, _srcs in spec:
            if is_quantized(name, quant_set):
                continue
            sec = r.section(name)
            af.seek(sec["data_offset"])
            bf.seek(sec["data_offset"])
            left = sec["data_bytes"]
            while left:
                n = min(left, CHUNK_COPY)
                if af.read(n) != bf.read(n):
                    raise AssertionError(f"{name}: bf16 bytes differ from source pack")
                left -= n
            bf16_checked += 1

    # spot-check 7 quantized sections x 4 rows: dequant vs bf16 originals
    # (per-row upcast keeps the temp at one row -- lm_head must not hit 1 GB)
    spot = []
    for name in SPOT_SECTIONS:
        sec = r.section(name)
        n, k = sec["dims"][0], sec["dims"][1]
        q, sc = r.fp8(name)
        src_rows = src.array(name)
        for row in (0, n // 3, 2 * n // 3, n - 1):
            w = bf16_to_f32(src_rows[row:row + 1])[0]
            dq = dequantize_fp8_rows(q[row:row + 1], sc[row:row + 1])[0]
            rel = float(np.linalg.norm(dq - w) / np.linalg.norm(w))
            cosv = float(np.dot(dq, w) / (np.linalg.norm(dq) * np.linalg.norm(w)))
            spot.append({"section": name, "row": int(row),
                         "rel_l2_err": rel, "cos": cosv,
                         "scale": float(sc[row]), "amax": float(np.abs(w).max())})
        del src_rows
    worst = max(s["rel_l2_err"] for s in spot)
    assert worst < 0.05, f"dequant spot-check rel L2 error {worst:.4%} >= 5%"

    return {"fp8_sections": n_fp8, "bf16_sections": n_bf16,
            "bf16_byte_equal_sections": bf16_checked, "spot": spot,
            "spot_worst_rel_l2": worst}


def main() -> None:
    ap = argparse.ArgumentParser(description="MiniCPM5-2B WPK BF16 -> mixed FP8 quantizer (P4 v1)")
    ap.add_argument("--model-dir", default="models/minicpm5-2b")
    args = ap.parse_args()
    model_dir = os.path.abspath(args.model_dir)
    t_start = time.perf_counter()

    src_path = os.path.join(model_dir, "model.wpk")
    out_path = os.path.join(model_dir, "model_fp8.wpk")
    man_path = os.path.join(model_dir, "model_manifest_fp8.json")
    src = WpkReader(src_path)
    spec = build_section_spec(NUM_LAYERS)
    assert src.section_count == len(spec) == 255, "source pack section_count != 255"
    quant_set = quantized_section_names(NUM_LAYERS)
    assert len(quant_set) == NUM_LAYERS * 4 + 1 == 169
    assert quant_set <= {n for n, _s, _x in spec}, "quantized names not all in spec"
    print(f"[quant] source: {src_path} ({src.file_size:,} B, {src.section_count} sections)")
    print(f"[quant] plan: {len(quant_set)} sections -> FP8 E4M3 per-row scaled "
          f"(42 x {QUANT_KINDS} + lm_head); {255 - len(quant_set)} sections stay BF16")

    data_start, entries, total_file = build_fp8_layout(spec, quant_set)
    payload = sum(e["data_bytes"] + e["scale_bytes"] for e in entries)
    pad_total = total_file - data_start - payload
    print(f"[quant] layout: data_start {data_start}, payload {payload:,} B "
          f"(fp8 + scales + bf16), padding {pad_total:,} B, file {total_file:,} B")

    dt_write, wstats = write_fp8_wpk(out_path, src, entries, src.model_hash,
                                     data_start, quant_set)
    assert os.path.getsize(out_path) == total_file
    acc = wstats["bytes"]
    print(f"[quant] wrote {out_path}: fp8 {acc['fp8']:,} B + scales {acc['scales']:,} B "
          f"+ bf16 {acc['bf16']:,} B in {dt_write:.1f}s "
          f"({(acc['fp8'] + acc['bf16']) / dt_write / 1e9:.2f} GB/s effective)")

    print("[quant] sha256: source pack ...")
    t0 = time.perf_counter()
    src_sha = sha256_file(src_path)
    t_src_sha = time.perf_counter() - t0
    bf16_man = os.path.join(model_dir, "model_manifest.json")
    if os.path.exists(bf16_man):
        with open(bf16_man) as fh:
            recorded = json.load(fh).get("wpk_sha256")
        assert recorded == src_sha, f"source pack sha256 drifted vs model_manifest.json: {recorded}"
    print(f"[quant]   model.wpk      {src_sha} ({t_src_sha:.1f}s)")

    t0 = time.perf_counter()
    fp8_sha = sha256_file(out_path)
    t_fp8_sha = time.perf_counter() - t0
    print(f"[quant]   model_fp8.wpk {fp8_sha} ({t_fp8_sha:.1f}s)")

    print("[quant] self-verify: re-parsing fp8 pack ...")
    t0 = time.perf_counter()
    v = verify_fp8_wpk(out_path, src, spec, quant_set)
    t_verify = time.perf_counter() - t0
    print(f"[quant] self-verify OK ({t_verify:.1f}s): {v['fp8_sections']} fp8 + "
          f"{v['bf16_sections']} bf16 == 255 sections; all {v['bf16_byte_equal_sections']} "
          f"bf16 sections byte-equal to source; dequant spot-check worst rel L2 "
          f"{v['spot_worst_rel_l2']:.4%} ({len(SPOT_SECTIONS)} sections x 4 rows)")
    for s in v["spot"]:
        print(f"[quant]   {s['section']:<18} row {s['row']:<6d} rel_l2 {s['rel_l2_err']:.4%}  "
              f"cos {s['cos']:.6f}  scale {s['scale']:.6e} (amax {s['amax']:.4e})")

    all_min = min(s["scale_min"] for s in wstats["scale_stats"])
    all_max = max(s["scale_max"] for s in wstats["scale_stats"])
    all_mean = float(np.mean([s["scale_mean"] for s in wstats["scale_stats"]]))

    manifest = {
        "model": MODEL_NAME,
        "revision": MODEL_REVISION,
        "precision": "mixed: bf16 + fp8-e4m3 (per-output-row scaled)",
        "kv_precision": "bf16",
        "mode": "P4 closeout mixed: prefill bf16 weights, decode dequantized fp8 "
                "weights (incl. lm_head)",
        "source_pack": {"path": "model.wpk", "sha256": src_sha, "bytes": src.file_size},
        "fp8_pack": {"path": "model_fp8.wpk", "sha256": fp8_sha, "bytes": os.path.getsize(out_path)},
        "quant": {
            "format": "fp8 e4m3 (1s + 4e bias7 + 3m, max finite 448, no inf)",
            "round": "rne",
            "saturation": "clamp |x| > 448 to +/-448 (NaN encodings never emitted)",
            "granularity": "per output row of [N,K]: one fp32 scale per row",
            "scale_rule": "max|row|/448, floor 1e-12",
            "quantized_sections": 169,
            "bf16_sections": 86,
            "quantized_kinds": list(QUANT_KINDS),
            "extra_quantized_sections": list(EXTRA_QUANT_SECTIONS),
            "lm_head": {
                "quantized": True,
                "shape": [130560, 2048],
                "native_effect": "fp8 pack lm_head dtype=2 -> loader fills "
                                 "fp8_lm_head/scale_lm_head -> decode (M=1) lm_head "
                                 "on gemv_fp8; prefill (M>1) lm_head stays BF16; "
                                 "MC_FP8_LMHEAD_OFF=1 reverts decode to BF16",
            },
        },
        "sections": {
            "count": 255,
            "toc_offset": WPK_HEADER_SIZE,
            "toc_entry_size": WPK_TOC_ENTRY_SIZE,
            "data_start": data_start,
            "alignment": WPK_ALIGN,
            "padding_bytes": pad_total,
        },
        "data_bytes": {
            "fp8": acc["fp8"],
            "scales": acc["scales"],
            "bf16": acc["bf16"],
            "total_payload": payload,
        },
        "scale_stats": {
            "global": {"min": all_min, "max": all_max, "mean": all_mean},
            "per_section": wstats["scale_stats"],
        },
        "quantized_section_list": [e["name"] for e in entries if e["dtype"] == DTYPE_FP8_E4M3],
        "verification": {
            "sections_total": v["fp8_sections"] + v["bf16_sections"],
            "fp8_sections": v["fp8_sections"],
            "bf16_sections": v["bf16_sections"],
            "bf16_byte_equal": v["bf16_byte_equal_sections"] == 86,
            "dequant_spotcheck": v["spot"],
            "dequant_spotcheck_worst_rel_l2": v["spot_worst_rel_l2"],
        },
        "abi_version": "0x0002",
        "target_sm": WPK_TARGET_SM,
        "timing": {
            "quantize_write_s": dt_write,
            "sha256_source_s": t_src_sha,
            "sha256_fp8_s": t_fp8_sha,
            "verify_s": t_verify,
            "total_s": time.perf_counter() - t_start,
        },
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(man_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
    print(f"[quant] wrote {man_path}")
    print(f"[quant] scale stats: min {all_min:.6e} max {all_max:.6e} mean {all_mean:.6e}")
    print(f"[quant] total elapsed: {time.perf_counter() - t_start:.1f}s "
          f"(quantize {dt_write:.1f}s + sha256 {t_src_sha + t_fp8_sha:.1f}s + verify {t_verify:.1f}s)")


if __name__ == "__main__":
    main()
