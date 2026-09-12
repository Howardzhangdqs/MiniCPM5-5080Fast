#!/usr/bin/env python3
"""pack_weights.py -- MiniCPM5-2B offline weight packer (plan §16.1/§16.2).

Reads the source safetensors (self-parsed header; bf16 data), fuses and
reorders tensors into the section layout required by the SM120 kernels, and
writes:

  <model-dir>/model.wpk           WPK v1 container (header + TOC + bf16 data)
  <model-dir>/model_manifest.json provenance manifest (§16.3 subset)

Section order: embedding, lm_head, final_norm, then for i=0..41:
layer{i}.norm1, layer{i}.qkv, layer{i}.o, layer{i}.norm2, layer{i}.gate_up,
layer{i}.down.  Fused sections are row concatenations (qkv = q|k|v,
gate_up = gate|up).  All sections are BF16 row-major [out, in] (norms rank 1),
data 256B aligned, no scales.

Memory discipline: the safetensors stays mmap'd as uint16 views; section data
is streamed tensor-by-tensor (row concat = sequential writes), so peak extra
RAM stays in the tens of MB.

Self-verification (runs after writing): re-parse the wpk, validate header/TOC
invariants, sample sections and byte-compare rows against the safetensors,
assert the expected section count (3 + 42*6 = 255).

Usage:
  python tools/pack_weights.py --model-dir models/minicpm5-2b
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
    LAYOUT_ROW_MAJOR,
    NUM_LAYERS,
    SafeTensorFile,
    WPK_ABI_VERSION,
    WPK_ALIGN,
    WPK_HEADER_SIZE,
    WPK_TARGET_SM,
    WPK_TOC_ENTRY_SIZE,
    WpkReader,
    align_up,
    build_section_spec,
    expected_source_keys,
    fnv1a64,
    pack_header,
    pack_toc_entry,
    sha256_file,
)

SAFETENSORS_SHARD = "model-00000-of-00001.safetensors"


def find_safetensors(model_dir: str) -> str:
    """Resolve the safetensors shard (honoring index.json if present)."""
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx) as fh:
            weight_map = json.load(fh)["weight_map"]
        files = sorted(set(weight_map.values()))
    else:
        files = sorted(f for f in os.listdir(model_dir) if f.endswith(".safetensors"))
    if len(files) != 1:
        raise SystemExit(
            f"expected exactly one safetensors shard for MiniCPM5-2B, found {len(files)}: {files}"
        )
    return os.path.join(model_dir, files[0])


def validate_source(stf: SafeTensorFile, spec) -> None:
    """Exact two-way check of expected vs actual safetensors keys/shapes/dtypes."""
    expected_shape: dict[str, tuple[int, ...]] = {}
    for _name, shape, srcs in spec:
        for key, r0, rcount in srcs:
            # rank-1 sections map the whole source vector as one chunk;
            # rank-2 sources are [row-slice, cols] slices.
            expected_shape[key] = shape if len(shape) == 1 else (rcount, shape[1])
    problems = []
    for key, entry in stf.header.items():
        if key == "__torch_function__" or key.startswith("metadata."):
            continue
        shape = tuple(entry["shape"])
        if key not in expected_shape:
            problems.append(f"unexpected extra tensor: {key} {shape}")
            continue
        if entry.get("dtype") != "BF16":
            problems.append(f"{key}: dtype {entry.get('dtype')} != BF16")
        if shape != expected_shape[key]:
            problems.append(f"{key}: shape {shape} != {expected_shape[key]}")
    for key in sorted(set(expected_shape) - set(expected_shape.keys() & stf.header.keys())):
        problems.append(f"missing tensor: {key}")
    if problems:
        raise SystemExit(
            "source safetensors does not match the section spec -- stopping:\n  "
            + "\n  ".join(problems)
        )


def build_layout(spec):
    """Compute (data_start, entries, total_file_size, total_data_bytes).

    entries: list of (name, shape, data_offset, data_bytes, sources)
    """
    toc_end = WPK_HEADER_SIZE + len(spec) * WPK_TOC_ENTRY_SIZE
    data_start = align_up(toc_end, WPK_ALIGN)
    cur = data_start
    entries = []
    total_data = 0
    pad_total = 0
    for name, shape, srcs in spec:
        nbytes = 2
        for d in shape:
            nbytes *= d
        entries.append((name, shape, cur, nbytes, srcs))
        total_data += nbytes
        nxt = align_up(cur + nbytes, WPK_ALIGN)
        pad_total += nxt - (cur + nbytes)
        cur = nxt
    return data_start, entries, cur, total_data, pad_total


def write_wpk(out_path: str, model_hash: bytes, entries, data_start: int, stf: SafeTensorFile):
    """Stream the wpk out; returns (elapsed_s, payload_bytes_written)."""
    t0 = time.perf_counter()
    tmp_path = out_path + ".tmp"
    written = 0
    with open(tmp_path, "wb") as out:
        out.write(pack_header(model_hash, len(entries), data_start))
        for name, shape, off, nbytes, srcs in entries:
            out.write(pack_toc_entry(name, shape, off, nbytes))
        pos = WPK_HEADER_SIZE + len(entries) * WPK_TOC_ENTRY_SIZE
        if pos < data_start:
            out.write(b"\x00" * (data_start - pos))

        pos = data_start
        for name, shape, off, nbytes, srcs in entries:
            assert pos == off, f"layout desync at {name}: {pos} != {off}"
            if len(shape) == 1:
                # rank-1 norm: whole vector, single source
                view, _s, _boff, _blen = stf.tensor(srcs[0][0])
                out.write(memoryview(np.ascontiguousarray(view)))
                written += view.nbytes
            else:
                # rank-2: row concatenation of sources (q|k|v, gate|up, or single)
                for key, r0, rcount in srcs:
                    view, _s, _boff, _blen = stf.tensor(key)
                    arr = view[r0:r0 + rcount]
                    out.write(memoryview(np.ascontiguousarray(arr)))
                    written += arr.nbytes
            pos += nbytes
            aligned = align_up(pos, WPK_ALIGN)
            if aligned > pos:
                out.write(b"\x00" * (aligned - pos))
                pos = aligned
        assert written == sum(e[3] for e in entries), "payload byte accounting mismatch"
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp_path, out_path)
    return time.perf_counter() - t0, written


def self_verify(wpk_path: str, stf: SafeTensorFile, spec, src_sha: str) -> dict:
    """Re-parse wpk; check invariants; byte-compare sample rows vs safetensors."""
    r = WpkReader(wpk_path)
    assert r.magic == 0x31504B4D and r.abi_version == WPK_ABI_VERSION and r.target_sm == WPK_TARGET_SM
    assert r.section_count == len(spec), f"section_count {r.section_count} != {len(spec)}"
    assert r.model_hash.hex() == src_sha, "model_hash mismatch"
    assert r.toc_offset == WPK_HEADER_SIZE and r.reserved == 0

    layout = {}
    for name, shape, off, nb, srcs in build_layout(spec)[1]:
        layout[name] = (shape, off, nb, srcs)

    # every section: TOC invariants
    for name, (shape, off, nb, _srcs) in layout.items():
        sec = r.section(name)
        assert sec["data_offset"] == off and sec["data_bytes"] == nb, f"{name}: TOC offset/bytes mismatch"
        assert sec["data_offset"] % WPK_ALIGN == 0, f"{name}: offset not 256B aligned"
        assert sec["dtype"] == DTYPE_BF16 and sec["layout"] == LAYOUT_ROW_MAJOR
        assert sec["rank"] == len(shape) and sec["dims"][: len(shape)] == shape
        assert sec["dims"][len(shape):] == (0,) * (8 - len(shape))
        assert sec["scale_offset"] == 0 and sec["scale_bytes"] == 0
        assert 2 * int(np.prod(shape)) == nb

    # name-hash spot check: TOC entry i has the FNV-1a of the expected name
    for i, name in enumerate(layout):
        sec = next(s for s in r._by_hash.values() if s["index"] == i)
        assert sec["name_hash"] == fnv1a64(name.encode("utf-8")), f"TOC[{i}] name_hash != FNV({name})"

    def map_row(name: str, row: int):
        """Map an output row of a fused section back to (source_key, source_row)."""
        srcs = layout[name][3]
        acc = 0
        for key, r0, rcount in srcs:
            if acc <= row < acc + rcount:
                return key, row - acc + r0
            acc += rcount
        raise AssertionError(f"{name}: row {row} out of fused range")

    samples = {
        "embedding": [0, 65280, 130559],
        "lm_head": [0, 60000, 130559],
        "final_norm": None,               # rank-1: whole-vector compare
        "layer0.qkv": [0, 2048, 2559],    # q row0 | k row0 | v row255
        "layer20.down": [0, 1000, 2047],
        "layer41.gate_up": [0, 6144, 12287],  # gate row0 | up row0 | up row6143
    }
    assert len(samples) >= 5
    rows_checked = 0
    bytes_checked = 0
    with open(wpk_path, "rb") as wf:
        for name, rows in samples.items():
            shape, off, nb, _srcs = layout[name]
            if rows is None:  # rank-1 vector
                key = layout[name][3][0][0]
                view, _s, boff, blen = stf.tensor(key)
                wf.seek(off)
                assert wf.read(nb) == stf._mm[boff // 2:(boff + blen) // 2].tobytes(), f"{name}: bytes differ"
                rows_checked += 1
                bytes_checked += nb
                continue
            for row in rows:
                key, srow = map_row(name, row)
                view, _s, boff, blen = stf.tensor(key)
                row_bytes = view.shape[1] * 2
                wf.seek(off + row * row_bytes)
                wpk_row = wf.read(row_bytes)
                st_row = stf._mm[(boff + srow * row_bytes) // 2:(boff + (srow + 1) * row_bytes) // 2].tobytes()
                assert wpk_row == st_row, f"{name} row {row} (= {key} row {srow}): bytes differ"
                rows_checked += 1
                bytes_checked += row_bytes
    return {"rows_checked": rows_checked, "bytes_checked": bytes_checked}


def main() -> None:
    ap = argparse.ArgumentParser(description="MiniCPM5-2B WPK v1 packer (§16.1/§16.2)")
    ap.add_argument("--model-dir", default="models/minicpm5-2b")
    args = ap.parse_args()
    model_dir = os.path.abspath(args.model_dir)
    t_start = time.perf_counter()

    st_path = find_safetensors(model_dir)
    stf = SafeTensorFile(st_path)
    spec = build_section_spec(NUM_LAYERS)
    validate_source(stf, spec)
    print(f"[pack] source OK: {st_path} ({stf.file_size:,} bytes, "
          f"{len(expected_source_keys(spec))} source tensors, dtype/shape all match spec)")

    print("[pack] computing source sha256 ...")
    t0 = time.perf_counter()
    src_sha = sha256_file(st_path)
    print(f"[pack] safetensors sha256 = {src_sha} ({time.perf_counter() - t0:.1f}s)")

    ref_manifest = os.path.join(model_dir, "reference_manifest.json")
    if os.path.exists(ref_manifest):
        with open(ref_manifest) as fh:
            recorded = json.load(fh)["files"].get(SAFETENSORS_SHARD, {}).get("sha256")
        if recorded and recorded != src_sha:
            raise SystemExit(f"sha256 mismatch vs reference_manifest.json: {recorded}")

    data_start, entries, total_file, total_data, pad_total = build_layout(spec)
    print(f"[pack] layout: {len(spec)} sections (= 3 global + {NUM_LAYERS} x 6 per-layer), "
          f"toc {len(spec)}x{WPK_TOC_ENTRY_SIZE}B, data_start {data_start}, "
          f"data {total_data:,} B, padding {pad_total} B, file {total_file:,} B")

    wpk_path = os.path.join(model_dir, "model.wpk")
    dt, written = write_wpk(wpk_path, bytes.fromhex(src_sha), entries, data_start, stf)
    assert os.path.getsize(wpk_path) == total_file
    print(f"[pack] wrote {wpk_path}: {written:,} payload bytes in {dt:.1f}s "
          f"({written / dt / 1e9:.2f} GB/s effective)")

    print("[pack] computing wpk sha256 ...")
    t0 = time.perf_counter()
    wpk_sha = sha256_file(wpk_path)
    print(f"[pack] wpk sha256 = {wpk_sha} ({time.perf_counter() - t0:.1f}s)")

    print("[pack] self-verify: re-parsing wpk ...")
    t0 = time.perf_counter()
    stats = self_verify(wpk_path, stf, spec, src_sha)
    assert len(spec) == 3 + NUM_LAYERS * 6 == 255
    print(f"[pack] self-verify OK ({time.perf_counter() - t0:.1f}s): header/TOC valid; "
          f"{stats['rows_checked']} sampled rows byte-equal ({stats['bytes_checked']:,} B); "
          f"section_count == 255 == 3 + 42*6")

    manifest = {
        "model": "openbmb/MiniCPM5-2B",
        "revision": "3497c460c89e00520c3cfa2e73f49ab7647f1177",
        "safetensors_sha256": src_sha,
        "safetensors_bytes": stf.file_size,
        "wpk_sha256": wpk_sha,
        "wpk_bytes": os.path.getsize(wpk_path),
        "sections": {
            "count": len(spec),
            "total_data_bytes": total_data,
            "global_sections": 3,
            "per_layer_sections": 6,
            "toc_offset": WPK_HEADER_SIZE,
            "toc_entry_size": WPK_TOC_ENTRY_SIZE,
            "data_start": data_start,
            "alignment": WPK_ALIGN,
            "padding_bytes": pad_total,
        },
        "precision": "bf16",
        "kv_precision": "bf16",
        "scale_scheme": "none",
        "abi_version": "0x0002",
        "target_sm": WPK_TARGET_SM,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    man_path = os.path.join(model_dir, "model_manifest.json")
    with open(man_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
    print(f"[pack] wrote {man_path}")
    print(f"[pack] total elapsed: {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    main()
