#!/usr/bin/env python3
"""compare_dump.py -- golden vs native-dump numerics comparison.

Usage:
  python tools/compare_dump.py <golden_dir> <dump_dir> [options]

Compares the .f32 tensors (cosine similarity + max abs error) and .token
files (int equality) of two isomorphic directories (golden produced by
tools/reference_forward.py; dump produced by the native runtime).

Pass criteria (overridable):
  * layer tensors (layer{i}.hidden_in/attn_out/mlp_out.f32 and
    step{t}.final_hidden.f32): cosine >= --layer-cos (default 0.995)
  * logits (step{t}.logits.f32): cosine >= --logits-cos (default 0.99)
  * tokens: matches >= --min-tokens (default: ceil(29/32 * n_golden_tokens))
  * tensor size mismatches and files missing on the dump side are listed and
    counted as failures.
Exit code 0 = PASS, 1 = FAIL. meta.json is compared textually for info only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys

import numpy as np

LAYER_RE = re.compile(r"^layer\d+\.(hidden_in|attn_out|mlp_out)\.f32$")
FINAL_H_RE = re.compile(r"^step\d+\.final_hidden\.f32$")
LOGITS_RE = re.compile(r"^step\d+\.logits\.f32$")
TOKEN_RE = re.compile(r"^step\d+\.token$")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 and nb == 0.0:
        return 1.0
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def classify(name: str) -> str:
    if LAYER_RE.match(name) or FINAL_H_RE.match(name):
        return "layer"
    if LOGITS_RE.match(name):
        return "logits"
    if TOKEN_RE.match(name):
        return "token"
    if name == "input_ids.bin":
        return "input_ids"
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser(description="golden vs dump comparison")
    ap.add_argument("golden_dir")
    ap.add_argument("dump_dir")
    ap.add_argument("--layer-cos", type=float, default=0.995)
    ap.add_argument("--logits-cos", type=float, default=0.99)
    ap.add_argument("--min-tokens", type=int, default=None,
                    help="default: ceil(29/32 * n_golden_tokens)")
    ap.add_argument("--min-layer-rate", type=float, default=1.0,
                    help="fraction of layer tensors that must pass (default 1.0)")
    ap.add_argument("--min-logits-rate", type=float, default=1.0,
                    help="fraction of logits tensors that must pass (default 1.0)")
    ap.add_argument("--post-mismatch", choices=["include", "exclude"], default="include",
                    help="exclude step{t}.final_hidden/logits of steps AFTER the "
                         "first token mismatch: the sampled-token chain (and thus "
                         "model inputs) has diverged there, so tensor comparison is "
                         "meaningless (default: include, historical behavior)")
    args = ap.parse_args()

    gdir, ddir = args.golden_dir, args.dump_dir
    for d in (gdir, ddir):
        if not os.path.isdir(d):
            print(f"ERROR: not a directory: {d}")
            return 2

    golden_files = sorted(f for f in os.listdir(gdir) if f != "meta.json")
    rows, missing, size_mismatch = [], [], []

    for name in golden_files:
        gpath = os.path.join(gdir, name)
        dpath = os.path.join(ddir, name)
        kind = classify(name)
        if not os.path.exists(dpath):
            missing.append(name)
            continue
        if kind in ("layer", "logits"):
            a = np.fromfile(gpath, dtype="<f4")
            b = np.fromfile(dpath, dtype="<f4")
            if a.size != b.size:
                size_mismatch.append((name, a.size, b.size))
                continue
            rows.append((name, kind, cosine(a, b), float(np.max(np.abs(a - b))) if a.size else 0.0, None))
        elif kind == "token":
            a = int(np.fromfile(gpath, dtype="<i4")[0])
            b = int(np.fromfile(dpath, dtype="<i4")[0])
            rows.append((name, "token", None, None, a == b))
        elif kind == "input_ids":
            a = np.fromfile(gpath, dtype="<i4")
            b = np.fromfile(dpath, dtype="<i4")
            ok = a.shape == b.shape and bool(np.array_equal(a, b))
            rows.append((name, "input_ids", None, None, ok))
        else:
            rows.append((name, "other", None, None, None))

    # ---- table ----
    hdr = f"{'file':44s} {'kind':9s} {'cosine':>9s} {'max_abs_err':>12s} {'match':>6s}"
    print(hdr)
    print("-" * len(hdr))
    for name, kind, cos, mae, match in rows:
        cos_s = f"{cos:.6f}" if cos is not None else "-"
        mae_s = f"{mae:.6e}" if mae is not None else "-"
        match_s = {True: "yes", False: "NO"}.get(match, "-") if match is not None else "-"
        print(f"{name:44s} {kind:9s} {cos_s:>9s} {mae_s:>12s} {match_s:>6s}")

    # ---- summary ----
    layer_rows = [r for r in rows if r[1] == "layer"]
    logits_rows = [r for r in rows if r[1] == "logits"]
    token_rows = [r for r in rows if r[1] == "token"]

    # --post-mismatch exclude：首个 token 失配步之后的 step 张量不参与
    # 汇总（token 链已分叉、输入不同，张量比较无意义）。layer{i}.* 属
    # step0 采样前，不受影响。
    excluded = []
    if args.post_mismatch == "exclude" and token_rows:
        step_of = lambda r: int(re.match(r"^step(\d+)\.", r[0]).group(1)) \
            if re.match(r"^step(\d+)\.", r[0]) else -1
        bad_steps = [step_of(r) for r in token_rows if not r[4]]
        first_bad = min(bad_steps) if bad_steps else None
        if first_bad is not None:
            def post_bad(r):
                s = step_of(r)
                return s > first_bad
            excluded = [r[0] for r in layer_rows if post_bad(r)] + \
                       [r[0] for r in logits_rows if post_bad(r)]
            layer_rows = [r for r in layer_rows if not post_bad(r)]
            logits_rows = [r for r in logits_rows if not post_bad(r)]

    def passed(r):
        if r[1] == "layer":
            return r[2] >= args.layer_cos
        if r[1] == "logits":
            return r[2] >= args.logits_cos
        return bool(r[4])

    layer_pass = sum(passed(r) for r in layer_rows)
    logits_pass = sum(passed(r) for r in logits_rows)
    token_pass = sum(1 for r in token_rows if r[4])
    # missing / size-mismatched files count as failures in their class
    n_layer = len(layer_rows) + sum(1 for m in missing if classify(m) == "layer") \
        + sum(1 for n, _a, _b in size_mismatch if classify(n) == "layer")
    n_logits = len(logits_rows) + sum(1 for m in missing if classify(m) == "logits") \
        + sum(1 for n, _a, _b in size_mismatch if classify(n) == "logits")
    n_token = len(token_rows) + sum(1 for m in missing if classify(m) == "token")

    print()
    print(f"tensors compared : {len(rows)}   missing on dump side: {len(missing)}   "
          f"size mismatch: {len(size_mismatch)}")
    if excluded:
        print(f"  post-token-mismatch excluded (inputs diverged): {len(excluded)} -> "
              f"{', '.join(excluded)}")
    if missing:
        for m in missing:
            print(f"  MISSING: {m}")
    for name, ga, gb in size_mismatch:
        print(f"  SIZE MISMATCH: {name}: golden {ga} vs dump {gb} elements")

    min_tokens = args.min_tokens
    if min_tokens is None and n_token:
        min_tokens = math.ceil(29 / 32 * n_token)

    checks = []
    checks.append(("layer tensors cos>=%.4f" % args.layer_cos,
                   layer_pass, n_layer, n_layer > 0 and layer_pass >= math.ceil(args.min_layer_rate * n_layer)))
    checks.append(("logits cos>=%.3f" % args.logits_cos,
                   logits_pass, n_logits, n_logits > 0 and logits_pass >= math.ceil(args.min_logits_rate * n_logits)))
    if n_token:
        checks.append((f"tokens match >= {min_tokens}/{n_token}",
                       token_pass, n_token, token_pass >= min_tokens))

    print()
    overall = True
    for label, p, n, ok in checks:
        print(f"  {label:44s} {p}/{n}  {'PASS' if ok else 'FAIL'}")
        overall &= bool(ok)
    if not rows and not missing:
        print("  (nothing comparable found)")
        overall = False

    print(f"\nOVERALL: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
