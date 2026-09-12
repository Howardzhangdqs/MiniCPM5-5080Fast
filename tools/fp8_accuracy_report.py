#!/usr/bin/env python3
"""fp8_accuracy_report.py -- golden_fp8 (P4 v1 mixed) vs golden (BF16) accuracy.

For each case (case_zh / case_en / case_mixed) compares, step by step:
  * logits cosine      (step{t}.logits.f32, 130560 fp32)
  * final_hidden cosine (step{t}.final_hidden.f32, 2048 fp32)
  * token agreement    (step{t}.token i32)

Step 0 is the BF16 prefill in BOTH runs (mixed-mode semantics), so it is
reported separately (expected bit-identical).  Steps >= 1 are decode steps
computed on dequantized FP8 weights in the fp8 run.

Note on interpretation: greedy decoding is chaotic -- after the first token
mismatch the two runs consume different token histories, so later-step cosines
measure behavioural similarity rather than pure numeric drift.  The report
therefore splits decode steps into:
  * pre-divergence  (identical token history -> pure FP8 numeric drift;
                     primary accuracy metric, expected cos 0.99+)
  * at-divergence   (the step where argmax flipped; usually a top-2 near-tie)
  * post-divergence (different content being generated; low cos is expected
                     greedy chaos, NOT a quantization error)

Outputs a summary table to stdout and JSON to benchmark/results/fp8_accuracy.json.
Expectation (plan P4): high token agreement, logits cos 0.99+ (pre-divergence);
significant deviations are reported honestly (they may indicate a
quantization-recipe problem rather than a reporting problem).
"""
from __future__ import annotations

import json
import os
import struct
import sys

import numpy as np

GOLDEN_BF16 = os.environ.get("GOLDEN_BF16", "tests/runtime/golden")
GOLDEN_FP8 = os.environ.get("GOLDEN_FP8", "tests/runtime/golden_fp8")
OUT_JSON = os.environ.get("FP8_REPORT_JSON", "benchmark/results/fp8_accuracy.json")
CASES = ["case_zh", "case_en", "case_mixed"]


def cos(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float(np.dot(a, b) / (na * nb))


def read_token(path: str) -> int:
    with open(path, "rb") as fh:
        return struct.unpack("<i", fh.read(4))[0]


def compare_case(case: str) -> dict:
    d_a = os.path.join(GOLDEN_BF16, case)
    d_b = os.path.join(GOLDEN_FP8, case)
    with open(os.path.join(d_a, "meta.json")) as fh:
        ma = json.load(fh)
    with open(os.path.join(d_b, "meta.json")) as fh:
        mb = json.load(fh)
    if ma["input_ids"] != mb["input_ids"] or ma["prompt"] != mb["prompt"]:
        raise SystemExit(f"{case}: prompt/input_ids differ between golden and golden_fp8 -- not comparable")
    n = min(ma["steps"], mb["steps"])
    note = None
    if ma["steps"] != mb["steps"]:
        note = f"step counts differ (bf16 {ma['steps']}, fp8 {mb['steps']}); comparing first {n}"

    logits_cos = []
    hidden_cos = []
    toks_a, toks_b = [], []
    for t in range(n):
        la = np.fromfile(os.path.join(d_a, f"step{t}.logits.f32"), dtype="<f4")
        lb = np.fromfile(os.path.join(d_b, f"step{t}.logits.f32"), dtype="<f4")
        assert la.size == lb.size == 130560, f"{case} step{t}: logits size {la.size}/{lb.size}"
        logits_cos.append(cos(la, lb))
        ha = np.fromfile(os.path.join(d_a, f"step{t}.final_hidden.f32"), dtype="<f4")
        hb = np.fromfile(os.path.join(d_b, f"step{t}.final_hidden.f32"), dtype="<f4")
        assert ha.size == hb.size == 2048
        hidden_cos.append(cos(ha, hb))
        toks_a.append(read_token(os.path.join(d_a, f"step{t}.token")))
        toks_b.append(read_token(os.path.join(d_b, f"step{t}.token")))

    tok_match = sum(a == b for a, b in zip(toks_a, toks_b))
    first_mismatch = next((t for t in range(n) if toks_a[t] != toks_b[t]), None)

    # history at step t is identical iff tokens 0..t-1 all match
    same_history = [all(toks_a[j] == toks_b[j] for j in range(t)) for t in range(n)]
    # pre-divergence decode steps = same_history and t >= 1
    pre = [logits_cos[t] for t in range(1, n) if same_history[t]]
    pre_h = [hidden_cos[t] for t in range(1, n) if same_history[t]]
    post = [logits_cos[t] for t in range(1, n) if not same_history[t]]

    # diagnostic: top-2 logit margin in the bf16 golden at the divergence step
    div_margin = None
    if first_mismatch is not None:
        la = np.fromfile(os.path.join(d_a, f"step{first_mismatch}.logits.f32"), dtype="<f4")
        lb = np.fromfile(os.path.join(d_b, f"step{first_mismatch}.logits.f32"), dtype="<f4")
        srt_a = np.sort(la)[::-1]
        srt_b = np.sort(lb)[::-1]
        div_margin = {
            "bf16_top1_top2_gap": float(srt_a[0] - srt_a[1]),
            "fp8_top1_top2_gap": float(srt_b[0] - srt_b[1]),
            "bf16_top1_id": int(np.argmax(la)),
            "fp8_top1_id": int(np.argmax(lb)),
            "logits_cos_at_divergence": logits_cos[first_mismatch],
        }

    return {
        "case": case,
        "steps_compared": n,
        "note": note,
        "tokens_matched": tok_match,
        "token_agreement": tok_match / n,
        "first_token_mismatch_step": first_mismatch,
        "divergence_top2_margin": div_margin,
        "step0_logits_cos": logits_cos[0],
        "step0_logits_bit_equal": bool(logits_cos[0] == 1.0),
        "pre_divergence_logits_cos_min": min(pre) if pre else None,
        "pre_divergence_logits_cos_avg": float(np.mean(pre)) if pre else None,
        "pre_divergence_hidden_cos_min": min(pre_h) if pre_h else None,
        "pre_divergence_hidden_cos_avg": float(np.mean(pre_h)) if pre_h else None,
        "post_divergence_logits_cos_min": min(post) if post else None,
        "post_divergence_logits_cos_avg": float(np.mean(post)) if post else None,
        "all_logits_cos_min": min(logits_cos),
        "all_logits_cos_avg": float(np.mean(logits_cos)),
        "last_hidden_cos": hidden_cos[-1],
        "_per_step_logits_cos": logits_cos,
        "_same_history": same_history,
    }


def main() -> int:
    results = [compare_case(c) for c in CASES]

    W = 118
    print("=" * W)
    print("FP8 (P4 v1 mixed: prefill BF16 / decode FP8-E4M3) vs BF16 golden")
    print("=" * W)
    hdr = (f"{'case':<12}{'steps':>6}{'tok ok':>8}{'agree':>8}"
           f"{'cos(s0)':>9}{'pre min':>9}{'pre avg':>9}{'post avg':>9}"
           f"{'1st diff':>9}{'div bf16 top1-top2':>20}")
    print(hdr)
    print("-" * W)
    for r in results:
        pre_min = f"{r['pre_divergence_logits_cos_min']:.6f}" if r["pre_divergence_logits_cos_min"] is not None else "n/a"
        pre_avg = f"{r['pre_divergence_logits_cos_avg']:.6f}" if r["pre_divergence_logits_cos_avg"] is not None else "n/a"
        post_avg = f"{r['post_divergence_logits_cos_avg']:.6f}" if r["post_divergence_logits_cos_avg"] is not None else "--"
        first = "never" if r["first_token_mismatch_step"] is None else str(r["first_token_mismatch_step"])
        margin = (f"{r['divergence_top2_margin']['bf16_top1_top2_gap']:.4f}"
                  if r["divergence_top2_margin"] else "--")
        print(f"{r['case']:<12}{r['steps_compared']:>6}{r['tokens_matched']:>8}"
              f"{r['token_agreement']:>8.1%}{r['step0_logits_cos']:>9.6f}"
              f"{pre_min:>9}{pre_avg:>9}{post_avg:>9}{first:>9}{margin:>20}")
        if r["note"]:
            print(f"    note: {r['note']}")
        dm = r["divergence_top2_margin"]
        if dm:
            print(f"    divergence step {r['first_token_mismatch_step']}: logits cos {dm['logits_cos_at_divergence']:.6f}, "
                  f"bf16 top1-top2 gap {dm['bf16_top1_top2_gap']:.4f} "
                  f"(bf16 picks {dm['bf16_top1_id']}, fp8 picks {dm['fp8_top1_id']}) -- argmax flip on a near-tie")
    print("-" * W)

    all_tok = sum(r["tokens_matched"] for r in results)
    all_steps = sum(r["steps_compared"] for r in results)
    pre_all = [c for r in results for t, c in enumerate(r["_per_step_logits_cos"])
               if t >= 1 and r["_same_history"][t]]
    pre_min_all = min(pre_all)
    pre_avg_all = float(np.mean(pre_all))
    print(f"overall: tokens {all_tok}/{all_steps} ({all_tok / all_steps:.1%}); "
          f"PRE-DIVERGENCE decode logits cos min {pre_min_all:.6f} avg {pre_avg_all:.6f}")
    print("         (post-divergence steps compare different generated content: greedy chaos, not numeric error)")

    # expectations (soft flags; honest reporting)
    exp = {
        "pre_divergence_logits_cos_avg_ge_0.99": bool(pre_avg_all >= 0.99),
        "pre_divergence_logits_cos_min_ge_0.98": bool(pre_min_all >= 0.98),
        "token_agreement_overall_ge_0.5": bool(all_tok / all_steps >= 0.5),
    }
    for k, v in exp.items():
        print(f"expectation {k}: {'MET' if v else 'NOT MET'}")

    out = {
        "comparison": f"{GOLDEN_FP8} (mixed fp8) vs {GOLDEN_BF16} (bf16)",
        "mode": "P4 v1 mixed: prefill bf16 weights, decode dequantized fp8 weights",
        "cases": [{k: v for k, v in r.items() if not k.startswith("_")} for r in results],
        "overall": {
            "steps_compared": all_steps,
            "tokens_matched": all_tok,
            "token_agreement": all_tok / all_steps,
            "pre_divergence_logits_cos_min": pre_min_all,
            "pre_divergence_logits_cos_avg": pre_avg_all,
        },
        "expectations": exp,
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as fh:
        json.dump(out, fh, indent=2)
        fh.write("\n")
    print(f"wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
