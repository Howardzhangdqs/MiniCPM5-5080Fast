#!/usr/bin/env python3
"""Merge the three sweep-segment JSONs into the canonical perf_sweep.json and
print the BF16-vs-vLLM comparison (fair-comparison axis per user rule: same
precision BF16 vs BF16; FP8 shown only as internal quantization uplift)."""
import json
import statistics
from datetime import datetime

SEGS = [
    "/workspace/benchmark/results/perf_sweep_lo.json",
    "/workspace/benchmark/results/perf_sweep_hi1.json",
    "/workspace/benchmark/results/perf_sweep_hi2.json",
]
VLLM = "/workspace/benchmark/results/vllm_sweep.json"
OUT = "/workspace/benchmark/results/perf_sweep.json"

rows = []
for p in SEGS:
    rows += json.load(open(p))["rows"]
assert len(rows) == 14, f"expected 14 rows, got {len(rows)}"
expect = {(m, c) for m in ("fp8", "bf16")
          for c in (16, 512, 2048, 8192, 32768, 65536, 130816)}
got = {(r["mode"], r["ctx"]) for r in rows}
assert got == expect, f"row mismatch: {expect ^ got}"

for r in rows:
    rounds = r["rounds_ms"]
    r["rounds_n"] = len(rounds)
    r["mean_ms"] = round(sum(rounds) / len(rounds), 4)
    r["median_ms"] = round(statistics.median(rounds), 4)
    r["tok_per_s_mean"] = round(1000.0 / r["mean_ms"], 2)

result = {
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "purpose": ("native 0-128K ctx decode sweep, FINAL after optimization "
                "campaign: HMMA flash-decoding attention (production default, "
                "MC_ATTN_HMMA_OFF fallback) + two-level merge + chunk tiers "
                "64/128/256 + flash prefill (production default, "
                "MC_PREFILL_FLASH_OFF fallback); per-tier model load so each "
                "tier gets its deployment chunk tier"),
    "discipline": ("idle-GPU back-to-back (util<=5% checked before every "
                   "round), greedy, batch=1, fresh cold prefill per round, "
                   "ignore_eos; 5 rounds x 200 output tokens (ctx>=65536: 3); "
                   "mean + median"),
    "comparison_axis": ("BF16 vs vLLM BF16 only (user rule); FP8 rows are "
                        "internal quantization uplift, NOT a vLLM comparison"),
    "rows": sorted(rows, key=lambda r: (r["mode"], r["ctx"])),
}
with open(OUT, "w") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

v = json.load(open(VLLM))
byname = {c["name"]: c for c in v["cases"]}
ctx2name = {16: "ctx16_out256", 512: "ctx512_out128", 2048: "ctx2048_out128",
            8192: "ctx8192_out128", 32768: "ctx32768_out128",
            65536: "ctx65536_out128", 130816: "ctx130816_out128"}
native = {(r["mode"], r["ctx"]): r for r in rows}

hdr = f"{'ctx':>7} | {'ours BF16 mean':>16} | {'ours FP8 mean':>16} | {'vLLM BF16':>13} | {'BF16 vs vLLM':>12}"
print(hdr); print("-" * len(hdr))
for ctx in (16, 512, 2048, 8192, 32768, 65536, 130816):
    b, p = native[("bf16", ctx)], native[("fp8", ctx)]
    vt = byname[ctx2name[ctx]]["mean"]["tpot_ms"]
    lead = (vt / b["mean_ms"] - 1) * 100
    print(f"{ctx:>7} | {b['mean_ms']:>6.3f}ms {b['tok_per_s_mean']:>6.1f}t/s | "
          f"{p['mean_ms']:>6.3f}ms {p['tok_per_s_mean']:>6.1f}t/s | "
          f"{vt:>6.3f}ms | {'+' if lead>=0 else ''}{lead:.1f}%")
print(f"\nWrote {OUT}")
