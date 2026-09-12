#!/usr/bin/env python3
"""vLLM FP8 fair-comparison baseline for MiniCPM5-2B on RTX 5080.

Fair axis per user rule: our-FP8 vs vLLM-FP8 (same quantization class).
Official scenario: ctx 2048 prompt / 64 out / greedy / ignore_eos.
  - K=1 TPOT (5 rounds median) — single-request line
  - K=8 concurrent aggregate (3 rounds median) — concurrent line
Env shim reused from vllm_bench by import.
"""
import json
import random
import statistics
import sys
import time
from datetime import datetime

sys.path.insert(0, "/workspace/benchmark/scripts")
import vllm_bench  # noqa: F401  (CUDA_HOME shim + env fixups)

MODEL_DIR = "/workspace/models/minicpm5-2b"
OUT_JSON = "/workspace/benchmark/results/vllm_fp8.json"
PROMPT_TOKENS = 2048
OUT_TOKENS = 64
K_CONC = 8


def build_prompt_ids(n: int, k: int) -> list[int]:
    rng = random.Random(vllm_bench.PROMPT_SEED + n + 1000 * k)
    lo, hi = vllm_bench.PROMPT_ID_RANGE
    return [rng.randrange(lo, hi) for _ in range(n)]


def main() -> int:
    import torch
    import vllm
    from vllm import LLM, SamplingParams

    driver = vllm_bench.sh(["nvidia-smi", "--query-gpu=driver_version,name",
                            "--format=csv,noheader,nounits"])
    llm = None
    mem_util = 0.85
    for attempt in (0.85, 0.75):
        print(f"Initializing vLLM LLM (quantization=fp8, gpu_mem={attempt}) ...",
              flush=True)
        try:
            llm = LLM(model=MODEL_DIR, dtype="bfloat16", max_model_len=8448,
                      gpu_memory_utilization=attempt, enforce_eager=False,
                      enable_prefix_caching=False, disable_log_stats=False,
                      quantization="fp8")
            mem_util = attempt
            break
        except (torch.OutOfMemoryError, MemoryError, Exception) as e:
            print(f"  init failed: {type(e).__name__}: {e}", flush=True)
    if llm is None:
        print("ERROR: fp8 engine init failed", flush=True)
        return 1

    def tpot_of(o) -> float | None:
        m = o.metrics
        if m and m.first_token_ts is not None and m.last_token_ts is not None:
            gen = len(o.outputs[0].token_ids)
            if gen > 1:
                return (m.last_token_ts - m.first_token_ts) / (gen - 1) * 1000.0
        return None

    # --- K=1: single-request TPOT, 5 rounds ---
    sp = SamplingParams(temperature=0.0, max_tokens=OUT_TOKENS, ignore_eos=True)
    p1 = [{"prompt_token_ids": build_prompt_ids(PROMPT_TOKENS, 0)}]
    vllm_bench.wait_gpu_idle()
    llm.generate(p1, sampling_params=SamplingParams(temperature=0.0, max_tokens=8,
                                                    ignore_eos=True), use_tqdm=False)
    k1_rounds = []
    for r in range(5):
        vllm_bench.wait_gpu_idle()
        t0 = time.perf_counter()
        outs = llm.generate(p1, sampling_params=sp, use_tqdm=False)
        wall = time.perf_counter() - t0
        tpot = tpot_of(outs[0])
        k1_rounds.append({"round": r, "wall_s": wall, "tpot_ms": tpot})
        print(f"K=1 round {r}: TPOT {tpot:.3f}ms wall {wall:.2f}s", flush=True)
        time.sleep(1.0)

    # --- K=8 concurrent, 3 rounds ---
    prompts = [{"prompt_token_ids": build_prompt_ids(PROMPT_TOKENS, i)}
               for i in range(K_CONC)]
    vllm_bench.wait_gpu_idle()
    llm.generate(prompts, sampling_params=SamplingParams(temperature=0.0,
                                                         max_tokens=8,
                                                         ignore_eos=True),
                 use_tqdm=False)
    k8_rounds = []
    for r in range(3):
        vllm_bench.wait_gpu_idle()
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sampling_params=sp, use_tqdm=False)
        wall = time.perf_counter() - t0
        tpots = [t for t in (tpot_of(o) for o in outs) if t is not None]
        agg = K_CONC * OUT_TOKENS / wall
        k8_rounds.append({"round": r, "wall_s": wall,
                          "aggregate_tok_per_s": agg,
                          "tpot_ms_mean_per_req": statistics.mean(tpots)})
        print(f"K=8 round {r}: wall {wall:.2f}s aggregate {agg:.1f} tok/s "
              f"per-req TPOT {statistics.mean(tpots):.2f}ms", flush=True)
        time.sleep(1.0)

    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "purpose": ("vLLM FP8 fair-comparison baseline (dynamic weight-only "
                    "fp8); official scenario ctx2048/64tok/greedy; companion "
                    "line for our fp8 serving completion"),
        "environment": {"vllm": vllm.__version__, "torch": torch.__version__,
                        "driver_gpu": driver, "gpu_memory_utilization": mem_util,
                        "quantization": "fp8 (vLLM dynamic)"},
        "k1": {"tpot_ms_median": statistics.median(r["tpot_ms"] for r in k1_rounds),
               "rounds": k1_rounds},
        "k8": {"aggregate_tok_per_s_median":
               statistics.median(r["aggregate_tok_per_s"] for r in k8_rounds),
               "rounds": k8_rounds},
    }
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
