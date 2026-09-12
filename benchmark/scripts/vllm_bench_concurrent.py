#!/usr/bin/env python3
"""vLLM concurrent-throughput baseline for MiniCPM5-2B on RTX 5080.

Quantifies the Path-B target: vLLM aggregate decode throughput under K
concurrent requests (offline LLM.generate with K prompts = vLLM continuous
batching), matched to our concurrency_path_a.json scenario (ctx 2048,
64 output tokens, greedy, ignore_eos).

Env shim (CUDA_HOME/NVCC flags) is reused from vllm_bench by importing it.
"""
import json
import random
import statistics
import sys
import time
from datetime import datetime

sys.path.insert(0, "/workspace/benchmark/scripts")
import vllm_bench  # noqa: F401  (applies CUDA_HOME shim + env fixups)

MODEL_DIR = "/workspace/models/minicpm5-2b"
OUT_JSON = "/workspace/benchmark/results/vllm_concurrent.json"
PROMPT_TOKENS = 2048
OUT_TOKENS = 64
KS = (4, 8)
ROUNDS = 3


def build_prompt_ids(n: int, k: int) -> list[int]:
    rng = random.Random(vllm_bench.PROMPT_SEED + n + 1000 * k)
    lo, hi = vllm_bench.PROMPT_ID_RANGE
    return [rng.randrange(lo, hi) for _ in range(n)]


def main() -> int:
    import torch
    import vllm
    from vllm import SamplingParams

    driver = vllm_bench.sh(["nvidia-smi", "--query-gpu=driver_version,name",
                            "--format=csv,noheader,nounits"])
    mem_util = 0.85
    llm = None
    for attempt in (0.85, 0.7):
        print(f"Initializing vLLM LLM (gpu_memory_utilization={attempt}) ...", flush=True)
        try:
            llm = vllm_bench.make_llm(attempt)
            mem_util = attempt
            break
        except (torch.OutOfMemoryError, MemoryError):
            print("  OOM, falling back", flush=True)
    assert llm is not None

    cases = []
    sp = SamplingParams(temperature=0.0, max_tokens=OUT_TOKENS, ignore_eos=True)
    for K in KS:
        prompts = [{"prompt_token_ids": build_prompt_ids(PROMPT_TOKENS, i)}
                   for i in range(K)]
        # warmup (short)
        vllm_bench.wait_gpu_idle()
        llm.generate(prompts, sampling_params=SamplingParams(
            temperature=0.0, max_tokens=8, ignore_eos=True), use_tqdm=False)
        rounds = []
        for r in range(1, ROUNDS + 1):
            vllm_bench.wait_gpu_idle()
            t0 = time.perf_counter()
            outs = llm.generate(prompts, sampling_params=sp, use_tqdm=False)
            wall_s = time.perf_counter() - t0
            tpots = []
            for o in outs:
                m = o.metrics
                if m and m.first_token_ts is not None and m.last_token_ts is not None:
                    gen = len(o.outputs[0].token_ids)
                    if gen > 1:
                        tpots.append((m.last_token_ts - m.first_token_ts) / (gen - 1) * 1000.0)
            agg = K * OUT_TOKENS / wall_s
            rounds.append({
                "round": r, "wall_s": wall_s,
                "aggregate_tok_per_s": agg,
                "tpot_ms_mean_per_req": statistics.mean(tpots) if tpots else None,
            })
            print(f"K={K} round {r}: wall {wall_s:.2f}s aggregate {agg:.1f} tok/s "
                  f"per-req TPOT {rounds[-1]['tpot_ms_mean_per_req']:.2f}ms", flush=True)
            time.sleep(1.0)
        agg_med = statistics.median(r["aggregate_tok_per_s"] for r in rounds)
        cases.append({"K": K, "prompt_tokens": PROMPT_TOKENS,
                      "output_tokens": OUT_TOKENS, "rounds": rounds,
                      "aggregate_tok_per_s_median": agg_med})

    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "purpose": ("vLLM concurrent decode throughput baseline (continuous "
                    "batching via offline multi-prompt generate); companion "
                    "of concurrency_path_a.json (ours, time-sliced)"),
        "environment": {"vllm": vllm.__version__, "torch": torch.__version__,
                        "driver_gpu": driver,
                        "gpu_memory_utilization": mem_util},
        "config": {"dtype": "bfloat16", "max_model_len": 8448,
                   "enable_prefix_caching": False, "sampling": "greedy ignore_eos"},
        "cases": cases,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {OUT_JSON}")
    del llm
    return 0


if __name__ == "__main__":
    sys.exit(main())
