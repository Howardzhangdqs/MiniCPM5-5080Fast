#!/usr/bin/env python3
"""vLLM context-length sweep benchmark for MiniCPM5-2B on RTX 5080 (0-128K scan).
Variant of vllm_bench.py: 7-point prompt-length sweep (16..130816 tokens) at
max_model_len=131072 (config max_position_embeddings=131072, rope_scaling=null);
top tier 130816 = 131072 - 256 so prompt + output tokens fit the window.
Native-runtime cross-reference (perf_bf16/perf_v2) is merged offline in a later
step, so this script writes vLLM-only rows.

Discipline (matches native harness):
  - greedy (temperature=0), batch=1, ignore_eos=True so exact token counts are generated
  - GPU util checked before every measurement; wait+retry while util > 5%
  - per case: 1 warmup (short output), then 5 timed rounds (3 rounds for ctx >= 65536),
    take median AND arithmetic mean
  - prefix caching disabled so every round does a full cold prefill
    (otherwise rounds 2+ would hit the KV prefix cache and TTFT would be meaningless)

Metrics come from RequestOutput.metrics (vLLM 0.29: RequestStateStats):
  TTFT  = metrics.first_token_latency
  TPOT  = (last_token_ts - first_token_ts) / (n_generated - 1)   [engine-core monotonic clock]
  prefill tok/s = prompt_tokens / TTFT
Wall-clock e2e is recorded as a cross-check.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import random
import re
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime

# --- environment fixups (must run before importing vllm) ---
# Host /usr/local/cuda is 12.8; flashinfer refuses SM 12.x (sm_120, RTX 5080) with nvcc < 12.9,
# killing vLLM engine init ("FlashInfer requires GPUs with sm75 or higher"). The venv ships a full
# CUDA 13.3 toolkit (nvidia-cu13 wheel); point CUDA_HOME at it so JIT sees a >=12.9 nvcc.
VENV_CUDA_HOME = (
    "/workspace/.venv-vllm/lib/python3.12/site-packages/nvidia/cu13"
)


def _ensure_cuda_home_shim() -> str | None:
    """Build a CUDA_HOME layout around the pip CUDA 13.3 wheel.

    The wheel keeps shared libs in lib/ (versioned only, e.g. libcudart.so.13) while
    flashinfer/triton link against $CUDA_HOME/lib64/libcudart.so and libcuda.so, so we
    assemble a shim dir with symlinks. Returns the shim path, or None if unusable.
    """
    cu13 = VENV_CUDA_HOME
    if not os.path.isdir(os.path.join(cu13, "bin")):
        return None
    shim = "/workspace/.venv-vllm/cuda-home-shim"
    lib64 = os.path.join(shim, "lib64")
    os.makedirs(lib64, exist_ok=True)
    for sub in ("bin", "include", "nvvm", "cccl"):
        src = os.path.join(cu13, sub)
        if os.path.isdir(src):
            dst = os.path.join(shim, sub)
            if not os.path.lexists(dst):
                os.symlink(src, dst)
    links = {
        "libcudart.so": os.path.join(cu13, "lib", "libcudart.so.13"),
        "libcuda.so": "/lib/x86_64-linux-gnu/libcuda.so",
    }
    for name, src in links.items():
        dst = os.path.join(lib64, name)
        if not os.path.lexists(dst) and os.path.exists(src):
            os.symlink(src, dst)
    return shim


_SHIM = _ensure_cuda_home_shim()
if _SHIM:
    os.environ.setdefault("CUDA_HOME", _SHIM)
# flashinfer 0.6.18 bundles an older CCCL whose cuda_toolkit.h version check rejects the
# CUDA 13.3 nvcc ("CUDA compiler and CUDA toolkit headers are incompatible"). nvcc honors
# NVCC_PREPEND_FLAGS (verified), and CCCL itself documents this opt-out macro.
os.environ.setdefault("NVCC_PREPEND_FLAGS", "-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK")

MODEL_DIR = "/workspace/models/minicpm5-2b"
OUT_JSON = "/workspace/benchmark/results/vllm_sweep.json"

UTIL_THRESHOLD_PCT = 5
ROUNDS = 5
BIGCTX_ROUNDS = 3
BIGCTX_MIN_TOKENS = 65536  # cases with prompt_tokens >= this use BIGCTX_ROUNDS
CASES = [
    {"name": "ctx16_out256", "prompt_tokens": 16, "output_tokens": 256},
    {"name": "ctx512_out128", "prompt_tokens": 512, "output_tokens": 128},
    {"name": "ctx2048_out128", "prompt_tokens": 2048, "output_tokens": 128},
    {"name": "ctx8192_out128", "prompt_tokens": 8192, "output_tokens": 128},
    {"name": "ctx32768_out128", "prompt_tokens": 32768, "output_tokens": 128},
    {"name": "ctx65536_out128", "prompt_tokens": 65536, "output_tokens": 128},
    {"name": "ctx130816_out128", "prompt_tokens": 130816, "output_tokens": 128},
]
# random prompt token ids sampled from [2, 130000): avoids bos/pad(0/1) and eos(1, 130073)
PROMPT_SEED = 20260909
PROMPT_ID_RANGE = (2, 130000)
LOG_KEEP_PATTERNS = [
    r"kv cache", r"attention", r"backend", r"weights take", r"maximum concurrency",
    r"cuda graph", r"graph capture", r"compil", r"engine", r"avail mem", r"memory profiling",
    r"deterministic", r"prefix cach", r"model config", r"device", r"dtype",
]
ENGINE_LOG_PATH = "/tmp/opencode/vllm_engine.log"


def sh(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()


def gpu_util_pct() -> int:
    out = sh(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"])
    return int(out.splitlines()[0])


def wait_gpu_idle(timeout_s: int = 600) -> int:
    """Block until GPU util <= threshold; returns util observed right before we proceed."""
    t0 = time.time()
    while True:
        u = gpu_util_pct()
        if u <= UTIL_THRESHOLD_PCT:
            return u
        if time.time() - t0 > timeout_s:
            raise RuntimeError(f"GPU stayed busy (util>{UTIL_THRESHOLD_PCT}%) for {timeout_s}s")
        print(f"  [wait] gpu util {u}% > {UTIL_THRESHOLD_PCT}%, retrying in 10s...", flush=True)
        time.sleep(10)


@contextmanager
def capture_fds(path: str, fds=(1, 2)):
    """fd-level redirect of stdout+stderr: vLLM logs to stdout, tqdm to stderr,
    and child processes (EngineCore) inherit both."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved = {fd: os.dup(fd) for fd in fds}
    f = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        for fd in fds:
            os.dup2(f, fd)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        for fd in fds:
            os.dup2(saved[fd], fd)
            os.close(saved[fd])
        os.close(f)


def parse_engine_log(lines: list[str]) -> list[str]:
    """Filter interesting engine-init lines; drops tqdm progress noise."""
    kept, seen = [], set()
    for line in lines:
        l = line.rstrip()
        if not l or l in seen:
            continue
        low = l.lower()
        if any(t in low for t in ("it/s]", "%|", "completed |", "it]")):
            continue  # tqdm progress bars
        if any(re.search(p, low) for p in LOG_KEEP_PATTERNS):
            seen.add(l)
            kept.append(l[:400])
        if len(kept) >= 120:
            break
    return kept


class _ListHandler(logging.Handler):
    def __init__(self, sink: list):
        super().__init__(level=logging.INFO)
        self.sink = sink

    def emit(self, record):
        self.sink.append(self.format(record))


def build_prompt_ids(n: int) -> list[int]:
    rng = random.Random(PROMPT_SEED + n)
    lo, hi = PROMPT_ID_RANGE
    return [rng.randrange(lo, hi) for _ in range(n)]


def make_llm(gpu_mem_util: float):
    from vllm import LLM
    kw = dict(
        model=MODEL_DIR,
        dtype="bfloat16",
        max_model_len=131072,
        gpu_memory_utilization=gpu_mem_util,
        enforce_eager=False,          # default: CUDA graphs on
        enable_prefix_caching=False,  # force full cold prefill every round
        # offline LLM forces disable_log_stats=True (llm.py), which nulls out
        # RequestOutput.metrics; re-enable to get TTFT/TPOT per request.
        disable_log_stats=False,
    )
    return LLM(**kw)


def extract_metrics(output) -> dict:
    m = getattr(output, "metrics", None)
    gen_tokens = len(output.outputs[0].token_ids)
    prompt_tokens = len(output.prompt_token_ids)
    res = {
        "prompt_tokens": prompt_tokens,
        "gen_tokens": gen_tokens,
        "metric_source": "engine_metrics",
    }
    ttft = getattr(m, "first_token_latency", None) if m else None
    ftt = getattr(m, "first_token_ts", None) if m else None
    ltt = getattr(m, "last_token_ts", None) if m else None
    if ttft is not None:
        res["ttft_ms"] = ttft * 1000.0
    if ftt is not None and ltt is not None and gen_tokens > 1:
        decode_s = ltt - ftt
        res["decode_time_ms"] = decode_s * 1000.0
        res["tpot_ms"] = decode_s / (gen_tokens - 1) * 1000.0
    if ttft is None and (ftt is None or ltt is None):
        res["metric_source"] = "missing"
    return res


def run_case(llm, case: dict) -> dict:
    from vllm import SamplingParams
    ids = build_prompt_ids(case["prompt_tokens"])
    prompt = [{"prompt_token_ids": ids}]

    rounds_n = BIGCTX_ROUNDS if case["prompt_tokens"] >= BIGCTX_MIN_TOKENS else ROUNDS

    # warmup: short output
    wait_gpu_idle()
    sp_warm = SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True)
    llm.generate(prompt, sampling_params=sp_warm, use_tqdm=False)

    rounds = []
    sp = SamplingParams(temperature=0.0, max_tokens=case["output_tokens"], ignore_eos=True)
    for r in range(1, rounds_n + 1):
        util_before = wait_gpu_idle()
        t0 = time.perf_counter()
        outs = llm.generate(prompt, sampling_params=sp, use_tqdm=False)
        wall_ms = (time.perf_counter() - t0) * 1000.0
        assert len(outs) == 1, f"expected 1 output, got {len(outs)}"
        m = extract_metrics(outs[0])
        m["round"] = r
        m["util_before_pct"] = util_before
        m["wall_e2e_ms"] = wall_ms
        if "ttft_ms" in m and m["prompt_tokens"] > 0:
            m["prefill_tok_per_s"] = m["prompt_tokens"] / (m["ttft_ms"] / 1000.0)
        if "tpot_ms" in m:
            m["tok_per_s"] = 1000.0 / m["tpot_ms"]
        rounds.append(m)
        print(f"  round {r}/{rounds_n}: TTFT {m.get('ttft_ms', float('nan')):.2f}ms "
              f"TPOT {m.get('tpot_ms', float('nan')):.3f}ms "
              f"({m.get('tok_per_s', float('nan')):.1f} tok/s) "
              f"prefill {m.get('prefill_tok_per_s', float('nan')):.0f} tok/s "
              f"wall {wall_ms:.0f}ms util_before {util_before}%", flush=True)
        time.sleep(1.0)

    def med(key):
        vals = [r[key] for r in rounds if key in r]
        return statistics.median(vals) if vals else None

    def mean(key):
        vals = [r[key] for r in rounds if key in r]
        return sum(vals) / len(vals) if vals else None

    mean_tpot = mean("tpot_ms")

    return {
        "name": case["name"],
        "prompt_tokens": case["prompt_tokens"],
        "output_tokens": case["output_tokens"],
        "rounds_used": rounds_n,
        "rounds": rounds,
        "median": {
            "ttft_ms": med("ttft_ms"),
            "tpot_ms": med("tpot_ms"),
            "prefill_tok_per_s": med("prefill_tok_per_s"),
            "tok_per_s": med("tok_per_s"),
            "wall_e2e_ms": med("wall_e2e_ms"),
        },
        "mean": {
            "ttft_ms": mean("ttft_ms"),
            "tpot_ms": mean_tpot,
            "prefill_tok_per_s": mean("prefill_tok_per_s"),
            "tok_per_s": mean("tok_per_s"),
            "tok_per_s_from_mean_tpot": (1000.0 / mean_tpot) if mean_tpot else None,
            "wall_e2e_ms": mean("wall_e2e_ms"),
        },
    }


def main() -> int:
    os.makedirs(os.path.dirname(ENGINE_LOG_PATH), exist_ok=True)
    import torch
    import vllm

    driver = sh(["nvidia-smi", "--query-gpu=driver_version,name,memory.total", "--format=csv,noheader,nounits"])
    driver_ver, gpu_name, gpu_total_mb = [x.strip() for x in driver.split(",")]

    mem_util = 0.85
    llm = None
    oom_notes = []
    log_sink: list[str] = []
    handler = _ListHandler(log_sink)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(handler)
    logging.getLogger("vllm").addHandler(handler)
    for attempt_util in (0.85, 0.7):
        print(f"Initializing vLLM LLM (gpu_memory_utilization={attempt_util}) ...", flush=True)
        try:
            with capture_fds(ENGINE_LOG_PATH):
                llm = make_llm(attempt_util)
            mem_util = attempt_util
            break
        except (torch.OutOfMemoryError, MemoryError) as e:
            oom_notes.append(f"gpu_memory_utilization={attempt_util} OOM: {type(e).__name__}; retrying lower")
            print(f"  OOM at {attempt_util}, falling back", flush=True)
            gc.collect(); torch.cuda.empty_cache()
    logging.getLogger().removeHandler(handler)
    logging.getLogger("vllm").removeHandler(handler)
    if llm is None:
        print("ERROR: could not initialize vLLM LLM even at gpu_memory_utilization=0.7", flush=True)
        return 1

    # vLLM engine INFO lines (KV cache size, backends, ...) go through the logging
    # framework, not raw fd 2; merge handler records with the fd-level capture.
    try:
        with open(ENGINE_LOG_PATH, errors="replace") as f:
            fd_lines = f.read().splitlines()
    except OSError:
        fd_lines = []
    all_lines = log_sink + fd_lines
    engine_log = parse_engine_log(all_lines)

    def find_line(*pats):
        for l in all_lines:
            if all(re.search(p, l, re.I) for p in pats):
                return l.strip()
        return None

    m = None
    for l in all_lines:
        mm = re.search(r"GPU KV cache size:\s*([\d,]+)\s*tokens", l)
        if mm:
            m = l.strip()
            break
    engine_info = {
        "kv_cache_size_tokens": (int(re.search(r"GPU KV cache size:\s*([\d,]+)", m).group(1).replace(",", ""))
                                 if m else None),
        "kv_cache_line": m,
        "max_concurrency_line": find_line(r"maximum concurrency"),
        "available_kv_mem_line": find_line(r"available kv cache memory"),
        "attention_backend_lines": [l.strip() for l in all_lines
                                    if re.search(r"using\s+\w+\s+attention backend", l, re.I)
                                    or re.search(r"using flashattention version", l, re.I)],
        "kv_layout_line": find_line(r"kv cache layout"),
        "sampler_line": find_line(r"sampling"),
    }

    cases = []
    for case in CASES:
        print(f"== case {case['name']} (prompt {case['prompt_tokens']}, out {case['output_tokens']}) ==", flush=True)
        try:
            cases.append(run_case(llm, case))
        except Exception as e:
            # a single case failing (e.g. unexpected OOM at 131072 ctx) must not abort the sweep
            cases.append({"name": case["name"], "error": str(e)})
            print(f"  case {case['name']} FAILED: {type(e).__name__}: {e}; continuing with next case", flush=True)
            gc.collect()
            torch.cuda.empty_cache()

    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "purpose": ("vLLM context-length sweep 0-128K on MiniCPM5-2B (ctx 16/512/2048/8192/32768/65536/130816, "
                    "max_model_len=131072; top tier 131072-256 leaves room for output), RTX 5080; "
                    "native-runtime cross-reference is merged offline in a later step"),
        "environment": {
            "vllm": vllm.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "driver": driver_ver,
            "gpu": gpu_name,
            "gpu_total_mb": int(gpu_total_mb),
            "python": sys.version.split()[0],
            "date": datetime.now().strftime("%Y-%m-%d"),
        },
        "config": {
            "model_dir": MODEL_DIR,
            "dtype": "bfloat16",
            "max_model_len": 131072,
            "gpu_memory_utilization_requested": 0.85,
            "gpu_memory_utilization_actual": mem_util,
            "enforce_eager": False,
            "enable_prefix_caching": False,
            "batch_size": 1,
            "sampling": "greedy temperature=0, ignore_eos=True (exact output counts)",
            "eos_token_ids_in_config": [1, 130073],
        },
        "prompt_construction": (f"uniform random token ids in [{PROMPT_ID_RANGE[0]}, {PROMPT_ID_RANGE[1]}), "
                                f"seed {PROMPT_SEED}+prompt_len; avoids special tokens (0/1) and both eos ids"),
        "metric_definitions": {
            "ttft_ms": "RequestOutput.metrics.first_token_latency",
            "tpot_ms": "(metrics.last_token_ts - metrics.first_token_ts)/(gen_tokens-1), engine monotonic clock",
            "prefill_tok_per_s": "prompt_tokens / TTFT (scheduling included; ctx16 is scheduler-dominated)",
            "wall_e2e_ms": "perf_counter around llm.generate (cross-check)",
            "mean": ("arithmetic mean over the raw per-round values of each metric; "
                     "mean.tok_per_s = mean of per-round 1000/tpot, while "
                     "mean.tok_per_s_from_mean_tpot = 1000/mean(tpot_ms) (harmonic-mean equivalent)"),
        },
        "discipline": (f"idle-GPU check (util<={UTIL_THRESHOLD_PCT}%) before warmup and every round; "
                       f"1 warmup (short output) then {ROUNDS} timed rounds per case, "
                       f"{BIGCTX_ROUNDS} rounds for prompt_tokens>={BIGCTX_MIN_TOKENS}; median and mean reported"),
        "engine_info": engine_info,
        "engine_log_excerpts": engine_log,
        "cases": cases,
        "notes": [
            "torch 2.13.0+cu130 installed as vllm 0.29.0 default dependency; sm_120 kernels present, no cu128 fallback needed",
            "pitfall 1: system lacked python3.12-dev (Python.h); triton cuda_utils.c build failed -> fixed via apt python3.12-dev",
            "pitfall 2: host /usr/local/cuda is 12.8; flashinfer refuses SM12.x with nvcc<12.9 -> CUDA_HOME pointed at venv CUDA 13.3 wheel (with a lib64 shim for libcudart/libcuda dev symlinks)",
            "pitfall 3: flashinfer's bundled CCCL rejects CUDA 13.3 nvcc -> NVCC_PREPEND_FLAGS=-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",
            "pitfall 4: offline LLM() forces disable_log_stats=True which nulls RequestOutput.metrics -> disable_log_stats=False passed explicitly",
            "prefix caching disabled on purpose: identical prompt across rounds would hit KV cache and fake TTFT",
            ("max_model_len=131072 (model config max_position_embeddings=131072, rope_scaling=null -> natively supported, "
             "no scaling needed); single-sequence full-length KV = 131072 tok x 42 layers x 2 KV heads x 128 dim "
             "x 2 (K+V) x 2 bytes (BF16) ~= 5.64 GB"),
            "a failing case (e.g. unexpected OOM at ctx130816) is recorded as {name, error} and does not abort the sweep; engine init failure still aborts",
            "top tier prompt is 130816 (not 131072): prompt_tokens + output_tokens must fit max_model_len, else vLLM truncates generation and TPOT is undefined",
        ] + oom_notes,
    }

    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {OUT_JSON}", flush=True)

    # teardown
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
