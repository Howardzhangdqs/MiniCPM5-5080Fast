#!/usr/bin/env python3
"""Distribution-level similarity: native runtime vs vLLM greedy next-token distributions.

Design (user-specified): small prompt set (12 from crosscheck_prompts.json, all 9
categories, 2 fixed-history multi-turn), greedy, max_tokens=32. Both ends capture the
per-step next-token distribution; distributions are compared position-by-position along
the COMMON PREFIX (positions where both ends selected the same token). The prefix walk
stops at the first selection disagreement; at that flip point we record both ends'
top-2 probability margins (near-tie evidence) and whether each end's top-1 is the
other's top-2.

Capture paths:
  native : BF16 minicpm-server started with
           MC_DEBUG_DUMP_DIR=/tmp/opencode/dist_dump, MC_DEBUG_DUMP_STEPS=<steps+1>.
           Each forward dumps step{t}.logits.f32 (130560 fp32, bf16 upcast — the SAME
           buffer the greedy argmax kernel reads) + step{t}.token (argmax). debug_step
           resets per request (session reset), so the next serial request OVERWRITES
           the same filenames -> archive to a per-prompt dir immediately after each
           response (response returns only after generation finished; no race).
           Note: dump-enabled runs disable the fused lm_head tail, which the runtime
           documents as bit-identical, so numerics are unchanged.
  vLLM   : llm.chat(..., SamplingParams(temperature=0, max_tokens=steps, logprobs=32));
           outputs[0].outputs[0].token_ids = selected ids, .logprobs = per-step
           {token_id: logprob} (top-32 incl. sampled), full-softmax probabilities.

Metrics per common-prefix position (on the union support of both top-32 sets,
renormalized; |dp1| and margins use the raw full-softmax values):
  JS divergence (base2), |p1_a - p1_b|, top-1 id match, top-10 id Jaccard, cosine.

GPU discipline: native phase first (all requests), server stopped + GPU release
verified, then vLLM, then compare (CPU-only, needs numpy -> venv python).

Usage (re-runnable):
  python3 dist_similarity.py all
  python3 dist_similarity.py native [--steps 32] [--prompts id1 id2 ...]
  /workspace/.venv-vllm/bin/python dist_similarity.py vllm   (must be venv python)
  /workspace/.venv-vllm/bin/python dist_similarity.py compare (needs numpy)

Artifacts:
  /tmp/opencode/dist_dump          live dump dir (native phase)
  /tmp/opencode/dist_native/<id>/  archived step dumps per prompt
  /tmp/opencode/dist_native.json   native phase meta
  /tmp/opencode/dist_vllm.json     vllm phase raw logprobs
  benchmark/results/dist_similarity.json   final report
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from crosscheck import (SERVER_BIN, NATIVE_LIB, MODEL_DIR, PORT, load_prompts,  # noqa: E402
                        wait_gpu_free, effective_finish)

PROMPTS_JSON = os.path.join(HERE, "crosscheck_prompts.json")
RESULTS_JSON = "/workspace/benchmark/results/dist_similarity.json"

VENV_PY = "/workspace/.venv-vllm/bin/python"
CUDA_HOME_SHIM = "/workspace/.venv-vllm/cuda-home-shim"
VLLM_LOG = "/tmp/opencode/dist_vllm.log"
VLLM_JSON = "/tmp/opencode/dist_vllm.json"
NATIVE_LOG = "/tmp/opencode/dist_server.log"
NATIVE_JSON = "/tmp/opencode/dist_native.json"
DUMP_DIR = "/tmp/opencode/dist_dump"
ARCHIVE_ROOT = "/tmp/opencode/dist_native"

TOPK = 32
EOS_IDS = (1, 130073)  # MiniCPM5-2B eos ids (scheduler EOS_IDS / tokenizer config)
DEFAULT_STEPS = 32
# 12 prompts: all 9 categories, 2 fixed-history multi-turn; mix of crosscheck outcomes
# (exact / prefix / early-flip / deep-flip / answer-level-divergent).
DEFAULT_PROMPTS = ["zh_01", "zh_04", "en_03", "code_01", "code_03", "math_01",
                   "tr_02", "mix_01", "cre_03", "com_01", "multi_01", "multi_02"]


def select_prompts(ids: list[str]) -> list[dict]:
    by_id = {p["id"]: p for p in load_prompts()}
    missing = [i for i in ids if i not in by_id]
    assert not missing, f"unknown prompt ids: {missing}"
    return [by_id[i] for i in ids]


# --------------------------------------------------------------------------- phase: native

def cmd_native(steps: int, ids: list[str]) -> None:
    prompts = select_prompts(ids)
    shutil.rmtree(DUMP_DIR, ignore_errors=True)
    shutil.rmtree(ARCHIVE_ROOT, ignore_errors=True)
    os.makedirs(DUMP_DIR)
    os.makedirs(ARCHIVE_ROOT)

    env = dict(os.environ,
               MC_DEBUG_DUMP_DIR=DUMP_DIR,
               MC_DEBUG_DUMP_STEPS=str(steps + 1))  # step0=prefill .. step{steps-1}
    log = open(NATIVE_LOG, "wb")
    print(f"[native] starting server with dump env -> {NATIVE_LOG}", flush=True)
    proc = subprocess.Popen(
        [SERVER_BIN, "--native-lib", NATIVE_LIB, "--model-dir", MODEL_DIR,
         "--precision", "bf16", "--port", str(PORT)],
        stdout=log, stderr=subprocess.STDOUT, env=env)
    try:
        t0 = time.time()
        while True:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
                    if r.status == 200:
                        break
            except Exception:
                pass
            if proc.poll() is not None:
                raise RuntimeError(f"server died (exit={proc.returncode}); see {NATIVE_LOG}")
            if time.time() - t0 > 300:
                raise RuntimeError("server health timeout")
            time.sleep(2)
        print("[native] server healthy; dump banner check:", flush=True)
        with open(NATIVE_LOG, errors="replace") as f:
            for line in f:
                if "debug dump enabled" in line:
                    print("  " + line.strip(), flush=True)
                    break

        results = []
        for p in prompts:
            body = json.dumps({"model": "minicpm5-2b", "messages": p["messages"],
                               "max_tokens": steps, "temperature": 0.0}).encode()
            req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                         body, {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                d = json.load(r)
            ch = d["choices"][0]
            # archive immediately: the next serial request overwrites step* files;
            # shorter generations leave a STALE TAIL from the previous request, so
            # the dump dir is wiped after archiving (fresh dir per request)
            arch = os.path.join(ARCHIVE_ROOT, p["id"])
            os.makedirs(arch, exist_ok=True)
            n_files = 0
            for f in glob.glob(os.path.join(DUMP_DIR, "step*")):
                shutil.copy(f, arch)
                n_files += 1
            for f in glob.glob(os.path.join(DUMP_DIR, "*")):
                os.remove(f)
            rec = {"id": p["id"], "category": p["category"],
                   "finish_reason": ch.get("finish_reason"),
                   "completion_tokens": d["usage"]["completion_tokens"],
                   "prompt_tokens": d["usage"]["prompt_tokens"],
                   "archived_step_files": n_files,
                   "dumped_steps": len(glob.glob(os.path.join(arch, "step*.logits.f32"))),
                   "text_head": ch["message"]["content"][:120]}
            results.append(rec)
            print(f"  [native] {p['id']}: {rec['finish_reason']} "
                  f"ctok={rec['completion_tokens']} ptok={rec['prompt_tokens']} "
                  f"dumped={rec['dumped_steps']} steps", flush=True)
    finally:
        print("[native] stopping server ...", flush=True)
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)
        log.close()
        wait_gpu_free()
        print("[native] GPU released", flush=True)

    with open(NATIVE_JSON, "w") as f:
        json.dump({"phase": "native", "steps": steps,
                   "timestamp": datetime.now().isoformat(timespec="seconds"),
                   "dump_dir_banner": f"MC_DEBUG_DUMP_DIR={DUMP_DIR} "
                                      f"MC_DEBUG_DUMP_STEPS={steps + 1}",
                   "results": results}, f, ensure_ascii=False, indent=1)
    print(f"[native] wrote {NATIVE_JSON}", flush=True)


# --------------------------------------------------------------------------- phase: vllm

def cmd_vllm(steps: int, ids: list[str]) -> None:
    os.environ.setdefault("CUDA_HOME", CUDA_HOME_SHIM)
    os.environ.setdefault("NVCC_PREPEND_FLAGS", "-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK")
    from vllm import LLM, SamplingParams

    prompts = select_prompts(ids)
    wait_gpu_free()
    print("[vllm] building LLM ...", flush=True)
    llm = LLM(model=MODEL_DIR, dtype="bfloat16", max_model_len=8448,
              gpu_memory_utilization=0.85, enable_prefix_caching=False,
              max_logprobs=64)  # default cap 20 would reject SamplingParams(logprobs=32)
    llm.chat([{"role": "user", "content": "hi"}],
             SamplingParams(temperature=0.0, max_tokens=4), use_tqdm=False)  # warmup

    sp = SamplingParams(temperature=0.0, max_tokens=steps, logprobs=TOPK)
    results = []
    for p in prompts:
        outs = llm.chat([p["messages"]], sp, use_tqdm=False)
        o = outs[0].outputs[0]
        fr = getattr(o.finish_reason, "value", o.finish_reason)
        lps = []
        for pos in o.logprobs:
            lps.append({str(tid): float(l.logprob) for tid, l in pos.items()})
        results.append({
            "id": p["id"], "category": p["category"],
            "finish_reason": str(fr) if fr is not None else None,
            "completion_tokens": len(o.token_ids),
            "prompt_tokens": len(outs[0].prompt_token_ids),
            "token_ids": list(o.token_ids),
            "logprobs": lps,
        })
        print(f"  [vllm] {p['id']}: {results[-1]['finish_reason']} "
              f"ctok={results[-1]['completion_tokens']} ptok={results[-1]['prompt_tokens']}",
              flush=True)

    with open(VLLM_JSON, "w") as f:
        json.dump({"phase": "vllm", "steps": steps,
                   "timestamp": datetime.now().isoformat(timespec="seconds"),
                   "results": results}, f, ensure_ascii=False, indent=1)
    print(f"[vllm] wrote {VLLM_JSON}", flush=True)


# --------------------------------------------------------------------------- phase: compare

def read_native_steps(pid: str) -> list[dict]:
    """[{step, top:[(id,p)...] desc, argmax, token_file}] for step0.. in order."""
    import numpy as np
    arch = os.path.join(ARCHIVE_ROOT, pid)
    files = glob.glob(os.path.join(arch, "step*.logits.f32"))
    steps = []
    for f in files:
        m = re.search(r"step(\d+)\.logits\.f32$", f)
        if m:
            steps.append(int(m.group(1)))
    out = []
    for t in sorted(steps):
        logits = np.fromfile(os.path.join(arch, f"step{t}.logits.f32"), dtype=np.float32)
        assert logits.size == 130560, f"{pid} step{t}: {logits.size} floats"
        x = logits.astype(np.float64)
        x -= x.max()
        p = np.exp(x)
        p /= p.sum()  # full-vocab softmax in fp64 (bf16-upcast logits)
        am = int(np.argmax(logits))
        tok_file = os.path.join(arch, f"step{t}.token")
        tok_dump = None
        if os.path.exists(tok_file):
            with open(tok_file, "rb") as fh:
                tok_dump = int.from_bytes(fh.read(4), "little", signed=True)
        k = min(TOPK, p.size)
        idx = np.argpartition(p, -k)[-k:]
        idx = idx[np.argsort(p[idx])[::-1]]
        top = [(int(i), float(p[i])) for i in idx]
        out.append({"step": t, "top": top, "argmax": am, "token_dump": tok_dump,
                    "p_full_argmax": float(p[am])})
    return out


def metrics_position(top_a: list[tuple[int, float]], top_b: list[tuple[int, float]]) -> dict:
    """JS(base2), cosine on union-normalized support; |dp1|, top1 match, top10 Jaccard."""
    da = dict(top_a)
    db = dict(top_b)
    support = set(da) | set(db)
    sa, sb = sum(da.values()), sum(db.values())
    na = {t: da.get(t, 0.0) / sa for t in support}
    nb = {t: db.get(t, 0.0) / sb for t in support}
    m = {t: 0.5 * (na[t] + nb[t]) for t in support}

    def kl(p, q):  # base2 KL over the union support (zeros contribute nothing)
        total = 0.0
        for t, v in p.items():
            if v > 0 and q.get(t, 0.0) > 0:
                total += v * math.log2(v / q[t])
        return total

    js = 0.5 * kl(na, m) + 0.5 * kl(nb, m)
    dot = sum(na[t] * nb[t] for t in support)
    ca = sum(v * v for v in na.values()) ** 0.5
    cb = sum(v * v for v in nb.values()) ** 0.5
    ta = sorted(top_a, key=lambda x: -x[1])[:10]
    tb = sorted(top_b, key=lambda x: -x[1])[:10]
    return {
        "js_base2": js,
        "cosine": dot / (ca * cb) if ca > 0 and cb > 0 else 1.0,
        "dp1": abs(da[top_a[0][0]] - db[top_b[0][0]]),
        "top1_match": top_a[0][0] == top_b[0][0],
        "top10_jaccard": len({i for i, _ in ta} & {i for i, _ in tb}) /
                         len({i for i, _ in ta} | {i for i, _ in tb}),
    }


def cmd_compare(steps: int, ids: list[str]) -> None:
    import numpy as np  # noqa: F401  (compare runs under the venv python)

    with open(NATIVE_JSON) as f:
        nmeta = {r["id"]: r for r in json.load(f)["results"]}
    with open(VLLM_JSON) as f:
        vraw = {r["id"]: r for r in json.load(f)["results"]}

    prompts = select_prompts(ids)
    per_prompt, all_js, all_dp1, all_cos, all_jac = [], [], [], [], []
    all_margins_prefix = []          # top-2 margins at agreement positions (both ends)
    div_records = []                 # flip-point evidence
    prefix_lens, top1_mismatch_positions = [], 0
    prompt_ptok_mismatch = []
    js_forensics = []                # (js, pid, t) for the worst positions

    for p in prompts:
        pid = p["id"]
        n_steps = read_native_steps(pid)
        v = vraw[pid]
        # selections are compared RAW, EOS included when a side decided to stop
        # (both ends have a distribution for their eos decision position; a shared
        # eos is then an agreeing position, an eos-vs-content pair a real flip)
        v_sel = list(v["token_ids"])
        v_eos_last = bool(v_sel and v_sel[-1] in EOS_IDS
                          and v["finish_reason"] == "stop")
        n_sel = [s["argmax"] for s in n_steps]
        # the engine performs one extra forward after its first eos decision
        # (e.g. multi_01: ..., 396, eos@29, eos@30): keep the eos DECISION position,
        # drop any post-eos bookkeeping steps so both selection streams end at eos
        first_eos = next((i for i, t in enumerate(n_sel) if t in EOS_IDS), None)
        if first_eos is not None and first_eos < len(n_sel) - 1:
            n_steps = n_steps[:first_eos + 1]
            n_sel = n_sel[:first_eos + 1]
            post_eos_trimmed = len(n_steps)
        # stale-dump guard: a request must dump exactly its forwards
        # (completion_tokens, +1 when the eos decision is not counted in ctok)
        ctok = nmeta[pid]["completion_tokens"]
        steps_anomaly = not (ctok <= len(n_steps) <= ctok + 3)

        if nmeta[pid]["prompt_tokens"] != v["prompt_tokens"]:
            prompt_ptok_mismatch.append((pid, nmeta[pid]["prompt_tokens"], v["prompt_tokens"]))

        # sanity: greedy top-1 == selected on both ends; native argmax == dumped token
        n_tokdump_mismatch = sum(1 for s in n_steps if s["token_dump"] is not None
                                 and s["token_dump"] != s["argmax"])
        v_top1_mismatch = 0

        pos_metrics = []
        L = min(len(n_sel), len(v_sel))
        div_t = None
        for t in range(L):
            if n_sel[t] != v_sel[t]:
                div_t = t
                break
            top_a = n_steps[t]["top"]
            top_b = [(int(k), math.exp(lp)) for k, lp in v["logprobs"][t].items()]
            top_b.sort(key=lambda x: -x[1])
            v_top1_mismatch += 0 if (top_b and top_b[0][0] == v_sel[t]) else 1
            m = metrics_position(top_a, top_b)
            m["t"] = t
            pos_metrics.append(m)
            all_js.append(m["js_base2"])
            js_forensics.append((m["js_base2"], pid, t, top_a[0][0], top_b[0][0]))
            all_dp1.append(m["dp1"])
            all_cos.append(m["cosine"])
            all_jac.append(m["top10_jaccard"])
            # top-2 margins at agreement positions (both ends, raw full-softmax)
            for top in (top_a, top_b):
                all_margins_prefix.append(top[0][1] - top[1][1])

        # classify the walk end
        ended = {"ours": nmeta[pid]["finish_reason"], "vllm": v["finish_reason"]}
        n_eos_last = bool(n_sel and n_sel[-1] in EOS_IDS
                          and nmeta[pid]["finish_reason"] == "stop")
        if div_t is not None:
            # covers content flips AND eos-vs-continue (one end's top-1 is an eos id;
            # both ends always have a distribution at div_t)
            end_kind = "selection_divergence"
        elif len(n_sel) != len(v_sel):
            # shorter side hit its token cap without an eos decision position
            end_kind = "cap_boundary_no_divergence"
        else:
            end_kind = "no_divergence_within_steps"

        # flip-point evidence (both ends have a distribution at div_t)
        flip = None
        if div_t is not None and div_t < len(n_steps) and div_t < len(v["logprobs"]):
            top_a = n_steps[div_t]["top"]
            top_b = [(int(k), math.exp(lp)) for k, lp in v["logprobs"][div_t].items()]
            top_b.sort(key=lambda x: -x[1])
            flip = {
                "t": div_t,
                "ours_top4": [(i, round(p, 6)) for i, p in top_a[:4]],
                "vllm_top4": [(i, round(p, 6)) for i, p in top_b[:4]],
                "ours_margin": top_a[0][1] - top_a[1][1],
                "vllm_margin": top_b[0][1] - top_b[1][1],
                "ours_top1_is_vllm_top2": top_a[0][0] == (top_b[1][0] if len(top_b) > 1 else -1),
                "vllm_top1_is_ours_top2": top_b[0][0] == (top_a[1][0] if len(top_a) > 1 else -1),
                "ours_selected": n_sel[div_t], "vllm_selected": v_sel[div_t],
                "eos_involved": n_sel[div_t] in EOS_IDS or v_sel[div_t] in EOS_IDS,
            }
            div_records.append({"id": pid, "category": p["category"], **flip})

        top1_mismatch_positions += v_top1_mismatch
        prefix_lens.append(len(pos_metrics))
        per_prompt.append({
            "id": pid, "category": p["category"],
            "prompt_tokens": {"ours": nmeta[pid]["prompt_tokens"], "vllm": v["prompt_tokens"]},
            "ours_steps_dumped": len(n_steps), "vllm_tokens": len(v_sel),
            "steps_vs_ctok_anomaly": steps_anomaly,
            "ours_last_is_eos": n_eos_last,
            "vllm_last_is_eos": v_eos_last,
            "finish": ended,
            "common_prefix_positions": len(pos_metrics),
            "end_kind": end_kind,
            "native_tokendump_argmax_mismatch": n_tokdump_mismatch,
            "vllm_top1_vs_selected_mismatch": v_top1_mismatch,
            "position_metrics_summary": _stats([m["js_base2"] for m in pos_metrics], "js"),
            "flip": flip,
        })

    def pooled_stats(vals):
        return _stats(vals, "")

    margins_div = ([r["ours_margin"] for r in div_records]
                   + [r["vllm_margin"] for r in div_records])
    typical_margin = _stats(all_margins_prefix, "top2_margin_prefix")
    div_margin_stats = _stats(margins_div, "top2_margin_flip")
    cross_top2 = sum(1 for r in div_records if r["ours_top1_is_vllm_top2"]
                     or r["vllm_top1_is_ours_top2"])

    js_stats, dp1_stats = pooled_stats(all_js), pooled_stats(all_dp1)
    cos_stats, jac_stats = pooled_stats(all_cos), pooled_stats(all_jac)

    # judgment inputs
    numerical_equiv = js_stats["median"] is not None and js_stats["median"] < 1e-3 \
        and cos_stats["mean"] is not None and cos_stats["mean"] > 0.999
    margin_ratio = None
    if typical_margin["mean"] and div_margin_stats["mean"] is not None:
        margin_ratio = div_margin_stats["mean"] / typical_margin["mean"]
    all_near_ties = bool(margins_div) and all(
        m < typical_margin["mean"] / 10 for m in margins_div) if typical_margin["mean"] else None
    # complementary criterion: a flip is near-tied if EITHER end shows a small top-2
    # margin (the ends share candidates; one end's softmax can be softer than the other's)
    min_side_margins = [min(r["ours_margin"], r["vllm_margin"]) for r in div_records]
    any_end_near_ties = bool(min_side_margins) and all(
        m < typical_margin["mean"] / 10 for m in min_side_margins) if typical_margin["mean"] else None

    n_div = sum(1 for r in per_prompt if r["end_kind"] == "selection_divergence")
    summary = {
        "prompts": len(prompts), "steps": steps,
        "positions_compared": len(all_js),
        "mean_common_prefix_positions": _mean(prefix_lens),
        "end_kinds": {k: sum(1 for r in per_prompt if r["end_kind"] == k)
                      for k in {r["end_kind"] for r in per_prompt}},
        "js_base2": js_stats, "dp1": dp1_stats, "cosine": cos_stats, "top10_jaccard": jac_stats,
        "top1_id_mismatch_positions": top1_mismatch_positions,
        "divergence_margins": {
            "flip_points": len(div_records),
            "flip_margin_stats": div_margin_stats,
            "flip_min_side_margin_stats": _stats(min_side_margins, "top2_margin_flip_min_side"),
            "prefix_top2_margin_stats": typical_margin,
            "flip_vs_prefix_margin_ratio": margin_ratio,
            "all_flips_below_tenth_of_typical_margin": all_near_ties,
            "all_flips_below_tenth_on_either_end": any_end_near_ties,
            "cross_top2_count": f"{cross_top2}/{len(div_records)} flips have one end's "
                                "top-1 as the other end's top-2",
        },
        "worst_js_positions": [
            {"js": js, "id": p, "t": t, "ours_top1": a1, "vllm_top1": b1}
            for js, p, t, a1, b1 in sorted(js_forensics, reverse=True)[:5]
        ],
        "sanity": {
            "prompt_token_mismatches": prompt_ptok_mismatch,
            "native_stale_step_anomalies": sum(
                1 for r in per_prompt if r["steps_vs_ctok_anomaly"]),
            "native_tokendump_vs_argmax_mismatches": sum(
                r["native_tokendump_argmax_mismatch"] for r in per_prompt),
            "vllm_top1_vs_selected_mismatches": top1_mismatch_positions,
        },
        "judgment": {
            "numerically_equivalent(JS_med<1e-3 & cos>0.999)": numerical_equiv,
            "flips_all_near_tie(margin<typ/10)": all_near_ties,
            "flips_near_tied_on_either_end(min_margin<typ/10)": any_end_near_ties,
        },
        "verdict": "",
    }
    summary["verdict"] = (
        f"{len(all_js)} common-prefix positions across {len(prompts)} prompts: "
        f"JS median {js_stats['median']:.2e} (max {js_stats['max']:.2e}), "
        f"cosine mean {cos_stats['mean']:.6f}, |dp1| median {dp1_stats['median']:.2e}; "
        f"{n_div}/{len(prompts)} prompts hit a flip within {steps} steps, flip margins "
        f"avg {div_margin_stats['mean']:.2e} vs prefix-internal avg "
        f"{typical_margin['mean']:.2e} (ratio {margin_ratio:.3f}) — "
        + ("distributions numerically equivalent; flips are near-ties."
           if numerical_equiv and all_near_ties is not False else
           "see judgment flags." if numerical_equiv else
           "distributions differ beyond near-tie noise; inspect per-prompt metrics.")
    )

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "purpose": "distribution-level similarity of greedy next-token distributions on the "
                   "common prefix: native bf16 runtime vs vLLM (MiniCPM5-2b, 12 prompts, "
                   "max_tokens=32, top-32 capture both ends)",
        "inputs": {"prompts": PROMPTS_JSON, "prompt_ids": ids,
                   "native_meta": NATIVE_JSON, "native_archive": ARCHIVE_ROOT,
                   "vllm_raw": VLLM_JSON, "server_log": NATIVE_LOG, "vllm_log": VLLM_LOG,
                   "dump_env": f"MC_DEBUG_DUMP_DIR={DUMP_DIR} MC_DEBUG_DUMP_STEPS={steps+1}"},
        "method": {
            "native": "server step dumps (bf16 logits upcast to f32 — same buffer the "
                      "greedy argmax kernel reads); softmax in fp64 over full 130560 vocab",
            "vllm": "SamplingParams(logprobs=32) full-softmax logprobs",
            "position_metric": "union of both top-32 supports, renormalized (JS base2, "
                               "cosine, top-10 Jaccard); |dp1| and margins use raw "
                               "full-softmax values",
            "prefix_rule": "walk while selected tokens agree; stop at first disagreement "
                           "(or natural-EOS-vs-continue); record top-2 margins there",
        },
        "summary": summary,
        "per_prompt": per_prompt,
        "flip_points": div_records,
    }
    os.makedirs(os.path.dirname(RESULTS_JSON), exist_ok=True)
    with open(RESULTS_JSON, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)

    print(f"\n=== dist similarity -> {RESULTS_JSON} ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "verdict"}, indent=1,
                     default=str))
    print(f"\nVERDICT: {summary['verdict']}")
    for r in div_records:
        print(f"  flip {r['id']} ({r['category']}) @t={r['t']}: "
              f"ours top2={r['ours_top4'][:2]} vllm top2={r['vllm_top4'][:2]} "
              f"margins {r['ours_margin']:.2e}/{r['vllm_margin']:.2e} "
              f"cross={r['ours_top1_is_vllm_top2'] or r['vllm_top1_is_ours_top2']}")


def _stats(vals: list[float], tag: str) -> dict:
    if not vals:
        return {"n": 0, "mean": None, "median": None, "p95": None, "max": None, "min": None,
                "tag": tag}
    s = sorted(vals)

    def pct(q):
        k = (len(s) - 1) * q
        f, c = int(k), min(int(k) + 1, len(s) - 1)
        return s[f] + (s[c] - s[f]) * (k - f)

    return {"n": len(s), "mean": sum(s) / len(s), "median": pct(0.5), "p95": pct(0.95),
            "max": s[-1], "min": s[0], "tag": tag}


def _mean(vals):
    return sum(vals) / len(vals) if vals else None


# --------------------------------------------------------------------------- driver

def cmd_all(steps: int, ids: list[str]) -> None:
    wait_gpu_free()
    cmd_native(steps, ids)
    env = dict(os.environ, CUDA_HOME=CUDA_HOME_SHIM,
               NVCC_PREPEND_FLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK")
    for phase in ("vllm", "compare"):
        wait_gpu_free()
        print(f"\n[all] phase {phase} (venv python)", flush=True)
        logf = open(VLLM_LOG if phase == "vllm" else "/tmp/opencode/dist_compare.log", "ab")
        try:
            r = subprocess.run([VENV_PY, os.path.abspath(__file__), phase,
                                "--steps", str(steps), "--prompts", *ids],
                               env=env, stdout=logf, stderr=subprocess.STDOUT)
        finally:
            logf.close()
        if r.returncode != 0:
            raise RuntimeError(f"{phase} phase failed rc={r.returncode}")
    with open(RESULTS_JSON) as f:
        print("\n[all] final summary:\n" + json.dumps(json.load(f)["summary"],
                                                      ensure_ascii=False, indent=1, default=str),
              flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["native", "vllm", "compare", "all"])
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS)
    args = ap.parse_args()
    if args.phase == "native":
        cmd_native(args.steps, args.prompts)
    elif args.phase == "vllm":
        cmd_vllm(args.steps, args.prompts)
    elif args.phase == "compare":
        cmd_compare(args.steps, args.prompts)
    else:
        cmd_all(args.steps, args.prompts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
