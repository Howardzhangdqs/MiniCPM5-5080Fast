#!/usr/bin/env python3
"""两种推理后端在相同数据集上的输出相似度指标。

用法:
    .venv/bin/python benchmark/scripts/similarity_metrics.py \
        --ours /tmp/opencode/crosscheck_ours.json \
        --vllm /tmp/opencode/crosscheck_vllm.json \
        [--out benchmark/results/similarity_metrics.json]

输入为 crosscheck.py 产出的 JSON（results[].text 按 id+max_tokens 配对）。
指标（全文与去 <think> 正文分别计算）:
  - exact        完全一致率
  - prefix_ratio 公共前缀字符比（分歧深度）
  - lev_ratio    编辑相似度（difflib SequenceMatcher.ratio）
  - bleu4        token 级 BLEU-4（+1 平滑；模型自身 tokenizer，中英一致）
  - rouge_l      token 级 ROUGE-L F1
聚合: 总体 mean / median、按 category、按 max_tokens。
"""
import argparse
import difflib
import json
import statistics
from collections import defaultdict

from tokenizers import Tokenizer

TOKENIZER = None


def toks(text):
    global TOKENIZER
    if TOKENIZER is None:
        TOKENIZER = Tokenizer.from_file("models/minicpm5-2b/tokenizer.json")
    return TOKENIZER.encode(text, add_special_tokens=False).ids


def common_prefix_chars(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def bleu4(hyp_ids, ref_ids):
    """+1 平滑的 token 级 BLEU-4；空句返回 0。"""
    if not hyp_ids or not ref_ids:
        return 0.0
    import math
    log_p = 0.0
    for n in range(1, 5):
        h_ngrams = defaultdict(int)
        for i in range(len(hyp_ids) - n + 1):
            h_ngrams[tuple(hyp_ids[i : i + n])] += 1
        match = 0
        total = max(len(ref_ids) - n + 1, 0)
        r_ngrams = defaultdict(int)
        for i in range(len(ref_ids) - n + 1):
            r_ngrams[tuple(ref_ids[i : i + n])] += 1
        for g, c in h_ngrams.items():
            match += min(c, r_ngrams.get(g, 0))
        log_p += math.log((match + 1.0) / (total + 1.0))
    bp = min(1.0, math.exp(1 - len(ref_ids) / len(hyp_ids))) if hyp_ids else 0.0
    return bp * math.exp(log_p / 4)


def lcs_len(a, b):
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            cur[j] = prev[j - 1] + 1 if ai == b[j - 1] else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def rouge_l(hyp_ids, ref_ids):
    if not hyp_ids or not ref_ids:
        return 0.0
    l = lcs_len(hyp_ids, ref_ids)
    p, r = l / len(hyp_ids), l / len(ref_ids)
    return 2 * p * r / (p + r) if p + r else 0.0


def strip_think(s):
    return s.split("</think>")[-1].strip() if "</think>" in s else s


def pair_metrics(a, b):
    m = {
        "exact": a == b,
        "prefix_ratio": common_prefix_chars(a, b) / max(len(a), len(b), 1),
        "lev_ratio": difflib.SequenceMatcher(None, a, b).ratio(),
    }
    ta, tb = toks(a), toks(b)
    m["bleu4"] = bleu4(ta, tb)
    m["rouge_l"] = rouge_l(ta, tb)
    return m


def agg(rows):
    out = {"n": len(rows)}
    if not rows:
        return out
    out["exact_rate"] = sum(r["exact"] for r in rows) / len(rows)
    for k in ("prefix_ratio", "lev_ratio", "bleu4", "rouge_l"):
        vals = [r[k] for r in rows]
        out[f"{k}_mean"] = statistics.mean(vals)
        out[f"{k}_median"] = statistics.median(vals)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True)
    ap.add_argument("--vllm", required=True)
    ap.add_argument("--out", default="benchmark/results/similarity_metrics.json")
    args = ap.parse_args()

    ours = {(r["id"], r["max_tokens"]): r for r in json.load(open(args.ours))["results"]}
    vllm = {(r["id"], r["max_tokens"]): r for r in json.load(open(args.vllm))["results"]}
    keys = sorted(set(ours) & set(vllm))
    print(f"配对 {len(keys)} 组（ours {len(ours)}，vllm {len(vllm)}）")

    full_rows, body_rows, by_cat, by_mt, details = [], [], defaultdict(list), defaultdict(list), []
    for k in keys:
        a, b, cat, mt = ours[k]["text"], vllm[k]["text"], ours[k]["category"], ours[k]["max_tokens"]
        mf, mb = pair_metrics(a, b), pair_metrics(strip_think(a), strip_think(b))
        full_rows.append(mf)
        body_rows.append(mb)
        by_cat[cat].append((mf, mb))
        by_mt[mt].append((mf, mb))
        details.append({"id": k[0], "max_tokens": k[1], "category": cat, "full": mf, "body": mb})

    report = {
        "pairs": len(keys),
        "overall": {"full_text": agg(full_rows), "body_after_think": agg(body_rows)},
        "by_category": {c: {"full_text": agg([f for f, _ in rs]), "body_after_think": agg([b for _, b in rs])} for c, rs in sorted(by_cat.items())},
        "by_max_tokens": {str(mt): {"full_text": agg([f for f, _ in rs]), "body_after_think": agg([b for _, b in rs])} for mt, rs in sorted(by_mt.items())},
        "details": details,
        "metric_notes": "exact=完全一致; prefix_ratio=公共前缀字符比; lev_ratio=编辑相似度; bleu4=token级BLEU-4(+1平滑, 模型tokenizer); rouge_l=token级ROUGE-L F1; full=含<think>全文, body=去think正文",
    }
    json.dump(report, open(args.out, "w"), ensure_ascii=False, indent=1)

    def row(name, a):
        print(f"  {name:24s} n={a['n']:3d}  exact={a.get('exact_rate', 0):6.1%}  prefix={a.get('prefix_ratio_mean', 0):.3f}  "
              f"lev={a.get('lev_ratio_mean', 0):.3f}  bleu4={a.get('bleu4_mean', 0):.3f}  rougeL={a.get('rouge_l_mean', 0):.3f}")

    print("\n===== 总体 =====")
    print(" 全文（含 think）:"); row("overall.full", report["overall"]["full_text"])
    print(" 正文（去 think）:"); row("overall.body", report["overall"]["body_after_think"])
    print("\n===== 按类别（正文指标）=====")
    for c, r in report["by_category"].items():
        row(c, r["body_after_think"])
    print("\n===== 按 max_tokens（正文指标）=====")
    for mt, r in report["by_max_tokens"].items():
        row(f"mt={mt}", r["body_after_think"])
    print(f"\n写入 {args.out}")


if __name__ == "__main__":
    main()
