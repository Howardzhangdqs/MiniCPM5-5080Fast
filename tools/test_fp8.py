#!/usr/bin/env python3
"""test_fp8.py -- known-answer + property tests for the numpy FP8 E4M3 codec.

Run:  /workspace/.venv/bin/python tools/test_fp8.py
  or  /workspace/.venv/bin/python -c "import sys; sys.path.insert(0, 'tools'); import test_fp8; test_fp8.main()"

Covers (P4 spec, field-exact with the native lane):
  * known-answer encode vectors (normals, subnormals, ties->even, saturation)
  * known-answer decode vectors
  * max relative error of encode->decode vs the E4M3 half-ULP bound
  * the NaN encodings 0x7F / 0xFF are never emitted (incl. inf / NaN inputs)
  * per-row scale + quantize/dequantize round-trip on a known row
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from minicpm_common import (  # noqa: E402
    FP8_MAX,
    FP8_SCALE_FLOOR,
    e4m3_to_f32,
    f32_to_e4m3,
    fp8_row_scales,
    quantize_fp8_rows,
    dequantize_fp8_rows,
)

# (fp32 value, expected E4M3 byte)
ENCODE_VECTORS = [
    (0.0, 0x00),                    # zero
    (1.0, 0x38),                    # spec-mandated known answer
    (-1.0, 0xB8),                   # sign bit only
    (0.5, 0x30),                    # 2^-1
    (2.0, 0x40),                    # 2^1
    (1.5, 0x3C),                    # exact 4-bit significand
    (6.0, 0x4C),                    # 1.5 * 2^2 exact
    (240.0, 0x77),                  # 1.875 * 2^7 exact, exp field 14
    (448.0, 0x7E),                  # spec-mandated known answer (max finite)
    (449.0, 0x7E),                  # saturate
    (1000.0, 0x7E),                 # saturate
    (-1000.0, 0xFE),                # saturate negative
    (float("inf"), 0x7E),           # no inf encoding: saturate magnitude
    (float("nan"), 0x7E),           # magnitude saturates, never emit 0x7F
    (2.0 ** -6, 0x08),              # min normal
    (0.001, 0x01),                  # subnormal boundary-ish: 0.512 ulp -> 1
    (2.0 ** -10, 0x00),             # exact tie 0 vs 1 -> even (0)
    (3.0 * 2.0 ** -10, 0x02),       # exact tie 1 vs 2 -> even (2)
    (1.0625, 0x38),                 # tie 0x38/0x39 -> even mantissa (0)
    (1.1875, 0x3A),                 # tie 0x39/0x3A -> even mantissa (2)
]

DECODE_VECTORS = [
    (0x00, 0.0),
    (0x80, -0.0),
    (0x38, 1.0),
    (0xB8, -1.0),
    (0x08, 2.0 ** -6),              # min normal
    (0x01, 2.0 ** -9),              # min subnormal
    (0x07, 7.0 * 2.0 ** -9),        # max subnormal
    (0x77, 240.0),
    (0x7E, 448.0),
    (0xFE, -448.0),
]

FAILURES: list[str] = []


def check(cond: bool, msg: str) -> None:
    tag = "ok  " if cond else "FAIL"
    print(f"[{tag}] {msg}")
    if not cond:
        FAILURES.append(msg)


def main() -> int:
    print("== fp8 e4m3 known-answer encode vectors ==")
    vals = np.asarray([v for v, _ in ENCODE_VECTORS], dtype=np.float32)
    got = f32_to_e4m3(vals)
    for (v, want), g in zip(ENCODE_VECTORS, got):
        check(int(g) == want, f"encode {v:+.10g} -> 0x{int(g):02X} (want 0x{want:02X})")

    print("== fp8 e4m3 known-answer decode vectors ==")
    codes = np.asarray([c for c, _ in DECODE_VECTORS], dtype=np.uint8)
    dec = e4m3_to_f32(codes)
    for (c, want), d in zip(DECODE_VECTORS, dec):
        check(np.float32(d) == np.float32(want), f"decode 0x{c:02X} -> {np.float32(d):+.10g} (want {np.float32(want):+.10g})")

    print("== property: encode->decode error within half-ULP bound ==")
    # log-uniform magnitudes across the representable range (below the
    # saturation point), plus adversarial values sitting exactly on tie points
    rng = np.random.default_rng(20260909)
    x = (2.0 ** rng.uniform(-12, 8.858, 1 << 20)).astype(np.float32) * \
        rng.choice(np.float32([1, -1]), 1 << 20)
    ties = (np.arange(1, 64, dtype=np.float64) / 16.0).astype(np.float32)  # ties on the 1/16 grid
    x = np.concatenate([x, ties, -ties])
    rt = e4m3_to_f32(f32_to_e4m3(x))
    err = np.abs(rt.astype(np.float64) - x.astype(np.float64))
    bound = np.maximum(np.abs(x.astype(np.float64)) * (2.0 ** -4),  # <= half ULP for normals
                       2.0 ** -10)                                   # <= half subnormal step
    check(np.all(err <= bound + 1e-12),
          f"max |dequant-quant error| within half-ULP bound (observed worst excess "
          f"{float(np.max(err - bound)):.3g})")

    print("== property: NaN encodings never emitted ==")
    extreme = np.asarray([1e30, -1e30, np.inf, -np.inf, np.nan, -np.nan, 464.0, 465.0],
                         dtype=np.float32)
    enc = f32_to_e4m3(extreme)
    check(np.all((enc & 0x7F) <= 0x7E),
          f"no 0x7F/0xFF codes for extreme inputs (got {[hex(int(e)) for e in enc]})")

    print("== property: negative-zero / sign symmetry ==")
    z = f32_to_e4m3(np.asarray([np.float32(-0.0)]))
    check(int(z[0]) == 0x80, f"-0.0 -> 0x{int(z[0]):02X} (want 0x80)")
    pos = np.abs(x)
    sym = np.array_equal(f32_to_e4m3(pos), f32_to_e4m3(-pos) & np.uint8(0x7F))
    check(sym, "encode(-x) == sign | encode(|x|) for all sampled x")

    print("== per-row quantize round-trip on a known row ==")
    row = np.asarray([[1.0, -2.0, 0.5, 448.0, 0.0]], dtype=np.float32)
    s = fp8_row_scales(row)
    check(np.float32(s[0]) == np.float32(1.0), f"scale of amax-448 row == 1.0 (got {s[0]})")
    q = quantize_fp8_rows(row, s)
    want_q = [0x38, 0xC0, 0x30, 0x7E, 0x00]   # -2.0 -> 0x80|0x40
    check([int(v) for v in q[0]] == want_q,
          f"row codes {[hex(int(v)) for v in q[0]]} == {[hex(v) for v in want_q]}")
    rt2 = dequantize_fp8_rows(q, s)
    check(np.array_equal(rt2, row), f"row round-trip exact (got {rt2[0].tolist()})")

    zero_row = np.zeros((1, 8), dtype=np.float32)
    sz = fp8_row_scales(zero_row)
    check(np.float32(sz[0]) == np.float32(FP8_SCALE_FLOOR),
          f"all-zero row scale floored at {FP8_SCALE_FLOOR} (got {sz[0]})")
    qz = quantize_fp8_rows(zero_row, sz)
    check(np.all(qz == 0), "all-zero row quantizes to all 0x00")

    print("== property: random [N,K] rows, scale contract ==")
    w = (rng.standard_normal(64, dtype=np.float32) * 0.02).reshape(8, 8)
    w[3] *= 500.0  # one outlier row to exercise differing scales
    sc = fp8_row_scales(w)
    amax = np.max(np.abs(w), axis=-1)
    check(np.all(np.abs(sc - amax / np.float32(FP8_MAX)) < np.float32(1e-15)),
          "scale == max|row|/448 on random rows")
    dqw = dequantize_fp8_rows(quantize_fp8_rows(w, sc), sc)
    rel = np.linalg.norm(dqw - w, axis=-1) / np.linalg.norm(w, axis=-1)
    print(f"       per-row relative L2 error: min {rel.min():.4%} mean {rel.mean():.4%} max {rel.max():.4%}")
    check(np.all(rel < 0.06), f"random-row round-trip rel L2 < 6% (max {rel.max():.4%})")

    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s)")
        return 1
    n = len(ENCODE_VECTORS) + len(DECODE_VECTORS)
    print(f"PASS: all fp8 e4m3 codec checks ({n} known-answer vectors + properties)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
