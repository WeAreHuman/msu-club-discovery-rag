"""
Compare two RAGAS result files side by side.

Usage:
    python eval/compare.py results/2026-05-14_baseline.json results/2026-05-14_after_change.json
"""

import sys
import json
from pathlib import Path


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def main():
    if len(sys.argv) < 3:
        print("Usage: python eval/compare.py <result_a.json> <result_b.json>")
        sys.exit(1)

    a = load(sys.argv[1])
    b = load(sys.argv[2])

    label_a = a.get("label") or Path(sys.argv[1]).stem
    label_b = b.get("label") or Path(sys.argv[2]).stem

    metrics = list(a["summary"].keys())

    col_w = 22
    width = 72
    print("\n" + "=" * width)
    print(f"  {'METRIC':<26} {label_a[:col_w]:<{col_w+2}} {label_b[:col_w]:<{col_w+2}} DELTA")
    print("=" * width)

    for m in metrics:
        sa = a["summary"].get(m, 0.0)
        sb = b["summary"].get(m, 0.0)
        delta = sb - sa
        sign = "+" if delta >= 0 else ""
        flag = "  ▲" if delta > 0.02 else ("  ▼" if delta < -0.02 else "")
        print(f"  {m:<26} {sa:<{col_w+2}.3f} {sb:<{col_w+2}.3f} {sign}{delta:.3f}{flag}")

    print("=" * width)
    print(f"\n  A: {sys.argv[1]}")
    print(f"  B: {sys.argv[2]}")
    print(f"  ▲ = improved by >0.02   ▼ = degraded by >0.02\n")


if __name__ == "__main__":
    main()
