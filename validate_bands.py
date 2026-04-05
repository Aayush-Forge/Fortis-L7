"""
Monte Carlo check: sampled fingerprints vs architect RS bands.

Note: published "easy" RS band (0.82–0.96) can exceed what the same document's
linear RS formula achieves for the given easy feature ranges (empirical max ~0.73).
Use this script to see realized distributions; tune generator ranges or weights only
with architect approval.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from data_generator import RequestGenerator
from risk_engine import ObservationVector, RiskEngine


@dataclass
class Band:
    low: float
    high: float
    name: str


EXPECTED: Dict[str, Band] = {
    "easy": Band(0.82, 0.96, "static scraper"),
    "medium": Band(0.50, 0.72, "rotational proxy"),
    "hard": Band(0.42, 0.65, "LLM mimicry"),
}


def collect_rs(tier: str, n: int, seed: int) -> np.ndarray:
    gen = RequestGenerator(level=2, human_prior=0.0, seed=seed)
    eng = RiskEngine()
    out: List[float] = []
    for _ in range(n):
        if tier == "easy":
            feats = gen._sample_easy_bot()  # noqa: SLF001
        elif tier == "medium":
            feats = gen._sample_medium_bot()  # noqa: SLF001
        elif tier == "hard":
            feats = gen._sample_hard_bot()  # noqa: SLF001
        else:
            feats = gen._sample_human()  # noqa: SLF001
        obs = ObservationVector(**feats)
        rs, _ = eng.compute(obs)
        out.append(rs)
    return np.array(out, dtype=np.float64)


def summarize(name: str, samples: np.ndarray, band: Band | None) -> None:
    mean = float(samples.mean())
    p5, p95 = float(np.percentile(samples, 5)), float(np.percentile(samples, 95))
    print(f"\n{name} (n={len(samples)})")
    print(f"  mean={mean:.3f}  p5={p5:.3f}  p95={p95:.3f}  min={samples.min():.3f}  max={samples.max():.3f}")
    if band:
        in_band = float(((samples >= band.low) & (samples <= band.high)).mean())
        print(f"  target band [{band.low:.2f}, {band.high:.2f}] ({band.name}): fraction in band={in_band:.1%}")


def main() -> None:
    n = 5000
    print("Risk score validation (architect formula + generator ranges)")
    for tier, band in EXPECTED.items():
        samples = collect_rs(tier, n, seed=42 + hash(tier) % 10000)
        summarize(tier, samples, band)

    human = collect_rs("human", n, seed=99)
    summarize("human", human, None)
    print("\nNote: human traffic should skew toward low RS (many verified/probably_human labels).")
    frac_low = float((human < 0.35).mean())
    print(f"  fraction RS < 0.35: {frac_low:.1%}")


if __name__ == "__main__":
    main()
