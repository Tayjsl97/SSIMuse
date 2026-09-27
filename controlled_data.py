"""Controlled-copy evaluation on 16-bar piano-roll clips.

Reference and mixture directories contain 16-bar ``.npy`` piano rolls at
16 time steps per bar. The script evaluates randomly paired baselines and
constructs exact rational-duration copies of the requested bar lengths.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
from itertools import product
import json
from pathlib import Path
import random

import numpy as np

from SSIMuse import DEFAULT_SCALE_FACTORS, ExactPianoRoll, SSIMuse


STEPS_PER_BAR = 16
CLIP_BARS = 16
MIN_SCALED_COPY_STEPS = 4


def _npy_files(directory: str | Path) -> list[Path]:
    files = sorted(Path(directory).glob("*.npy"))
    if not files:
        raise ValueError(f"no .npy piano rolls in {directory}")
    return files


def valid_construction_scales(copy_bars: int) -> tuple[Fraction, ...]:
    """Return paper scale factors that fit a 16-bar target clip."""
    if copy_bars <= 0 or copy_bars > CLIP_BARS:
        raise ValueError(f"copy_bars must be in [1, {CLIP_BARS}]")
    copy_steps = STEPS_PER_BAR * copy_bars
    minimum = max(Fraction(1, 16), Fraction(MIN_SCALED_COPY_STEPS, copy_steps))
    maximum = Fraction(CLIP_BARS, copy_bars)
    return tuple(scale for scale in DEFAULT_SCALE_FACTORS if minimum <= scale <= maximum)


def _score_record(metric: SSIMuse, reference, candidate) -> dict[str, float]:
    score = metric.score(reference, candidate)
    no_scale_structure = score.scale_scores[Fraction(1, 1)]
    return {
        "density": score.density,
        "structure": score.structure,
        "ssimuse": score.total,
        "structure_without_scale_search": no_scale_structure,
        "ssimuse_without_scale_search": score.density * no_scale_structure,
    }


def evaluate_controlled_data(
    reference_dir: str | Path,
    mixture_dir: str | Path,
    output_path: str | Path,
    *,
    copy_bars: tuple[int, ...] = (1, 2, 4, 8),
    baseline_pairs: int = 4000,
    samples_per_reference: int = 1,
    seed: int = 42,
    max_references: int | None = None,
) -> dict:
    references = _npy_files(reference_dir)
    mixtures = _npy_files(mixture_dir)
    if max_references is not None:
        references = references[:max_references]
    rng = random.Random(seed)
    metric = SSIMuse(weight_power=1.0)

    all_pairs = list(product(references, mixtures))
    if baseline_pairs > len(all_pairs):
        raise ValueError(f"baseline_pairs={baseline_pairs} exceeds {len(all_pairs)}")
    baseline = []
    for index, (reference_path, mixture_path) in enumerate(rng.sample(all_pairs, baseline_pairs)):
        reference = np.load(reference_path, allow_pickle=False)
        mixture = np.load(mixture_path, allow_pickle=False)
        baseline.append({
            "reference_file": reference_path.name,
            "mixture_file": mixture_path.name,
            **_score_record(metric, reference, mixture),
        })
        if (index + 1) % 100 == 0:
            print(f"baseline: {index + 1}/{baseline_pairs}", flush=True)

    controlled: dict[str, list[dict]] = {}
    for bars in copy_bars:
        copy_steps = STEPS_PER_BAR * bars
        scales = valid_construction_scales(bars)
        records = []
        sample_count = len(references) * len(scales) * samples_per_reference
        completed = 0
        for reference_path in reversed(references):
            reference = np.load(reference_path, allow_pickle=False)
            if len(reference) < copy_steps:
                raise ValueError(f"{reference_path} is shorter than {bars} bars")
            for construction_scale in scales:
                for _ in range(samples_per_reference):
                    start_reference = rng.randint(0, len(reference) - copy_steps)
                    copied = reference[start_reference:start_reference + copy_steps]
                    mixture_path = rng.choice(mixtures)
                    mixture = np.load(mixture_path, allow_pickle=False)
                    scaled_rows = int(np.ceil(copy_steps * float(construction_scale)))
                    if scaled_rows > len(mixture):
                        raise ValueError(f"scaled copy does not fit {mixture_path}")
                    possible_starts = range(
                        0, len(mixture) - scaled_rows + 1, construction_scale.numerator
                    )
                    start_mixture = rng.choice(possible_starts)
                    exact_mixture = ExactPianoRoll.with_replacement(
                        mixture, start_mixture, copied, construction_scale
                    )
                    records.append({
                        "reference_file": reference_path.name,
                        "mixture_file": mixture_path.name,
                        "copy_bars": bars,
                        "construction_scale": str(construction_scale),
                        "start_reference": start_reference,
                        "start_mixture": start_mixture,
                        **_score_record(metric, reference, exact_mixture),
                    })
                    completed += 1
                    if completed % 100 == 0 or completed == sample_count:
                        print(f"copy-{bars}-bars: {completed}/{sample_count}", flush=True)
        controlled[str(bars)] = records

    output = {
        "configuration": {
            "clip_bars": CLIP_BARS,
            "steps_per_bar": STEPS_PER_BAR,
            "copy_bars": list(copy_bars),
            "scale_factors": [str(scale) for scale in DEFAULT_SCALE_FACTORS],
            "samples_per_reference": samples_per_reference,
            "random_seed": seed,
        },
        "baseline": baseline,
        "controlled_copy": controlled,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="SSIMuse controlled 16-bar evaluation")
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--mixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bars", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--baseline-pairs", type=int, default=4000)
    parser.add_argument("--samples-per-reference", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-references", type=int)
    args = parser.parse_args()
    evaluate_controlled_data(
        args.references,
        args.mixtures,
        args.output,
        copy_bars=tuple(args.bars),
        baseline_pairs=args.baseline_pairs,
        samples_per_reference=args.samples_per_reference,
        seed=args.seed,
        max_references=args.max_references,
    )


if __name__ == "__main__":
    main()
