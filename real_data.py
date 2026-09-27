"""Real-data windowed source-retrieval evaluation used by the paper."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from SSIMuse import SSIMuse
from midi_utils import midi_to_piano_roll


@dataclass(frozen=True)
class RetrievalMatch:
    score: float
    edges: tuple[tuple[int, int, float], ...]
    allocated_query_steps: tuple[float, ...]


@dataclass(frozen=True)
class BatchRetrievalResult:
    raw_scores: np.ndarray
    calibrated_scores: np.ndarray
    source_means: np.ndarray
    ranking: np.ndarray


def split_windows(
    roll: np.ndarray,
    *,
    window_bars: int = 8,
    hop_bars: int = 6,
    steps_per_bar: int = 16,
) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    """Split a roll into overlapping windows and append a tail-aligned window."""
    roll = np.asarray(roll, dtype=np.float64)
    if roll.ndim != 2 or len(roll) == 0:
        raise ValueError("roll must be a non-empty 2-D array")
    window_steps = int(window_bars) * int(steps_per_bar)
    hop_steps = int(hop_bars) * int(steps_per_bar)
    if window_steps <= 0 or hop_steps <= 0:
        raise ValueError("window and hop sizes must be positive")
    if len(roll) <= window_steps:
        padded = np.zeros((window_steps, roll.shape[1]), dtype=np.float64)
        padded[:len(roll)] = roll
        return [padded], [(0, len(roll))]
    starts = list(range(0, len(roll) - window_steps + 1, hop_steps))
    tail = len(roll) - window_steps
    if starts[-1] != tail:
        starts.append(tail)
    return (
        [roll[start:start + window_steps].copy() for start in starts],
        [(start, start + window_steps) for start in starts],
    )


def _allocate_unique_duration(
    intervals: Sequence[tuple[int, int]], selected: Sequence[int]
) -> np.ndarray:
    durations = np.zeros(len(selected), dtype=np.float64)
    boundaries = sorted({point for index in selected for point in intervals[index]})
    for left, right in zip(boundaries, boundaries[1:]):
        covering = [
            slot for slot, index in enumerate(selected)
            if intervals[index][0] <= left and right <= intervals[index][1]
        ]
        if covering:
            durations[covering] += (right - left) / len(covering)
    return durations


def compare_window_sets(
    query_windows: Sequence[np.ndarray],
    source_windows: Sequence[np.ndarray],
    metric: SSIMuse | None = None,
) -> np.ndarray:
    """Compute all query/source window scores.

    The source window is the SSIMuse reference and the query window is the
    transformed candidate, matching the paper's retrieval implementation.
    """
    if not query_windows or not source_windows:
        raise ValueError("query_windows and source_windows cannot be empty")
    metric = metric or SSIMuse(weight_power=0.0)
    matrix = np.empty((len(query_windows), len(source_windows)), dtype=np.float64)
    for query_index, query in enumerate(query_windows):
        for source_index, source in enumerate(source_windows):
            matrix[query_index, source_index] = metric.score(source, query).total
    return matrix


def score_window_matrix(
    scores: np.ndarray,
    query_intervals: Sequence[tuple[int, int]],
    *,
    full_query_steps: int | None = None,
    top_k: int = 3,
) -> RetrievalMatch:
    """Apply full one-to-one matching, Top-k, and unique query coverage."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or min(scores.shape) == 0:
        raise ValueError("scores must be a non-empty 2-D matrix")
    if scores.shape[0] != len(query_intervals):
        raise ValueError("query_intervals must match the matrix rows")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    query_indices, source_indices = linear_sum_assignment(scores, maximize=True)
    edges = sorted(
        (
            (int(query), int(source), float(scores[query, source]))
            for query, source in zip(query_indices, source_indices)
        ),
        key=lambda edge: (-edge[2], edge[0], edge[1]),
    )[:top_k]
    selected = [edge[0] for edge in edges]
    allocated = _allocate_unique_duration(query_intervals, selected)
    if full_query_steps is None:
        full_query_steps = max(right for _, right in query_intervals)
    if full_query_steps <= 0:
        raise ValueError("full_query_steps must be positive")
    score = float(np.dot([edge[2] for edge in edges], allocated) / full_query_steps)
    return RetrievalMatch(score, tuple(edges), tuple(float(x) for x in allocated))


def batch_retrieve(raw_scores: np.ndarray, *, epsilon: float = 1e-12) -> BatchRetrievalResult:
    """Calibrate each source by its evaluation-query mean and rank sources.

    This is transductive batch calibration, not independent pairwise scoring.
    Rows are queries and columns are candidate sources.
    """
    raw_scores = np.asarray(raw_scores, dtype=np.float64)
    if raw_scores.ndim != 2 or min(raw_scores.shape) == 0:
        raise ValueError("raw_scores must be a non-empty 2-D matrix")
    means = raw_scores.mean(axis=0)
    calibrated = raw_scores / np.maximum(means, float(epsilon))
    ranking = np.argsort(-calibrated, axis=1, kind="stable")
    return BatchRetrievalResult(raw_scores.copy(), calibrated, means, ranking)


def _load_roll(path: Path, track_index: int | None = None) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=False)
    if path.suffix.lower() in {".mid", ".midi"}:
        return midi_to_piano_roll(path, track_index=track_index)
    raise ValueError(f"unsupported piano-roll input: {path}")


def _discover_rolls(directory: str | Path) -> list[tuple[str, np.ndarray]]:
    """Load supported files and use each filename stem as its case ID."""
    directory = Path(directory)
    paths = sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".npy", ".mid", ".midi"}
    )
    if not paths:
        raise ValueError(f"no NPY or MIDI inputs found in {directory}")
    identifiers = [path.stem for path in paths]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"duplicate case IDs in {directory}")
    return [(path.stem, _load_roll(path)) for path in paths]


def evaluate_real_data(
    query_dir: str | Path,
    source_dir: str | Path,
    output_path: str | Path,
) -> dict:
    """Evaluate all queries against all sources with the paper configuration.

    File stems are case IDs. A query's correct source is the source file with
    the same stem, e.g. ``queries/case01.npy`` -> ``sources/case01.npy``.
    """
    queries = [(case_id, case_id, roll) for case_id, roll in _discover_rolls(query_dir)]
    sources = _discover_rolls(source_dir)
    source_ids = {case_id for case_id, _ in sources}
    missing = [case_id for case_id, _, _ in queries if case_id not in source_ids]
    if missing:
        raise ValueError(f"queries without same-named sources: {missing}")
    metric = SSIMuse(weight_power=0.0)
    raw = np.empty((len(queries), len(sources)), dtype=np.float64)
    pair_details = []
    for query_index, (query_id, _, query_roll) in enumerate(queries):
        query_windows, query_intervals = split_windows(query_roll)
        for source_index, (source_id, source_roll) in enumerate(sources):
            source_windows, _ = split_windows(source_roll)
            matrix = compare_window_sets(query_windows, source_windows, metric)
            match = score_window_matrix(
                matrix,
                query_intervals,
                full_query_steps=len(query_roll),
                top_k=3,
            )
            raw[query_index, source_index] = match.score
            pair_details.append({
                "query_id": query_id,
                "source_id": source_id,
                "raw_score": match.score,
                "matched_edges": [list(edge) for edge in match.edges],
                "allocated_query_steps": list(match.allocated_query_steps),
            })
        print(f"queries: {query_index + 1}/{len(queries)}", flush=True)

    result = batch_retrieve(raw)
    source_ids = [source_id for source_id, _ in sources]
    rankings = []
    correct_count = 0
    for index, (query_id, correct_source_id, _) in enumerate(queries):
        ranked_ids = [source_ids[column] for column in result.ranking[index]]
        correct = ranked_ids[0] == correct_source_id
        correct_count += int(correct)
        rankings.append({
            "query_id": query_id,
            "correct_source_id": correct_source_id,
            "top1_source_id": ranked_ids[0],
            "correct": correct,
            "ranked_source_ids": ranked_ids,
            "calibrated_scores": [
                float(result.calibrated_scores[index, column])
                for column in result.ranking[index]
            ],
        })
    output = {
        "configuration": {
            "window_bars": 8,
            "hop_bars": 6,
            "local_weight_power": 0.0,
            "matching": "full Hungarian then top 3",
            "score": "unique query coverage / full query duration",
            "calibration": "per-source mean over evaluation queries",
        },
        "query_count": len(queries),
        "source_count": len(sources),
        "top1_correct": correct_count,
        "top1_accuracy": correct_count / len(queries),
        "source_ids": source_ids,
        "source_means": result.source_means.tolist(),
        "raw_scores": result.raw_scores.tolist(),
        "calibrated_scores": result.calibrated_scores.tolist(),
        "rankings": rankings,
        "pair_details": pair_details,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SSIMuse on real-data retrieval")
    parser.add_argument("--queries-dir", type=Path, required=True)
    parser.add_argument("--sources-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluate_real_data(args.queries_dir, args.sources_dir, args.output)


if __name__ == "__main__":
    main()
