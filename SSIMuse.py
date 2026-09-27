"""Core SSIMuse metric for binary piano rolls.

The implementation follows the current paper design: octave folding, exact
uniform duration scaling, temporal and pitch-shift search, soft Jaccard local
similarity, MSSIM-style aggregation, and FFT-accelerated circular alignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Iterable

import numpy as np
from scipy.fft import irfftn, rfftn


DEFAULT_SCALE_FACTORS = (
    Fraction(1, 16), Fraction(1, 8), Fraction(1, 4), Fraction(1, 3),
    Fraction(3, 8), Fraction(1, 2), Fraction(2, 3), Fraction(3, 4),
    Fraction(1, 1), Fraction(4, 3), Fraction(3, 2), Fraction(2, 1),
    Fraction(8, 3), Fraction(3, 1), Fraction(4, 1), Fraction(8, 1),
    Fraction(16, 1),
)


def _as_fraction(value: int | float | str | Fraction) -> Fraction:
    if isinstance(value, Fraction):
        result = value
    elif isinstance(value, (int, np.integer)):
        result = Fraction(int(value), 1)
    elif isinstance(value, str):
        result = Fraction(value)
    else:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("scale factors must be finite")
        result = Fraction(str(number)).limit_denominator(1_000_000)
    if result <= 0:
        raise ValueError("scale factors must be positive")
    return result


def _validate_roll(roll: np.ndarray, *, occupancy: bool = False) -> np.ndarray:
    roll = np.asarray(roll, dtype=np.float64)
    if roll.ndim != 2 or min(roll.shape) == 0:
        raise ValueError("a piano roll must be a non-empty 2-D array")
    if not np.all(np.isfinite(roll)) or np.any(roll < -1e-12):
        raise ValueError("piano-roll values must be finite and non-negative")
    if occupancy and np.any(roll > 1.0 + 1e-9):
        raise ValueError("occupancy values must lie in [0, 1]")
    return np.clip(roll, 0.0, 1.0) if occupancy else roll


def _piecewise_integral(roll: np.ndarray, positions: np.ndarray) -> np.ndarray:
    roll = _validate_roll(roll, occupancy=True)
    positions = np.clip(np.asarray(positions, dtype=np.float64), 0, len(roll))
    cumulative = np.vstack((
        np.zeros((1, roll.shape[1]), dtype=np.float64),
        np.cumsum(roll, axis=0, dtype=np.float64),
    ))
    integer = np.floor(positions).astype(np.int64)
    base = cumulative[np.minimum(integer, len(roll))]
    samples = roll[np.minimum(integer, len(roll) - 1)]
    return base + (positions - integer)[:, None] * samples


@dataclass(frozen=True)
class ExactPianoRoll:
    """A binary roll with exactly composable scaling and one replacement.

    The replacement representation is used by the controlled 16-bar data
    experiment so construction and detector scale factors cancel before the
    piano roll is rasterized.
    """

    base: np.ndarray
    timeline_scale: Fraction = Fraction(1, 1)
    replacement_start: Fraction | None = None
    replacement_source: np.ndarray | None = None
    replacement_scale: Fraction = Fraction(1, 1)

    def __post_init__(self) -> None:
        base = (_validate_roll(self.base) > 0).astype(float)
        object.__setattr__(self, "base", base)
        object.__setattr__(self, "timeline_scale", _as_fraction(self.timeline_scale))
        object.__setattr__(self, "replacement_scale", _as_fraction(self.replacement_scale))
        if self.replacement_start is None:
            if self.replacement_source is not None:
                raise ValueError("replacement_source requires replacement_start")
            return
        start = Fraction(self.replacement_start)
        source = (_validate_roll(self.replacement_source) > 0).astype(float)
        end = start + len(source) * self.replacement_scale
        if source.shape[1] != base.shape[1] or start < 0 or end > len(base):
            raise ValueError("replacement must fit the base roll and pitch dimension")
        object.__setattr__(self, "replacement_start", start)
        object.__setattr__(self, "replacement_source", source)

    @classmethod
    def from_array(cls, roll: np.ndarray) -> "ExactPianoRoll":
        return cls(roll)

    @classmethod
    def with_replacement(
        cls,
        base: np.ndarray,
        start: int,
        source: np.ndarray,
        scale: int | float | str | Fraction,
    ) -> "ExactPianoRoll":
        return cls(
            base=base,
            replacement_start=Fraction(start),
            replacement_source=source,
            replacement_scale=_as_fraction(scale),
        )

    def scaled(self, factor: int | float | str | Fraction) -> "ExactPianoRoll":
        return ExactPianoRoll(
            self.base,
            self.timeline_scale * _as_fraction(factor),
            self.replacement_start,
            self.replacement_source,
            self.replacement_scale,
        )

    @property
    def exact_duration(self) -> Fraction:
        return len(self.base) * self.timeline_scale

    def rasterize(self) -> np.ndarray:
        """Area-rasterize once, preserving fractional occupancy at boundaries."""
        duration = self.exact_duration
        target_length = math.ceil(duration)
        starts = np.arange(target_length, dtype=np.float64)
        ends = np.minimum(starts + 1.0, float(duration))
        scale = float(self.timeline_scale)
        result = scale * (
            _piecewise_integral(self.base, ends / scale)
            - _piecewise_integral(self.base, starts / scale)
        )
        if self.replacement_start is not None:
            hole_start = float(self.replacement_start)
            hole_end = float(
                self.replacement_start + len(self.replacement_source) * self.replacement_scale
            )
            base_starts = starts / scale
            base_ends = ends / scale
            clipped_starts = np.clip(base_starts, hole_start, hole_end)
            clipped_ends = np.clip(base_ends, hole_start, hole_end)
            active = clipped_ends > clipped_starts
            removed = scale * (
                _piecewise_integral(self.base, clipped_ends)
                - _piecewise_integral(self.base, clipped_starts)
            )
            removed[~active] = 0.0
            result -= removed
            construction_scale = float(self.replacement_scale)
            local_starts = (clipped_starts - hole_start) / construction_scale
            local_ends = (clipped_ends - hole_start) / construction_scale
            inserted = scale * construction_scale * (
                _piecewise_integral(self.replacement_source, local_ends)
                - _piecewise_integral(self.replacement_source, local_starts)
            )
            inserted[~active] = 0.0
            result += inserted
        return np.clip(result, 0.0, 1.0)


@dataclass(frozen=True)
class SSIMuseScore:
    """Interpretable SSIMuse score components."""

    density: float
    structure: float
    total: float
    scale_scores: dict[Fraction, float]


class SSIMuse:
    """Compare binary piano rolls shaped ``(time_steps, MIDI_pitches)``.

    ``weight_power=1`` is the paper's controlled-replication setting.
    ``weight_power=0`` is its classical-MSSIM setting for real-case retrieval.
    The first argument is the reference; the second is the candidate that is
    duration-scaled and shifted during alignment.
    """

    def __init__(
        self,
        *,
        scale_factors: Iterable[int | float | str | Fraction] = DEFAULT_SCALE_FACTORS,
        local_window_steps: int = 16,
        local_hop_steps: int = 16,
        weight_power: float = 1.0,
        time_penalty: float = 0.5,
        scale_penalty: float = 0.125,
        max_time_shift: int | None = None,
    ) -> None:
        self.scale_factors = tuple(_as_fraction(value) for value in scale_factors)
        self.local_window_steps = int(local_window_steps)
        self.local_hop_steps = int(local_hop_steps)
        self.weight_power = float(weight_power)
        self.time_penalty = float(time_penalty)
        self.scale_penalty = float(scale_penalty)
        self.max_time_shift = max_time_shift
        if not self.scale_factors:
            raise ValueError("scale_factors cannot be empty")
        if self.local_window_steps <= 0 or self.local_hop_steps <= 0:
            raise ValueError("window and hop sizes must be positive")
        if min(self.weight_power, self.time_penalty, self.scale_penalty) < 0:
            raise ValueError("weights and penalties must be non-negative")
        if max_time_shift is not None and max_time_shift < 0:
            raise ValueError("max_time_shift must be non-negative or None")

    @staticmethod
    def _fold_pitch_classes(roll: np.ndarray) -> np.ndarray:
        folded = np.zeros((len(roll), 12), dtype=np.float64)
        for pitch_class in range(12):
            folded[:, pitch_class] = roll[:, pitch_class::12].sum(axis=1)
        return folded

    def _window_specs(self, length: int) -> list[tuple[int, int]]:
        specs = [
            (start, self.local_window_steps)
            for start in range(0, length - self.local_window_steps + 1, self.local_hop_steps)
        ]
        return specs or [(0, length)]

    def _time_shifts(self, period: int) -> np.ndarray:
        radius = period // 2
        if self.max_time_shift is not None:
            radius = min(radius, self.max_time_shift)
        return np.arange(-radius, radius + 1, dtype=np.int64)

    def density(
        self, reference: np.ndarray, candidate: np.ndarray | ExactPianoRoll
    ) -> float:
        x = (_validate_roll(reference) > 0).astype(float)
        y = (
            candidate.rasterize()
            if isinstance(candidate, ExactPianoRoll)
            else (_validate_roll(candidate) > 0).astype(float)
        )
        if x.shape[1] != y.shape[1]:
            raise ValueError("piano rolls must have the same pitch dimension")
        length = min(len(x), len(y))
        values = []
        for start, size in self._window_specs(length):
            mu_x = float(x[start:start + size].mean())
            mu_y = float(y[start:start + size].mean())
            c1 = 0.01**2
            values.append((2 * mu_x * mu_y + c1) / (mu_x**2 + mu_y**2 + c1))
        return float(np.mean(values))

    @staticmethod
    def _layer_correlations(
        reference: np.ndarray,
        candidate: np.ndarray,
        period: int,
        windows: list[tuple[int, int]],
    ) -> np.ndarray:
        if not np.allclose(reference, np.rint(reference), rtol=0, atol=1e-10):
            raise ValueError("the reference roll must use hard binary occupancy")
        correlations = np.zeros((len(windows), period, 12), dtype=np.float64)
        for level in range(1, int(np.max(reference, initial=0)) + 1):
            layers = np.zeros((len(windows), period, 12), dtype=np.float64)
            for index, (start, size) in enumerate(windows):
                layers[index, start:start + size] = reference[start:start + size] >= level
            candidate_layer = np.clip(candidate - level + 1.0, 0.0, 1.0)
            x_fft = rfftn(layers, axes=(-2, -1), workers=1)
            y_fft = rfftn(candidate_layer, axes=(-2, -1), workers=1)
            correlations += irfftn(
                np.conj(x_fft) * y_fft[None, :, :],
                s=(period, 12), axes=(-2, -1), workers=1,
            ).real
        return correlations

    def _score_scale(
        self, reference: np.ndarray, candidate: np.ndarray, scale: Fraction
    ) -> float:
        target_length = len(reference)
        period = max(target_length, len(candidate))
        padded = np.zeros((period, 12), dtype=np.float64)
        padded[:len(candidate)] = candidate
        windows = self._window_specs(target_length)
        correlations = self._layer_correlations(reference, padded, period, windows)
        time_shifts = self._time_shifts(period)
        pitch_shifts = np.arange(-5, 7, dtype=np.int64)
        intersections = correlations[
            :, np.mod(-time_shifts, period)[:, None], np.mod(-pitch_shifts, 12)[None, :]
        ]
        x_sums = np.asarray([
            reference[start:start + size].sum(dtype=np.float64)
            for start, size in windows
        ])
        candidate_counts = padded.sum(axis=1, dtype=np.float64)
        y_sums = np.empty((len(windows), len(time_shifts)), dtype=np.float64)
        for index, (start, size) in enumerate(windows):
            positions = start + np.arange(size, dtype=np.int64)
            shifted = np.mod(positions[None, :] - time_shifts[:, None], period)
            y_sums[index] = candidate_counts[shifted].sum(axis=1)
        intersections = np.clip(
            intersections, 0.0, np.minimum(x_sums[:, None, None], y_sums[:, :, None])
        )
        unions = x_sums[:, None, None] + y_sums[:, :, None] - intersections
        local = np.divide(
            intersections, unions, out=np.zeros_like(intersections), where=unions > 0
        )
        weights = np.where(unions > 0, np.power(local, self.weight_power), 0.0)
        denominator = weights.sum(axis=0)
        aggregate = np.divide(
            (local * weights).sum(axis=0), denominator,
            out=np.zeros_like(denominator), where=denominator > 0,
        )
        time_weights = np.maximum(
            0.0, 1.0 - self.time_penalty * np.abs(time_shifts) / period
        )
        scale_weight = max(
            0.0, 1.0 - self.scale_penalty * abs(math.log2(float(scale)))
        )
        return float(np.max(aggregate * time_weights[:, None] * scale_weight, initial=0.0))

    def structure(
        self, reference: np.ndarray, candidate: np.ndarray | ExactPianoRoll
    ) -> tuple[float, dict[Fraction, float]]:
        x_array = (_validate_roll(reference) > 0).astype(float)
        candidate_base = candidate.base if isinstance(candidate, ExactPianoRoll) else candidate
        if x_array.shape[1] != np.asarray(candidate_base).shape[1]:
            raise ValueError("piano rolls must have the same pitch dimension")
        reference_folded = self._fold_pitch_classes(x_array)
        exact_candidate = (
            candidate if isinstance(candidate, ExactPianoRoll)
            else ExactPianoRoll.from_array(candidate)
        )
        scale_scores: dict[Fraction, float] = {}
        for scale in self.scale_factors:
            scaled = exact_candidate.scaled(scale).rasterize()
            score = self._score_scale(reference_folded, self._fold_pitch_classes(scaled), scale)
            scale_scores[scale] = score
        return max(scale_scores.values(), default=0.0), scale_scores

    def score(
        self, reference: np.ndarray, candidate: np.ndarray | ExactPianoRoll
    ) -> SSIMuseScore:
        density = self.density(reference, candidate)
        structure, scale_scores = self.structure(reference, candidate)
        return SSIMuseScore(density, structure, density * structure, scale_scores)

    def compute_ssim(
        self, reference: np.ndarray, candidate: np.ndarray | ExactPianoRoll
    ) -> tuple[float, float, float]:
        """Backward-compatible tuple interface: density, structure, product."""
        score = self.score(reference, candidate)
        return score.density, score.structure, score.total
