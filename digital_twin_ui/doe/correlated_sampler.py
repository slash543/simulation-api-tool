"""
Correlated Speed Sampler
========================
Generates physiologically plausible multi-step speed arrays for DOE campaigns
on the 10-step catheter insertion templates.

Algorithm
---------
For each sample:
  1. Draw a mean speed uniformly from [speed_min, speed_max] using LHS.
  2. Generate n_steps perturbation factors from Uniform(-max_perturbation,
     +max_perturbation).
  3. speed[j] = mean_speed * (1 + perturbation[j])
  4. Clip all values to [speed_min, speed_max].
  5. Sort each row ascending so the catheter ramps up speed progressively.

Usage
-----
    from digital_twin_ui.doe.correlated_sampler import CorrelatedSpeedSampler

    sampler = CorrelatedSpeedSampler()
    speeds = sampler.sample(
        n_samples=20,
        speed_min=10.0,
        speed_max=25.0,
        n_steps=10,
        seed=42,
    )
    # speeds.shape == (20, 10)
"""

from __future__ import annotations

import numpy as np
from scipy.stats.qmc import LatinHypercube

from digital_twin_ui.app.core.logging import get_logger

logger = get_logger(__name__)


class CorrelatedSpeedSampler:
    """
    Generate correlated per-step speed arrays for multi-step DOE campaigns.

    Attributes:
        name:             Sampler identifier (used in logs / result metadata).
        max_perturbation: Maximum fractional perturbation applied to each step
                          speed relative to the mean speed.  Can be changed
                          after construction.

    Args:
        max_perturbation: Initial value.  Default is 0.20 (±20%).
    """

    name: str = "correlated_lhs"

    def __init__(self, max_perturbation: float = 0.20) -> None:
        self.max_perturbation = max_perturbation

    def sample(
        self,
        n_samples: int,
        speed_min: float,
        speed_max: float,
        n_steps: int = 10,
        max_perturbation: float | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        """
        Generate a speed matrix of shape (n_samples, n_steps).

        Each row represents one DOE sample: a vector of per-step insertion
        speeds in mm/s.

        Args:
            n_samples:        Number of DOE samples to generate.
            speed_min:        Minimum allowed speed in mm/s.
            speed_max:        Maximum allowed speed in mm/s.
            n_steps:          Number of insertion steps per sample (default 10).
            max_perturbation: Override instance ``max_perturbation`` for this
                              call only.  If None, uses ``self.max_perturbation``.
            seed:             Optional RNG seed for reproducibility.

        Returns:
            ``np.ndarray`` of shape ``(n_samples, n_steps)``, dtype float64,
            with all values in ``[speed_min, speed_max]``, each row sorted
            ascending.

        Raises:
            ValueError: If n_samples < 1, n_steps < 1, or speed_min >= speed_max.
        """
        if n_samples < 1:
            raise ValueError(f"n_samples must be ≥ 1, got {n_samples}")
        if n_steps < 1:
            raise ValueError(f"n_steps must be ≥ 1, got {n_steps}")
        if speed_min >= speed_max:
            raise ValueError(
                f"speed_min ({speed_min}) must be < speed_max ({speed_max})"
            )

        mp = max_perturbation if max_perturbation is not None else self.max_perturbation

        rng = np.random.default_rng(seed)

        # Step 1: sample mean speeds via LHS in [speed_min, speed_max]
        lhs = LatinHypercube(d=1, seed=seed)
        unit_means = lhs.random(n=n_samples).ravel()  # shape (n_samples,)
        mean_speeds = speed_min + unit_means * (speed_max - speed_min)  # shape (n_samples,)

        # Step 2: perturbation factors — shape (n_samples, n_steps)
        perturbations = rng.uniform(-mp, mp, size=(n_samples, n_steps))

        # Step 3: apply perturbations
        speeds = mean_speeds[:, np.newaxis] * (1.0 + perturbations)

        # Step 4: clip to [speed_min, speed_max]
        speeds = np.clip(speeds, speed_min, speed_max)

        # Step 5: sort each row ascending (physiological ramp-up)
        speeds = np.sort(speeds, axis=1)

        logger.debug(
            "CorrelatedSpeedSampler: generated ({n_samples}, {n_steps}) "
            "speed matrix, range=[{lo:.1f}, {hi:.1f}] mm/s, "
            "max_perturbation={mp:.0%}, seed={seed}",
            n_samples=n_samples,
            n_steps=n_steps,
            lo=float(speeds.min()),
            hi=float(speeds.max()),
            mp=mp,
            seed=seed,
        )

        return speeds
