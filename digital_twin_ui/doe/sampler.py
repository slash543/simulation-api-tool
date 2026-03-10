"""
DOE Sampler
===========
Generates parameter samples for Design of Experiments (DOE) campaigns.

Three sampling strategies are provided:

  LHS      — Latin Hypercube Sampling (space-filling, good for moderate N)
  Sobol    — Sobol quasi-random sequence (low-discrepancy, good for large N)
  Uniform  — Evenly-spaced grid (deterministic, good for small N or sweeps)

All samplers accept a 1-D parameter space defined by ``(min_val, max_val)`` and
return a sorted numpy array of ``n_samples`` values within that range.

Usage
-----
    from digital_twin_ui.doe.sampler import get_sampler

    sampler = get_sampler("lhs")
    speeds = sampler.sample(n_samples=10, low=4.0, high=6.0, seed=42)
    # → array of 10 values in [4.0, 6.0]

    sampler = get_sampler("sobol")
    speeds = sampler.sample(n_samples=8, low=4.0, high=6.0, seed=0)

    sampler = get_sampler("uniform")
    speeds = sampler.sample(n_samples=5, low=4.0, high=6.0)
    # → array([4.0, 4.5, 5.0, 5.5, 6.0])
"""

from __future__ import annotations

import abc
from typing import Literal

import numpy as np
from scipy.stats.qmc import LatinHypercube, Sobol, scale as qmc_scale

from digital_twin_ui.app.core.logging import get_logger

logger = get_logger(__name__)

SamplerName = Literal["lhs", "sobol", "uniform"]


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseSampler(abc.ABC):
    """
    Abstract sampler interface.

    Sub-classes implement :meth:`sample` to generate a 1-D parameter sweep.
    """

    #: Human-readable name used for logging and factory lookup.
    name: str = ""

    def sample(
        self,
        n_samples: int,
        low: float,
        high: float,
        seed: int | None = None,
    ) -> np.ndarray:
        """
        Generate *n_samples* values in the closed interval ``[low, high]``.

        Args:
            n_samples: Number of sample points to generate.  Must be ≥ 1.
            low:       Lower bound of the parameter range (inclusive).
            high:      Upper bound of the parameter range (inclusive).
            seed:      Optional random seed for reproducibility.
                       Ignored by :class:`UniformSampler`.

        Returns:
            Sorted 1-D numpy array of *n_samples* floats in ``[low, high]``.

        Raises:
            ValueError: If *n_samples* < 1 or *low* >= *high*.
        """
        if n_samples < 1:
            raise ValueError(f"n_samples must be ≥ 1, got {n_samples}")
        if low >= high:
            raise ValueError(f"low ({low}) must be < high ({high})")

        samples = self._generate(n_samples=n_samples, low=low, high=high, seed=seed)

        logger.debug(
            "DOE samples generated",
            sampler=self.name,
            n_samples=n_samples,
            low=low,
            high=high,
            seed=seed,
        )
        return np.sort(samples)

    @abc.abstractmethod
    def _generate(
        self,
        n_samples: int,
        low: float,
        high: float,
        seed: int | None,
    ) -> np.ndarray:
        """Generate raw (unsorted) samples.  Called only after validation."""


# ---------------------------------------------------------------------------
# Concrete implementations
# ---------------------------------------------------------------------------


class LHSSampler(BaseSampler):
    """
    Latin Hypercube Sampler.

    Divides the parameter range into *n* equal strata and draws one sample
    from each stratum, ensuring good space coverage with few points.

    Backed by :class:`scipy.stats.qmc.LatinHypercube`.
    """

    name = "lhs"

    def _generate(
        self,
        n_samples: int,
        low: float,
        high: float,
        seed: int | None,
    ) -> np.ndarray:
        sampler = LatinHypercube(d=1, seed=seed)
        unit_samples = sampler.random(n=n_samples)          # shape (n, 1) in [0, 1)
        scaled = qmc_scale(unit_samples, l_bounds=[low], u_bounds=[high])
        return scaled.ravel()


class SobolSampler(BaseSampler):
    """
    Sobol Quasi-Random Sequence Sampler.

    Produces a low-discrepancy sequence that fills the parameter space more
    evenly than pseudo-random sampling, especially for powers-of-2 sample
    counts.

    Backed by :class:`scipy.stats.qmc.Sobol`.

    Note:
        Sobol sequences are most effective when *n_samples* is a power of 2.
        Non-power-of-2 values are accepted but may have higher discrepancy.
    """

    name = "sobol"

    def _generate(
        self,
        n_samples: int,
        low: float,
        high: float,
        seed: int | None,
    ) -> np.ndarray:
        sampler = Sobol(d=1, scramble=True, seed=seed)
        unit_samples = sampler.random(n=n_samples)          # shape (n, 1) in [0, 1)
        scaled = qmc_scale(unit_samples, l_bounds=[low], u_bounds=[high])
        return scaled.ravel()


class UniformSampler(BaseSampler):
    """
    Uniform (evenly-spaced) Grid Sampler.

    Places *n_samples* points at equal intervals between *low* and *high*
    (both endpoints included).  Deterministic — the *seed* argument is ignored.

    Useful for small parameter sweeps or diagnostic runs.
    """

    name = "uniform"

    def _generate(
        self,
        n_samples: int,
        low: float,
        high: float,
        seed: int | None,  # noqa: ARG002
    ) -> np.ndarray:
        return np.linspace(low, high, num=n_samples)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, BaseSampler] = {
    "lhs": LHSSampler(),
    "sobol": SobolSampler(),
    "uniform": UniformSampler(),
}


def get_sampler(name: SamplerName | str) -> BaseSampler:
    """
    Return a sampler instance by name.

    Args:
        name: One of ``"lhs"``, ``"sobol"``, or ``"uniform"``.

    Returns:
        :class:`BaseSampler` instance.

    Raises:
        ValueError: If *name* is not recognised.
    """
    key = name.lower().strip()
    if key not in _REGISTRY:
        valid = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unknown sampler '{name}'. Valid options: {valid}")
    return _REGISTRY[key]
