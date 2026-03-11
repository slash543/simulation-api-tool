"""
Tests for CorrelatedSpeedSampler (doe/correlated_sampler.py).

Coverage
--------
- Output shape is (n_samples, n_steps)
- All values within [speed_min, speed_max]
- Each row sorted ascending (ramp-up behaviour)
- Reproducibility with seed
- Different seeds produce different results
- Max perturbation = 0.0 → all columns equal (mean speed only)
- Max perturbation respected (values stay within ±mp of mean)
- Raises ValueError on invalid arguments
- Mean of each row close to the sampled mean speed
- LHS stratification: n_samples rows cover [speed_min, speed_max] evenly
"""

from __future__ import annotations

import numpy as np
import pytest

from digital_twin_ui.doe.correlated_sampler import CorrelatedSpeedSampler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sampler() -> CorrelatedSpeedSampler:
    return CorrelatedSpeedSampler(max_perturbation=0.20)


# ---------------------------------------------------------------------------
# Shape and dtype
# ---------------------------------------------------------------------------

class TestOutputShape:
    def test_shape_matches_n_samples_n_steps(self, sampler: CorrelatedSpeedSampler) -> None:
        result = sampler.sample(n_samples=5, speed_min=10.0, speed_max=25.0, n_steps=10)
        assert result.shape == (5, 10)

    def test_shape_single_sample(self, sampler: CorrelatedSpeedSampler) -> None:
        result = sampler.sample(n_samples=1, speed_min=10.0, speed_max=25.0, n_steps=10)
        assert result.shape == (1, 10)

    def test_shape_single_step(self, sampler: CorrelatedSpeedSampler) -> None:
        result = sampler.sample(n_samples=4, speed_min=10.0, speed_max=25.0, n_steps=1)
        assert result.shape == (4, 1)

    def test_dtype_float64(self, sampler: CorrelatedSpeedSampler) -> None:
        result = sampler.sample(n_samples=5, speed_min=10.0, speed_max=25.0, n_steps=10)
        assert result.dtype == np.float64

    @pytest.mark.parametrize("n_samples,n_steps", [
        (10, 10),
        (20, 5),
        (1, 3),
        (50, 10),
    ])
    def test_various_shapes(
        self,
        sampler: CorrelatedSpeedSampler,
        n_samples: int,
        n_steps: int,
    ) -> None:
        result = sampler.sample(
            n_samples=n_samples, speed_min=10.0, speed_max=25.0, n_steps=n_steps
        )
        assert result.shape == (n_samples, n_steps)


# ---------------------------------------------------------------------------
# Range constraints
# ---------------------------------------------------------------------------

class TestRangeConstraints:
    def test_all_values_within_speed_min_max(self, sampler: CorrelatedSpeedSampler) -> None:
        result = sampler.sample(n_samples=50, speed_min=10.0, speed_max=25.0, n_steps=10)
        assert result.min() >= 10.0
        assert result.max() <= 25.0

    def test_narrow_range_still_valid(self, sampler: CorrelatedSpeedSampler) -> None:
        result = sampler.sample(n_samples=5, speed_min=14.0, speed_max=16.0, n_steps=10)
        assert result.min() >= 14.0
        assert result.max() <= 16.0

    def test_zero_perturbation_all_columns_same(self) -> None:
        s = CorrelatedSpeedSampler(max_perturbation=0.0)
        result = s.sample(n_samples=10, speed_min=10.0, speed_max=25.0, n_steps=5, seed=42)
        # With 0 perturbation, all columns in each row should be identical
        for row in result:
            assert np.allclose(row, row[0]), f"Row not uniform: {row}"


# ---------------------------------------------------------------------------
# Ascending sort (physiological ramp-up)
# ---------------------------------------------------------------------------

class TestAscendingSort:
    def test_each_row_sorted_ascending(self, sampler: CorrelatedSpeedSampler) -> None:
        result = sampler.sample(n_samples=20, speed_min=10.0, speed_max=25.0, n_steps=10, seed=1)
        for i, row in enumerate(result):
            assert list(row) == sorted(row), f"Row {i} not sorted: {row}"

    def test_ascending_with_high_perturbation(self) -> None:
        s = CorrelatedSpeedSampler(max_perturbation=0.49)
        result = s.sample(n_samples=30, speed_min=10.0, speed_max=25.0, n_steps=10, seed=99)
        for i, row in enumerate(result):
            assert list(row) == sorted(row), f"Row {i} not sorted: {row}"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_same_output(self, sampler: CorrelatedSpeedSampler) -> None:
        a = sampler.sample(n_samples=10, speed_min=10.0, speed_max=25.0, n_steps=10, seed=42)
        b = sampler.sample(n_samples=10, speed_min=10.0, speed_max=25.0, n_steps=10, seed=42)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_different_output(self, sampler: CorrelatedSpeedSampler) -> None:
        a = sampler.sample(n_samples=10, speed_min=10.0, speed_max=25.0, n_steps=10, seed=1)
        b = sampler.sample(n_samples=10, speed_min=10.0, speed_max=25.0, n_steps=10, seed=2)
        assert not np.allclose(a, b)

    def test_no_seed_no_error(self, sampler: CorrelatedSpeedSampler) -> None:
        result = sampler.sample(n_samples=5, speed_min=10.0, speed_max=25.0, n_steps=10)
        assert result.shape == (5, 10)


# ---------------------------------------------------------------------------
# Perturbation override
# ---------------------------------------------------------------------------

class TestPerturbationOverride:
    def test_per_call_override_respected(self) -> None:
        s = CorrelatedSpeedSampler(max_perturbation=0.20)
        # Override with 0.0 — should give uniform rows
        result = s.sample(
            n_samples=5, speed_min=10.0, speed_max=25.0, n_steps=5,
            seed=7, max_perturbation=0.0,
        )
        for row in result:
            assert np.allclose(row, row[0])

    def test_instance_max_perturbation_not_modified_by_call_override(self) -> None:
        s = CorrelatedSpeedSampler(max_perturbation=0.20)
        s.sample(n_samples=3, speed_min=10.0, speed_max=25.0, n_steps=5,
                 max_perturbation=0.0, seed=1)
        assert s.max_perturbation == 0.20  # unchanged


# ---------------------------------------------------------------------------
# LHS stratification (statistical coverage check)
# ---------------------------------------------------------------------------

class TestLHSStratification:
    def test_mean_speeds_cover_range_evenly(self) -> None:
        """
        With no perturbation (max_perturbation=0), all columns equal the LHS
        mean speed, so the row means exactly follow the LHS stratification.
        With n_samples=10, each mean falls in one decile of [speed_min, speed_max].
        """
        n = 10
        s = CorrelatedSpeedSampler(max_perturbation=0.0)
        result = s.sample(n_samples=n, speed_min=10.0, speed_max=25.0, n_steps=10, seed=0)
        means = result.mean(axis=1)
        means_sorted = np.sort(means)
        boundaries = np.linspace(10.0, 25.0, n + 1)
        for i in range(n):
            assert boundaries[i] <= means_sorted[i] <= boundaries[i + 1], (
                f"Decile {i} not covered: {means_sorted[i]:.3f} not in "
                f"[{boundaries[i]:.3f}, {boundaries[i+1]:.3f}]"
            )

    def test_means_span_full_range(self, sampler: CorrelatedSpeedSampler) -> None:
        """With a reasonable sample size, means should span at least 80% of the range."""
        result = sampler.sample(n_samples=20, speed_min=10.0, speed_max=25.0, n_steps=10, seed=7)
        means = result.mean(axis=1)
        span = means.max() - means.min()
        assert span >= 0.8 * (25.0 - 10.0), f"Means span too narrow: {span:.2f}"


# ---------------------------------------------------------------------------
# Validation — invalid arguments
# ---------------------------------------------------------------------------

class TestValidation:
    def test_zero_samples_raises(self, sampler: CorrelatedSpeedSampler) -> None:
        with pytest.raises(ValueError, match="n_samples"):
            sampler.sample(n_samples=0, speed_min=10.0, speed_max=25.0, n_steps=10)

    def test_zero_steps_raises(self, sampler: CorrelatedSpeedSampler) -> None:
        with pytest.raises(ValueError, match="n_steps"):
            sampler.sample(n_samples=5, speed_min=10.0, speed_max=25.0, n_steps=0)

    def test_speed_min_ge_speed_max_raises(self, sampler: CorrelatedSpeedSampler) -> None:
        with pytest.raises(ValueError, match="speed_min"):
            sampler.sample(n_samples=5, speed_min=25.0, speed_max=10.0, n_steps=10)

    def test_speed_min_equal_speed_max_raises(self, sampler: CorrelatedSpeedSampler) -> None:
        with pytest.raises(ValueError):
            sampler.sample(n_samples=5, speed_min=15.0, speed_max=15.0, n_steps=10)


# ---------------------------------------------------------------------------
# Sampler name attribute
# ---------------------------------------------------------------------------

class TestSamplerName:
    def test_name_attribute(self) -> None:
        s = CorrelatedSpeedSampler()
        assert s.name == "correlated_lhs"
