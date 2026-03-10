"""
Tests for digital_twin_ui.doe.sampler and digital_twin_ui.doe.doe_pipeline.

Covers:
  - BaseSampler validation (n_samples, bounds)
  - LHSSampler: range, sorted output, reproducibility, uniqueness
  - SobolSampler: range, sorted output, reproducibility, uniqueness
  - UniformSampler: exact endpoints, even spacing, seed-agnostic
  - get_sampler factory: known names, unknown name
  - DOEPipeline.preview_samples: correct count, bounds, default fallbacks
  - DOEPipeline.run: uses runner, collects results, summary stats
  - DOEResult: completed/failed counts, success_rate, summary(), to_dataframe()
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from digital_twin_ui.doe.sampler import (
    LHSSampler,
    SobolSampler,
    UniformSampler,
    get_sampler,
)
from digital_twin_ui.doe.doe_pipeline import DOEPipeline, DOEResult
from digital_twin_ui.simulation.simulation_runner import RunResult, SimulationStatus


# ===========================================================================
# Helpers
# ===========================================================================


def _make_run_result(speed: float, status: SimulationStatus = SimulationStatus.COMPLETED) -> RunResult:
    """Build a minimal RunResult for mocking."""
    p = Path("/tmp/run_test")
    return RunResult(
        run_id="run_test",
        status=status,
        speed_mm_s=speed,
        run_dir=p,
        input_feb=p / "input.feb",
        log_file=p / "log.txt",
        xplt_file=p / "results.xplt",
        metadata_file=p / "metadata.json",
        command=["febio4", "-i", "input.feb"],
        lc_end_time=4.0,
        time_steps_step2=40,
    )


# ===========================================================================
# BaseSampler validation (via LHSSampler as concrete class)
# ===========================================================================


class TestBaseSamplerValidation:
    def test_n_samples_zero_raises(self):
        s = LHSSampler()
        with pytest.raises(ValueError, match="n_samples must be"):
            s.sample(n_samples=0, low=4.0, high=6.0)

    def test_n_samples_negative_raises(self):
        s = LHSSampler()
        with pytest.raises(ValueError, match="n_samples must be"):
            s.sample(n_samples=-1, low=4.0, high=6.0)

    def test_low_equal_high_raises(self):
        s = LHSSampler()
        with pytest.raises(ValueError, match="low.*must be < high"):
            s.sample(n_samples=5, low=5.0, high=5.0)

    def test_low_greater_than_high_raises(self):
        s = LHSSampler()
        with pytest.raises(ValueError, match="low.*must be < high"):
            s.sample(n_samples=5, low=6.0, high=4.0)

    def test_returns_sorted_array(self):
        s = LHSSampler()
        result = s.sample(n_samples=20, low=0.0, high=1.0, seed=99)
        assert list(result) == sorted(result)

    def test_returns_ndarray(self):
        s = LHSSampler()
        result = s.sample(n_samples=5, low=4.0, high=6.0, seed=0)
        assert isinstance(result, np.ndarray)


# ===========================================================================
# LHSSampler
# ===========================================================================


class TestLHSSampler:
    def setup_method(self):
        self.sampler = LHSSampler()

    def test_name(self):
        assert self.sampler.name == "lhs"

    def test_all_values_in_range(self):
        result = self.sampler.sample(n_samples=50, low=4.0, high=6.0, seed=0)
        assert np.all(result >= 4.0)
        assert np.all(result <= 6.0)

    def test_correct_count(self):
        result = self.sampler.sample(n_samples=15, low=4.0, high=6.0, seed=1)
        assert len(result) == 15

    def test_single_sample(self):
        result = self.sampler.sample(n_samples=1, low=4.0, high=6.0, seed=0)
        assert len(result) == 1
        assert 4.0 <= result[0] <= 6.0

    def test_reproducibility_with_same_seed(self):
        a = self.sampler.sample(n_samples=10, low=4.0, high=6.0, seed=42)
        b = self.sampler.sample(n_samples=10, low=4.0, high=6.0, seed=42)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_give_different_results(self):
        a = self.sampler.sample(n_samples=10, low=4.0, high=6.0, seed=1)
        b = self.sampler.sample(n_samples=10, low=4.0, high=6.0, seed=2)
        assert not np.array_equal(a, b)

    def test_no_seed_produces_results(self):
        result = self.sampler.sample(n_samples=5, low=4.0, high=6.0)
        assert len(result) == 5

    def test_unique_values(self):
        result = self.sampler.sample(n_samples=20, low=4.0, high=6.0, seed=7)
        assert len(np.unique(result)) == len(result), "LHS should produce unique samples"

    def test_stratification_coverage(self):
        """Each stratum [low + i*w, low + (i+1)*w) contains exactly one sample."""
        n = 10
        low, high = 4.0, 6.0
        result = self.sampler.sample(n_samples=n, low=low, high=high, seed=0)
        width = (high - low) / n
        for i in range(n):
            stratum_low = low + i * width
            stratum_high = low + (i + 1) * width
            in_stratum = np.sum((result >= stratum_low) & (result < stratum_high + 1e-12))
            assert in_stratum >= 1, f"Stratum {i} is empty: [{stratum_low:.3f}, {stratum_high:.3f})"

    def test_arbitrary_bounds(self):
        result = self.sampler.sample(n_samples=10, low=100.0, high=200.0, seed=0)
        assert np.all(result >= 100.0)
        assert np.all(result <= 200.0)


# ===========================================================================
# SobolSampler
# ===========================================================================


class TestSobolSampler:
    def setup_method(self):
        self.sampler = SobolSampler()

    def test_name(self):
        assert self.sampler.name == "sobol"

    def test_all_values_in_range(self):
        result = self.sampler.sample(n_samples=16, low=4.0, high=6.0, seed=0)
        assert np.all(result >= 4.0)
        assert np.all(result <= 6.0)

    def test_correct_count(self):
        result = self.sampler.sample(n_samples=8, low=4.0, high=6.0, seed=0)
        assert len(result) == 8

    def test_single_sample(self):
        result = self.sampler.sample(n_samples=1, low=4.0, high=6.0, seed=0)
        assert len(result) == 1

    def test_reproducibility_with_same_seed(self):
        a = self.sampler.sample(n_samples=8, low=4.0, high=6.0, seed=0)
        b = self.sampler.sample(n_samples=8, low=4.0, high=6.0, seed=0)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_give_different_results(self):
        a = self.sampler.sample(n_samples=8, low=4.0, high=6.0, seed=1)
        b = self.sampler.sample(n_samples=8, low=4.0, high=6.0, seed=2)
        assert not np.array_equal(a, b)

    def test_non_power_of_two(self):
        """Sobol should handle non-power-of-2 counts without error."""
        result = self.sampler.sample(n_samples=10, low=4.0, high=6.0, seed=0)
        assert len(result) == 10
        assert np.all(result >= 4.0)
        assert np.all(result <= 6.0)

    def test_unique_values(self):
        result = self.sampler.sample(n_samples=16, low=4.0, high=6.0, seed=5)
        assert len(np.unique(result)) == len(result)


# ===========================================================================
# UniformSampler
# ===========================================================================


class TestUniformSampler:
    def setup_method(self):
        self.sampler = UniformSampler()

    def test_name(self):
        assert self.sampler.name == "uniform"

    def test_exact_endpoints(self):
        result = self.sampler.sample(n_samples=5, low=4.0, high=6.0)
        assert result[0] == pytest.approx(4.0)
        assert result[-1] == pytest.approx(6.0)

    def test_even_spacing(self):
        result = self.sampler.sample(n_samples=5, low=4.0, high=6.0)
        diffs = np.diff(result)
        assert np.allclose(diffs, diffs[0]), "Uniform sampler should produce equal spacing"

    def test_single_sample_midpoint(self):
        """With n=1, linspace returns only low."""
        result = self.sampler.sample(n_samples=1, low=4.0, high=6.0)
        assert len(result) == 1
        assert result[0] == pytest.approx(4.0)

    def test_two_samples(self):
        result = self.sampler.sample(n_samples=2, low=4.0, high=6.0)
        assert result[0] == pytest.approx(4.0)
        assert result[1] == pytest.approx(6.0)

    def test_correct_count(self):
        result = self.sampler.sample(n_samples=11, low=4.0, high=6.0)
        assert len(result) == 11

    def test_seed_ignored(self):
        """Uniform sampler is deterministic — same result regardless of seed."""
        a = self.sampler.sample(n_samples=10, low=4.0, high=6.0, seed=1)
        b = self.sampler.sample(n_samples=10, low=4.0, high=6.0, seed=999)
        np.testing.assert_array_equal(a, b)

    def test_no_seed_needed(self):
        result = self.sampler.sample(n_samples=5, low=4.0, high=6.0)
        assert len(result) == 5

    def test_known_values(self):
        """5-point uniform sweep over [4, 6] → [4.0, 4.5, 5.0, 5.5, 6.0]."""
        result = self.sampler.sample(n_samples=5, low=4.0, high=6.0)
        expected = np.array([4.0, 4.5, 5.0, 5.5, 6.0])
        np.testing.assert_allclose(result, expected)


# ===========================================================================
# get_sampler factory
# ===========================================================================


class TestGetSampler:
    def test_returns_lhs_sampler(self):
        s = get_sampler("lhs")
        assert isinstance(s, LHSSampler)

    def test_returns_sobol_sampler(self):
        s = get_sampler("sobol")
        assert isinstance(s, SobolSampler)

    def test_returns_uniform_sampler(self):
        s = get_sampler("uniform")
        assert isinstance(s, UniformSampler)

    def test_case_insensitive(self):
        assert isinstance(get_sampler("LHS"), LHSSampler)
        assert isinstance(get_sampler("SOBOL"), SobolSampler)
        assert isinstance(get_sampler("Uniform"), UniformSampler)

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown sampler"):
            get_sampler("random_forest")

    def test_error_lists_valid_options(self):
        with pytest.raises(ValueError, match="lhs"):
            get_sampler("bad")

    def test_returns_same_instance(self):
        """Factory returns singletons — same object on repeated calls."""
        assert get_sampler("lhs") is get_sampler("lhs")


# ===========================================================================
# DOEResult
# ===========================================================================


class TestDOEResult:
    def _make_result(self, statuses: list[SimulationStatus]) -> DOEResult:
        speeds = np.linspace(4.0, 6.0, len(statuses))
        run_results = [_make_run_result(float(s), st) for s, st in zip(speeds, statuses)]
        return DOEResult(
            sampler_name="lhs",
            n_requested=len(statuses),
            seed=0,
            speed_samples=speeds,
            run_results=run_results,
            wall_time_s=1.5,
        )

    def test_completed_count(self):
        r = self._make_result([SimulationStatus.COMPLETED] * 3 + [SimulationStatus.FAILED])
        assert r.completed == 3

    def test_failed_count(self):
        r = self._make_result([SimulationStatus.COMPLETED, SimulationStatus.FAILED, SimulationStatus.TIMEOUT])
        assert r.failed == 2

    def test_success_rate_all_ok(self):
        r = self._make_result([SimulationStatus.COMPLETED] * 5)
        assert r.success_rate == pytest.approx(1.0)

    def test_success_rate_all_failed(self):
        r = self._make_result([SimulationStatus.FAILED] * 4)
        assert r.success_rate == pytest.approx(0.0)

    def test_success_rate_partial(self):
        r = self._make_result([SimulationStatus.COMPLETED] * 2 + [SimulationStatus.FAILED] * 2)
        assert r.success_rate == pytest.approx(0.5)

    def test_success_rate_empty(self):
        r = DOEResult(
            sampler_name="lhs",
            n_requested=0,
            seed=None,
            speed_samples=np.array([]),
        )
        assert r.success_rate == pytest.approx(0.0)

    def test_summary_keys(self):
        r = self._make_result([SimulationStatus.COMPLETED] * 3)
        s = r.summary()
        assert "sampler_name" in s
        assert "n_requested" in s
        assert "n_completed" in s
        assert "n_failed" in s
        assert "success_rate" in s
        assert "wall_time_s" in s
        assert "seed" in s
        assert "speed_range" in s

    def test_summary_values(self):
        r = self._make_result([SimulationStatus.COMPLETED] * 2 + [SimulationStatus.FAILED])
        s = r.summary()
        assert s["n_completed"] == 2
        assert s["n_failed"] == 1
        assert s["sampler_name"] == "lhs"

    def test_to_dataframe(self):
        pd = pytest.importorskip("pandas")
        r = self._make_result([SimulationStatus.COMPLETED] * 3)
        df = r.to_dataframe()
        assert len(df) == 3
        assert "speed_mm_s" in df.columns
        assert "status" in df.columns
        assert "run_id" in df.columns

    def test_to_dataframe_status_values(self):
        pd = pytest.importorskip("pandas")
        r = self._make_result([SimulationStatus.COMPLETED, SimulationStatus.FAILED])
        df = r.to_dataframe()
        statuses = df["status"].tolist()
        assert "COMPLETED" in statuses
        assert "FAILED" in statuses


# ===========================================================================
# DOEPipeline
# ===========================================================================


class TestDOEPipelinePreviewSamples:
    """Tests for DOEPipeline.preview_samples (no simulation runs)."""

    def test_returns_correct_count(self, clean_settings_cache):
        pipeline = DOEPipeline()
        samples = pipeline.preview_samples(n_samples=5, sampler_name="uniform", speed_min=4.0, speed_max=6.0)
        assert len(samples) == 5

    def test_returns_values_in_range(self, clean_settings_cache):
        pipeline = DOEPipeline()
        samples = pipeline.preview_samples(n_samples=10, sampler_name="lhs", speed_min=4.0, speed_max=6.0, seed=0)
        assert np.all(samples >= 4.0)
        assert np.all(samples <= 6.0)

    def test_uses_config_defaults(self, clean_settings_cache):
        """When called with no args, uses defaults from config."""
        pipeline = DOEPipeline()
        samples = pipeline.preview_samples(seed=0)
        cfg = pipeline._cfg
        assert len(samples) == cfg.doe.default_num_samples
        assert np.all(samples >= cfg.doe.speed_min_mm_s)
        assert np.all(samples <= cfg.doe.speed_max_mm_s)

    def test_returns_sorted(self, clean_settings_cache):
        pipeline = DOEPipeline()
        samples = pipeline.preview_samples(n_samples=8, sampler_name="sobol", speed_min=4.0, speed_max=6.0, seed=1)
        assert list(samples) == sorted(samples)

    def test_uniform_known_values(self, clean_settings_cache):
        pipeline = DOEPipeline()
        samples = pipeline.preview_samples(n_samples=3, sampler_name="uniform", speed_min=4.0, speed_max=6.0)
        np.testing.assert_allclose(samples, [4.0, 5.0, 6.0])


class TestDOEPipelineRun:
    """Tests for DOEPipeline.run using a mocked SimulationRunner."""

    def _make_mock_runner(self, status: SimulationStatus = SimulationStatus.COMPLETED):
        """Return a mock runner whose .run() returns a fixed RunResult."""
        runner = MagicMock(spec=["run"])
        runner.run.side_effect = lambda speed_mm_s: _make_run_result(speed_mm_s, status)
        return runner

    def test_run_calls_runner_n_times(self, clean_settings_cache):
        mock_runner = self._make_mock_runner()
        pipeline = DOEPipeline(runner=mock_runner)
        pipeline.run(n_samples=5, sampler_name="uniform", speed_min=4.0, speed_max=6.0)
        assert mock_runner.run.call_count == 5

    def test_run_returns_doe_result(self, clean_settings_cache):
        mock_runner = self._make_mock_runner()
        pipeline = DOEPipeline(runner=mock_runner)
        result = pipeline.run(n_samples=3, sampler_name="uniform", speed_min=4.0, speed_max=6.0)
        assert isinstance(result, DOEResult)

    def test_run_result_has_correct_count(self, clean_settings_cache):
        mock_runner = self._make_mock_runner()
        pipeline = DOEPipeline(runner=mock_runner)
        result = pipeline.run(n_samples=4, sampler_name="lhs", speed_min=4.0, speed_max=6.0, seed=0)
        assert len(result.run_results) == 4

    def test_run_completed_count(self, clean_settings_cache):
        mock_runner = self._make_mock_runner(SimulationStatus.COMPLETED)
        pipeline = DOEPipeline(runner=mock_runner)
        result = pipeline.run(n_samples=3, sampler_name="uniform", speed_min=4.0, speed_max=6.0)
        assert result.completed == 3
        assert result.failed == 0

    def test_run_failed_count(self, clean_settings_cache):
        mock_runner = self._make_mock_runner(SimulationStatus.FAILED)
        pipeline = DOEPipeline(runner=mock_runner)
        result = pipeline.run(n_samples=3, sampler_name="uniform", speed_min=4.0, speed_max=6.0)
        assert result.failed == 3
        assert result.completed == 0

    def test_run_stores_sampler_name(self, clean_settings_cache):
        mock_runner = self._make_mock_runner()
        pipeline = DOEPipeline(runner=mock_runner)
        result = pipeline.run(n_samples=2, sampler_name="sobol", speed_min=4.0, speed_max=6.0, seed=0)
        assert result.sampler_name == "sobol"

    def test_run_stores_seed(self, clean_settings_cache):
        mock_runner = self._make_mock_runner()
        pipeline = DOEPipeline(runner=mock_runner)
        result = pipeline.run(n_samples=2, sampler_name="lhs", speed_min=4.0, speed_max=6.0, seed=77)
        assert result.seed == 77

    def test_run_wall_time_positive(self, clean_settings_cache):
        mock_runner = self._make_mock_runner()
        pipeline = DOEPipeline(runner=mock_runner)
        result = pipeline.run(n_samples=2, sampler_name="uniform", speed_min=4.0, speed_max=6.0)
        assert result.wall_time_s >= 0.0

    def test_run_uses_config_defaults(self, clean_settings_cache):
        mock_runner = self._make_mock_runner()
        pipeline = DOEPipeline(runner=mock_runner)
        result = pipeline.run(seed=0)
        cfg = pipeline._cfg
        assert mock_runner.run.call_count == cfg.doe.default_num_samples

    def test_run_speed_values_in_range(self, clean_settings_cache):
        """The speeds passed to the runner must respect the requested bounds."""
        speeds_called = []
        mock_runner = MagicMock(spec=["run"])
        mock_runner.run.side_effect = lambda speed_mm_s: (
            speeds_called.append(speed_mm_s) or _make_run_result(speed_mm_s)
        )
        pipeline = DOEPipeline(runner=mock_runner)
        pipeline.run(n_samples=8, sampler_name="lhs", speed_min=4.0, speed_max=6.0, seed=0)
        for s in speeds_called:
            assert 4.0 <= s <= 6.0

    def test_run_reproducible_with_seed(self, clean_settings_cache):
        """Same seed → same speed sequence → runner called with same args."""
        speeds_a, speeds_b = [], []

        def make_runner(collector):
            r = MagicMock(spec=["run"])
            r.run.side_effect = lambda speed_mm_s: (
                collector.append(speed_mm_s) or _make_run_result(speed_mm_s)
            )
            return r

        DOEPipeline(runner=make_runner(speeds_a)).run(
            n_samples=5, sampler_name="lhs", speed_min=4.0, speed_max=6.0, seed=42
        )
        DOEPipeline(runner=make_runner(speeds_b)).run(
            n_samples=5, sampler_name="lhs", speed_min=4.0, speed_max=6.0, seed=42
        )
        np.testing.assert_allclose(speeds_a, speeds_b)

    def test_run_with_custom_sampler_instance(self, clean_settings_cache):
        mock_runner = self._make_mock_runner()
        pipeline = DOEPipeline(runner=mock_runner)
        custom_sampler = UniformSampler()
        result = pipeline.run(
            n_samples=3,
            sampler=custom_sampler,
            speed_min=4.0,
            speed_max=6.0,
        )
        assert result.sampler_name == "uniform"
        assert mock_runner.run.call_count == 3
