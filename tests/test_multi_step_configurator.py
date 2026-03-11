"""
Tests for MultiStepConfigurator (simulation/multi_step_configurator.py).

Strategy
--------
- A minimal 3-step synthetic FEB file is used for unit tests.
  This avoids the 17 MB real templates in fast unit tests.
- Integration tests validate against the real DT_BT_14Fr_FO_10E_IR12.feb
  template (marked with @pytest.mark.integration).

Coverage
--------
- configure() writes output file
- Speed/time calculations are correct
- ramp_i = displacement_i / speed_i
- step_duration_i = ramp_i + dwell_time_s
- time_steps_i = max(10, ceil(step_duration_i / base_step_size))
- LC points rebuilt correctly (3-point and 4-point patterns)
- LC timing is sequential (t_start of each = t_end of previous)
- total_duration_s is sum of all step durations + initial t_start
- MultiStepConfigResult fields populated
- as_dict() serialisable
- Validation: wrong number of speeds → ValueError
- Validation: non-positive speed → ValueError
- Validation: missing template file → FileNotFoundError
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

try:
    from lxml import etree
    _LXML_AVAILABLE = True
except ImportError:
    _LXML_AVAILABLE = False

from digital_twin_ui.simulation.multi_step_configurator import (
    MultiStepConfigurator,
    MultiStepConfigResult,
)
from digital_twin_ui.simulation.template_registry import SpeedRange, TemplateConfig


# ---------------------------------------------------------------------------
# Minimal synthetic FEB for unit tests (3 steps)
# ---------------------------------------------------------------------------

MINIMAL_MULTI_FEB = """\
<?xml version="1.0" encoding="ISO-8859-1"?>
<febio_spec version="4.0">
  <Step>
    <step id="1" name="Step1">
      <Control>
        <time_steps>40</time_steps>
        <step_size>0.1</step_size>
      </Control>
      <Boundary>
        <bc name="insert1" type="prescribed displacement">
          <value lc="1">20</value>
        </bc>
      </Boundary>
    </step>
    <step id="2" name="Step2">
      <Control>
        <time_steps>40</time_steps>
        <step_size>0.1</step_size>
      </Control>
      <Boundary>
        <bc name="insert2" type="prescribed displacement">
          <value lc="2">10</value>
        </bc>
      </Boundary>
    </step>
    <step id="3" name="Step3">
      <Control>
        <time_steps>40</time_steps>
        <step_size>0.1</step_size>
      </Control>
      <Boundary>
        <bc name="insert3" type="prescribed displacement">
          <value lc="3">10</value>
        </bc>
      </Boundary>
    </step>
  </Step>
  <LoadData>
    <load_controller id="1" name="LC1" type="loadcurve">
      <interpolate>SMOOTH STEP</interpolate>
      <points>
        <pt>0,0</pt>
        <pt>2,1</pt>
        <pt>3,1</pt>
      </points>
    </load_controller>
    <load_controller id="2" name="LC2" type="loadcurve">
      <interpolate>SMOOTH STEP</interpolate>
      <points>
        <pt>3,0</pt>
        <pt>4,1</pt>
        <pt>5,1</pt>
      </points>
    </load_controller>
    <load_controller id="3" name="LC3" type="loadcurve">
      <interpolate>SMOOTH STEP</interpolate>
      <points>
        <pt>5,0</pt>
        <pt>6,1</pt>
        <pt>7,1</pt>
      </points>
    </load_controller>
  </LoadData>
</febio_spec>
"""

#  Template describes: 3 steps, displacements=[20, 10, 10] mm, base_step_size=0.1s
DISPLACEMENTS = [20.0, 10.0, 10.0]
N_STEPS = 3
BASE_STEP_SIZE = 0.1


def _make_template(tmp_path: Path, feb_path: Path) -> TemplateConfig:
    """Return a TemplateConfig pointing at a minimal 3-step FEB file."""
    return TemplateConfig(
        name="test_3step",
        label="Test 3-step",
        feb_file=feb_path.name,
        n_steps=N_STEPS,
        base_step_size=BASE_STEP_SIZE,
        default_dwell_time_s=1.0,
        displacements_mm=DISPLACEMENTS,
        speed_range=SpeedRange(10.0, 25.0),
        _project_root=feb_path.parent.parent,  # templates/<feb_file>
    )


@pytest.fixture()
def minimal_multi_feb(tmp_path: Path) -> Path:
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()
    p = templates_dir / "test_3step.feb"
    p.write_bytes(MINIMAL_MULTI_FEB.encode("ISO-8859-1"))
    return p


@pytest.fixture()
def template_cfg(minimal_multi_feb: Path, tmp_path: Path) -> TemplateConfig:
    return _make_template(tmp_path, minimal_multi_feb)


@pytest.fixture()
def configurator(template_cfg: TemplateConfig) -> MultiStepConfigurator:
    return MultiStepConfigurator(template_cfg)


@pytest.fixture()
def output_feb(tmp_path: Path) -> Path:
    return tmp_path / "output" / "input.feb"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_output(output_feb: Path):
    """Parse the written output FEB and return the root element."""
    parser = etree.XMLParser(remove_blank_text=False)
    return etree.parse(str(output_feb), parser).getroot()


def _get_lc_points(root, lc_id: int) -> list[tuple[float, float]]:
    """Return the (time, value) pairs from a load_controller by id."""
    for lc in root.findall(".//load_controller"):
        if int(lc.get("id", -1)) == lc_id:
            pts = lc.find("points")
            result = []
            for pt in pts.findall("pt"):
                t_str, v_str = pt.text.split(",")
                result.append((float(t_str.strip()), float(v_str.strip())))
            return result
    raise AssertionError(f"LC id={lc_id} not found")


def _get_step_time_steps(root, step_id: int) -> int:
    for step in root.findall(".//step"):
        if int(step.get("id", -1)) == step_id:
            return int(step.find("Control/time_steps").text)
    raise AssertionError(f"Step id={step_id} not found")


# ---------------------------------------------------------------------------
# Output file creation
# ---------------------------------------------------------------------------

class TestOutputCreation:
    def test_output_file_created(
        self, configurator: MultiStepConfigurator, output_feb: Path
    ) -> None:
        configurator.configure(
            speeds_mm_s=[15.0, 15.0, 15.0],
            dwell_time_s=1.0,
            output_path=output_feb,
        )
        assert output_feb.exists()

    def test_output_is_valid_xml(
        self, configurator: MultiStepConfigurator, output_feb: Path
    ) -> None:
        configurator.configure(
            speeds_mm_s=[15.0, 15.0, 15.0],
            dwell_time_s=1.0,
            output_path=output_feb,
        )
        root = _parse_output(output_feb)
        assert root.tag == "febio_spec"

    def test_parent_dir_created_if_missing(
        self, configurator: MultiStepConfigurator, tmp_path: Path
    ) -> None:
        deep = tmp_path / "a" / "b" / "c" / "input.feb"
        configurator.configure(speeds_mm_s=[15.0, 15.0, 15.0], dwell_time_s=1.0, output_path=deep)
        assert deep.exists()


# ---------------------------------------------------------------------------
# Return value — MultiStepConfigResult
# ---------------------------------------------------------------------------

class TestConfigResult:
    def test_result_type(
        self, configurator: MultiStepConfigurator, output_feb: Path
    ) -> None:
        result = configurator.configure(
            speeds_mm_s=[15.0, 15.0, 15.0], dwell_time_s=1.0, output_path=output_feb
        )
        assert isinstance(result, MultiStepConfigResult)

    def test_result_speeds_mm_s(
        self, configurator: MultiStepConfigurator, output_feb: Path
    ) -> None:
        speeds = [12.0, 15.0, 18.0]
        result = configurator.configure(speeds_mm_s=speeds, dwell_time_s=1.0, output_path=output_feb)
        assert result.speeds_mm_s == speeds

    def test_result_template_name(
        self, configurator: MultiStepConfigurator, output_feb: Path
    ) -> None:
        result = configurator.configure(
            speeds_mm_s=[15.0, 15.0, 15.0], dwell_time_s=1.0, output_path=output_feb
        )
        assert result.template_name == "test_3step"

    def test_result_step_durations_length(
        self, configurator: MultiStepConfigurator, output_feb: Path
    ) -> None:
        result = configurator.configure(
            speeds_mm_s=[15.0, 15.0, 15.0], dwell_time_s=1.0, output_path=output_feb
        )
        assert len(result.step_durations) == N_STEPS

    def test_result_lc_timing_length(
        self, configurator: MultiStepConfigurator, output_feb: Path
    ) -> None:
        result = configurator.configure(
            speeds_mm_s=[15.0, 15.0, 15.0], dwell_time_s=1.0, output_path=output_feb
        )
        assert len(result.lc_timing) == N_STEPS

    def test_result_output_path(
        self, configurator: MultiStepConfigurator, output_feb: Path
    ) -> None:
        result = configurator.configure(
            speeds_mm_s=[15.0, 15.0, 15.0], dwell_time_s=1.0, output_path=output_feb
        )
        assert result.output_path == output_feb

    def test_as_dict_serialisable(
        self, configurator: MultiStepConfigurator, output_feb: Path
    ) -> None:
        import json
        result = configurator.configure(
            speeds_mm_s=[15.0, 15.0, 15.0], dwell_time_s=1.0, output_path=output_feb
        )
        d = result.as_dict()
        # Should not raise
        json.dumps(d, default=str)

    def test_as_dict_keys(
        self, configurator: MultiStepConfigurator, output_feb: Path
    ) -> None:
        result = configurator.configure(
            speeds_mm_s=[15.0, 15.0, 15.0], dwell_time_s=1.0, output_path=output_feb
        )
        d = result.as_dict()
        for key in ("speeds_mm_s", "dwell_time_s", "template_name",
                    "total_duration_s", "step_durations", "lc_timing", "output_path"):
            assert key in d


# ---------------------------------------------------------------------------
# Physics — timing calculations
# ---------------------------------------------------------------------------

class TestTimingCalculations:
    @pytest.mark.parametrize("speed,disp,dwell,expected_ramp", [
        (20.0, 20.0, 1.0, 1.0),    # 20mm / 20mm/s = 1s
        (10.0, 20.0, 1.0, 2.0),    # 20mm / 10mm/s = 2s
        (25.0, 10.0, 2.0, 0.4),    # 10mm / 25mm/s = 0.4s
    ])
    def test_ramp_duration_formula(
        self,
        configurator: MultiStepConfigurator,
        output_feb: Path,
        speed: float,
        disp: float,
        dwell: float,
        expected_ramp: float,
    ) -> None:
        # step 0 has disp=20mm, so set speed such that disp/speed = expected_ramp
        actual_speed = DISPLACEMENTS[0] / expected_ramp
        result = configurator.configure(
            speeds_mm_s=[actual_speed, actual_speed, actual_speed],
            dwell_time_s=dwell,
            output_path=output_feb,
        )
        # step_durations[0] = ramp + dwell
        assert result.step_durations[0] == pytest.approx(expected_ramp + dwell, rel=1e-6)

    def test_total_duration_is_sequential_sum(
        self, configurator: MultiStepConfigurator, output_feb: Path
    ) -> None:
        speeds = [20.0, 10.0, 25.0]
        dwell = 1.0
        result = configurator.configure(speeds_mm_s=speeds, dwell_time_s=dwell, output_path=output_feb)
        expected = sum(d / s + dwell for d, s in zip(DISPLACEMENTS, speeds))
        assert result.total_duration_s == pytest.approx(expected, rel=1e-9)

    def test_lc_timing_sequential(
        self, configurator: MultiStepConfigurator, output_feb: Path
    ) -> None:
        result = configurator.configure(
            speeds_mm_s=[15.0, 15.0, 15.0], dwell_time_s=1.0, output_path=output_feb
        )
        for i in range(1, len(result.lc_timing)):
            prev_end = result.lc_timing[i - 1]["t_end"]
            curr_start = result.lc_timing[i]["t_start"]
            assert curr_start == pytest.approx(prev_end, abs=1e-9), (
                f"LC {i} t_start={curr_start} != prev t_end={prev_end}"
            )

    @pytest.mark.parametrize("speed,disp,dwell,bss,expected_ts", [
        # ramp=disp/speed, duration=ramp+dwell, ts=max(10, ceil(duration/bss))
        (20.0, 20.0, 1.0, 0.1, max(10, math.ceil((20.0/20.0 + 1.0) / 0.1))),
        (10.0, 20.0, 1.0, 0.1, max(10, math.ceil((20.0/10.0 + 1.0) / 0.1))),
        (100.0, 20.0, 0.0, 0.1, max(10, math.ceil((20.0/100.0 + 0.0) / 0.1))),
    ])
    def test_time_steps_formula(
        self,
        minimal_multi_feb: Path,
        tmp_path: Path,
        output_feb: Path,
        speed: float,
        disp: float,
        dwell: float,
        bss: float,
        expected_ts: int,
    ) -> None:
        tc = TemplateConfig(
            name="test_3step",
            label="t",
            feb_file=minimal_multi_feb.name,
            n_steps=N_STEPS,
            base_step_size=bss,
            default_dwell_time_s=dwell,
            displacements_mm=DISPLACEMENTS,
            speed_range=SpeedRange(1.0, 200.0),
            _project_root=minimal_multi_feb.parent.parent,
        )
        cfg = MultiStepConfigurator(tc)
        cfg.configure(speeds_mm_s=[speed, speed, speed], dwell_time_s=dwell, output_path=output_feb)
        root = _parse_output(output_feb)
        ts = _get_step_time_steps(root, step_id=1)
        assert ts == expected_ts, f"Expected time_steps={expected_ts}, got {ts}"

    def test_time_steps_minimum_10(
        self, configurator: MultiStepConfigurator, output_feb: Path
    ) -> None:
        # Very fast speed → tiny ramp → time_steps should floor at 10
        result = configurator.configure(
            speeds_mm_s=[1000.0, 1000.0, 1000.0], dwell_time_s=0.001, output_path=output_feb
        )
        root = _parse_output(output_feb)
        for step_id in range(1, N_STEPS + 1):
            ts = _get_step_time_steps(root, step_id)
            assert ts >= 10, f"Step {step_id} time_steps={ts} < 10"


# ---------------------------------------------------------------------------
# LC points rebuilt correctly
# ---------------------------------------------------------------------------

class TestLCPoints:
    def test_lc1_has_three_points(
        self, configurator: MultiStepConfigurator, output_feb: Path
    ) -> None:
        configurator.configure(speeds_mm_s=[15.0, 15.0, 15.0], dwell_time_s=1.0, output_path=output_feb)
        root = _parse_output(output_feb)
        pts = _get_lc_points(root, lc_id=1)
        assert len(pts) == 3  # non-last LCs get 3-point pattern

    def test_last_lc_has_four_points(
        self, configurator: MultiStepConfigurator, output_feb: Path
    ) -> None:
        # LC3 is last (highest t_start in original file)
        configurator.configure(speeds_mm_s=[15.0, 15.0, 15.0], dwell_time_s=1.0, output_path=output_feb)
        root = _parse_output(output_feb)
        pts = _get_lc_points(root, lc_id=3)
        assert len(pts) == 4  # last LC gets 4-point pattern

    def test_last_lc_first_two_points_same_time(
        self, configurator: MultiStepConfigurator, output_feb: Path
    ) -> None:
        configurator.configure(speeds_mm_s=[15.0, 15.0, 15.0], dwell_time_s=1.0, output_path=output_feb)
        root = _parse_output(output_feb)
        pts = _get_lc_points(root, lc_id=3)
        assert pts[0][0] == pytest.approx(pts[1][0]), "4-point: first two times should match"
        assert pts[0][1] == pytest.approx(0.0)
        assert pts[1][1] == pytest.approx(0.0)

    def test_lc_starts_at_zero_after_t_cursor(
        self, configurator: MultiStepConfigurator, output_feb: Path
    ) -> None:
        configurator.configure(speeds_mm_s=[15.0, 15.0, 15.0], dwell_time_s=1.0, output_path=output_feb)
        root = _parse_output(output_feb)
        # LC1 starts at t=0 (inherited from original)
        pts1 = _get_lc_points(root, lc_id=1)
        assert pts1[0][0] == pytest.approx(0.0)
        assert pts1[0][1] == pytest.approx(0.0)

    def test_lc_ramp_value_reaches_one(
        self, configurator: MultiStepConfigurator, output_feb: Path
    ) -> None:
        configurator.configure(speeds_mm_s=[15.0, 15.0, 15.0], dwell_time_s=1.0, output_path=output_feb)
        root = _parse_output(output_feb)
        for lc_id in range(1, N_STEPS + 1):
            pts = _get_lc_points(root, lc_id)
            # The second-to-last point should have value=1
            assert pts[-2][1] == pytest.approx(1.0), f"LC{lc_id}: ramp end value not 1"
            assert pts[-1][1] == pytest.approx(1.0), f"LC{lc_id}: dwell end value not 1"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_wrong_number_of_speeds_raises(
        self, configurator: MultiStepConfigurator, output_feb: Path
    ) -> None:
        with pytest.raises(ValueError, match="requires 3 speeds"):
            configurator.configure(speeds_mm_s=[15.0, 15.0], dwell_time_s=1.0, output_path=output_feb)

    def test_extra_speeds_raises(
        self, configurator: MultiStepConfigurator, output_feb: Path
    ) -> None:
        with pytest.raises(ValueError):
            configurator.configure(
                speeds_mm_s=[15.0, 15.0, 15.0, 15.0], dwell_time_s=1.0, output_path=output_feb
            )

    def test_non_positive_speed_raises(
        self, configurator: MultiStepConfigurator, output_feb: Path
    ) -> None:
        with pytest.raises(ValueError, match="positive"):
            configurator.configure(speeds_mm_s=[0.0, 15.0, 15.0], dwell_time_s=1.0, output_path=output_feb)

    def test_negative_speed_raises(
        self, configurator: MultiStepConfigurator, output_feb: Path
    ) -> None:
        with pytest.raises(ValueError):
            configurator.configure(speeds_mm_s=[-5.0, 15.0, 15.0], dwell_time_s=1.0, output_path=output_feb)

    def test_missing_feb_file_raises(self, tmp_path: Path, output_feb: Path) -> None:
        tc = TemplateConfig(
            name="ghost",
            label="Ghost",
            feb_file="does_not_exist.feb",
            n_steps=3,
            base_step_size=0.1,
            default_dwell_time_s=1.0,
            displacements_mm=[10.0, 10.0, 10.0],
            speed_range=SpeedRange(10.0, 25.0),
            _project_root=tmp_path,
        )
        cfg = MultiStepConfigurator(tc)
        with pytest.raises(FileNotFoundError):
            cfg.configure(speeds_mm_s=[15.0, 15.0, 15.0], dwell_time_s=1.0, output_path=output_feb)


# ---------------------------------------------------------------------------
# Integration — real IR12 template
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
IR12_FEB = PROJECT_ROOT / "templates" / "DT_BT_14Fr_FO_10E_IR12.feb"
IR12_YAML = PROJECT_ROOT / "config" / "templates" / "DT_BT_14Fr_FO_10E_IR12.yaml"


@pytest.mark.integration
class TestRealIR12:
    @pytest.fixture(autouse=True)
    def skip_if_missing(self) -> None:
        if not IR12_FEB.exists():
            pytest.skip(f"Real FEB not found: {IR12_FEB}")

    def test_configure_uniform_speed(self, tmp_path: Path) -> None:
        from digital_twin_ui.simulation.template_registry import TemplateRegistry
        reg = TemplateRegistry(
            templates_dir=PROJECT_ROOT / "config" / "templates",
            project_root=PROJECT_ROOT,
        )
        tc = reg.get("DT_BT_14Fr_FO_10E_IR12")
        cfg = MultiStepConfigurator(tc)
        out = tmp_path / "input.feb"
        result = cfg.configure(speeds_mm_s=[15.0] * 10, dwell_time_s=1.0, output_path=out)
        assert out.exists()
        assert len(result.step_durations) == 10
        assert result.total_duration_s > 0

    def test_ir12_lc_timing_sequential(self, tmp_path: Path) -> None:
        from digital_twin_ui.simulation.template_registry import TemplateRegistry
        reg = TemplateRegistry(
            templates_dir=PROJECT_ROOT / "config" / "templates",
            project_root=PROJECT_ROOT,
        )
        tc = reg.get("DT_BT_14Fr_FO_10E_IR12")
        cfg = MultiStepConfigurator(tc)
        out = tmp_path / "input.feb"
        result = cfg.configure(speeds_mm_s=[15.0] * 10, dwell_time_s=1.0, output_path=out)
        for i in range(1, len(result.lc_timing)):
            assert result.lc_timing[i]["t_start"] == pytest.approx(
                result.lc_timing[i - 1]["t_end"], abs=1e-9
            )

    def test_ir12_all_speeds_applied(self, tmp_path: Path) -> None:
        from digital_twin_ui.simulation.template_registry import TemplateRegistry
        reg = TemplateRegistry(
            templates_dir=PROJECT_ROOT / "config" / "templates",
            project_root=PROJECT_ROOT,
        )
        tc = reg.get("DT_BT_14Fr_FO_10E_IR12")
        cfg = MultiStepConfigurator(tc)
        speeds = [10.0 + i for i in range(10)]  # 10,11,...,19 mm/s
        out = tmp_path / "input.feb"
        result = cfg.configure(speeds_mm_s=speeds, dwell_time_s=1.0, output_path=out)
        assert result.speeds_mm_s == speeds
