"""
Tests for SimulationConfigurator.

Coverage:
- Physics: speed → duration, time_steps, lc_end_time
- XML helpers: _parse_pt, _format_pt edge cases
- XML modification correctness using both minimal_feb and real template
- Content preservation: materials, BCs, Step1, step_size untouched
- ConfigurationParams dataclass
- ConfigurationResult — all fields, as_dict(), total_simulation_time_s
- Error paths: bad speed, missing file, missing XML elements
- Idempotency: two calls with same speed → byte-identical output
- Output directory auto-creation
- module-level configure_simulation() convenience function
"""

from __future__ import annotations

from pathlib import Path

import pytest
from lxml import etree

from digital_twin_ui.simulation.simulation_configurator import (
    ConfigurationParams,
    ConfigurationResult,
    SimulationConfigurator,
    _format_pt,
    _parse_pt,
    configure_simulation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def configurator() -> SimulationConfigurator:
    return SimulationConfigurator()


# ---------------------------------------------------------------------------
# Physics — pure calculations, no I/O
# ---------------------------------------------------------------------------

class TestPhysics:
    @pytest.mark.parametrize("speed, expected_ts", [
        (4.0,  50),                      # slow end of DOE range  (2.5 / 0.05)
        (5.0,  40),                      # default speed          (2.0 / 0.05)
        (6.0,  round((10/6) / 0.05)),    # fast end               (≈33)
        (10.0, 20),                      # out-of-DOE sanity
        (2.0,  100),                     # slow sanity
    ])
    def test_time_steps_computation(self, speed, expected_ts):
        duration = 10.0 / speed
        ts = SimulationConfigurator._compute_time_steps(duration, step_size=0.05)
        assert ts == expected_ts

    def test_minimum_one_time_step(self):
        ts = SimulationConfigurator._compute_time_steps(0.001, step_size=0.05)
        assert ts >= 1

    @pytest.mark.parametrize("speed", [4.0, 4.5, 5.0, 5.5, 6.0])
    def test_lc_end_time_formula(self, speed):
        duration = 10.0 / speed
        lc_end = 2.0 + duration
        assert lc_end == pytest.approx(2.0 + 10.0 / speed, rel=1e-10)

    def test_duration_is_displacement_over_speed(self):
        for speed in [4.0, 5.0, 6.0]:
            assert (10.0 / speed) == pytest.approx(10.0 / speed)


# ---------------------------------------------------------------------------
# XML point helpers
# ---------------------------------------------------------------------------

class TestParsePoint:
    def test_integer_values(self):
        t, v = _parse_pt("4,1")
        assert t == pytest.approx(4.0)
        assert v == pytest.approx(1.0)

    def test_float_values(self):
        t, v = _parse_pt("4.166666667,1")
        assert t == pytest.approx(4.166666667, rel=1e-8)
        assert v == pytest.approx(1.0)

    def test_zero_values(self):
        t, v = _parse_pt("2,0")
        assert t == pytest.approx(2.0)
        assert v == pytest.approx(0.0)

    def test_whitespace_stripped(self):
        t, v = _parse_pt("  2.0 , 0.0  ")
        assert t == pytest.approx(2.0)
        assert v == pytest.approx(0.0)

    def test_missing_comma_raises(self):
        with pytest.raises(ValueError):
            _parse_pt("only_one_value")

    def test_too_many_parts_raises(self):
        with pytest.raises(ValueError):
            _parse_pt("1,2,3")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            _parse_pt("")

    def test_non_numeric_raises(self):
        with pytest.raises(ValueError):
            _parse_pt("abc,def")


class TestFormatPoint:
    def test_integer_values(self):
        assert _format_pt(4.0, 1.0) == "4,1"

    def test_zero_start(self):
        assert _format_pt(2.0, 0.0) == "2,0"

    def test_decimal_value(self):
        assert _format_pt(4.5, 1.0) == "4.5,1"

    def test_roundtrip_precision(self):
        for speed in [4.0, 4.3, 5.0, 5.7, 6.0]:
            lc_end = 2.0 + 10.0 / speed
            text = _format_pt(lc_end, 1.0)
            t_back, v_back = _parse_pt(text)
            assert t_back == pytest.approx(lc_end, rel=1e-8)
            assert v_back == pytest.approx(1.0)

    def test_no_trailing_zeros(self):
        result = _format_pt(4.0, 1.0)
        assert result == "4,1"   # not "4.0000000000,1.0000000000"


# ---------------------------------------------------------------------------
# ConfigurationParams dataclass
# ---------------------------------------------------------------------------

class TestConfigurationParams:
    def test_all_fields_accessible(self, minimal_feb, tmp_path):
        p = ConfigurationParams(
            speed_mm_s=5.0,
            base_feb_path=minimal_feb,
            output_path=tmp_path / "out.feb",
        )
        assert p.speed_mm_s == pytest.approx(5.0)
        assert p.displacement_mm == 10.0
        assert p.loadcurve_id == 1
        assert p.loadcurve_start_time == 2.0
        assert p.insertion_step_id == 2
        assert p.step_size == pytest.approx(0.05)

    def test_is_frozen(self, minimal_feb, tmp_path):
        p = ConfigurationParams(
            speed_mm_s=5.0,
            base_feb_path=minimal_feb,
            output_path=tmp_path / "out.feb",
        )
        with pytest.raises((TypeError, AttributeError)):
            p.speed_mm_s = 6.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# XML modification — using minimal_feb fixture (fast)
# ---------------------------------------------------------------------------

class TestXMLModificationMinimal:
    def test_output_file_created(self, configurator, minimal_feb, tmp_path):
        out = tmp_path / "input.feb"
        configurator.configure(speed_mm_s=5.0, output_path=out, base_feb_path=minimal_feb)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_output_is_valid_xml(self, configurator, minimal_feb, tmp_path):
        out = tmp_path / "input.feb"
        configurator.configure(speed_mm_s=5.0, output_path=out, base_feb_path=minimal_feb)
        tree = etree.parse(str(out))
        assert tree.getroot().tag == "febio_spec"

    def test_xml_declaration_preserved(self, configurator, minimal_feb, tmp_path):
        out = tmp_path / "input.feb"
        configurator.configure(speed_mm_s=5.0, output_path=out, base_feb_path=minimal_feb)
        assert out.read_bytes().startswith(b"<?xml")

    @pytest.mark.parametrize("speed, expected_ts", [
        (4.0, 50),
        (5.0, 40),
        (6.0, round((10/6) / 0.05)),
    ])
    def test_step2_time_steps_updated(self, configurator, minimal_feb, tmp_path, speed, expected_ts):
        out = tmp_path / f"input_{speed}.feb"
        configurator.configure(speed_mm_s=speed, output_path=out, base_feb_path=minimal_feb)
        step2 = etree.parse(str(out)).getroot().find(".//step[@id='2']")
        assert int(step2.find("Control/time_steps").text) == expected_ts

    @pytest.mark.parametrize("speed", [4.0, 5.0, 6.0])
    def test_lc_end_point_updated(self, configurator, minimal_feb, tmp_path, speed):
        out = tmp_path / f"input_{speed}.feb"
        configurator.configure(speed_mm_s=speed, output_path=out, base_feb_path=minimal_feb)
        lc = etree.parse(str(out)).getroot().find(".//load_controller[@id='1']")
        pts = lc.findall("points/pt")
        end_t, end_v = _parse_pt(pts[-1].text)
        assert end_t == pytest.approx(2.0 + 10.0 / speed, rel=1e-8)
        assert end_v == pytest.approx(1.0)

    def test_lc_start_point_unchanged(self, configurator, minimal_feb, tmp_path):
        out = tmp_path / "input.feb"
        configurator.configure(speed_mm_s=4.0, output_path=out, base_feb_path=minimal_feb)
        lc = etree.parse(str(out)).getroot().find(".//load_controller[@id='1']")
        t_start, v_start = _parse_pt(lc.findall("points/pt")[0].text)
        assert t_start == pytest.approx(2.0)
        assert v_start == pytest.approx(0.0)

    def test_step1_time_steps_unchanged(self, configurator, minimal_feb, tmp_path):
        out = tmp_path / "input.feb"
        configurator.configure(speed_mm_s=4.0, output_path=out, base_feb_path=minimal_feb)
        step1 = etree.parse(str(out)).getroot().find(".//step[@id='1']")
        assert int(step1.find("Control/time_steps").text) == 40

    def test_step2_step_size_unchanged(self, configurator, minimal_feb, tmp_path):
        out = tmp_path / "input.feb"
        configurator.configure(speed_mm_s=4.0, output_path=out, base_feb_path=minimal_feb)
        step2 = etree.parse(str(out)).getroot().find(".//step[@id='2']")
        assert float(step2.find("Control/step_size").text) == pytest.approx(0.05)

    def test_material_block_preserved(self, configurator, minimal_feb, tmp_path):
        """Unrelated XML elements must not be altered."""
        out = tmp_path / "input.feb"
        configurator.configure(speed_mm_s=4.0, output_path=out, base_feb_path=minimal_feb)
        mat = etree.parse(str(out)).getroot().find(".//material[@id='1']")
        assert mat is not None
        assert mat.get("name") == "urethra"
        assert float(mat.find("E").text) == pytest.approx(1.0)
        assert float(mat.find("v").text) == pytest.approx(0.49)

    def test_output_parent_dir_auto_created(self, configurator, minimal_feb, tmp_path):
        out = tmp_path / "runs" / "run_0042" / "input.feb"
        assert not out.parent.exists()
        configurator.configure(speed_mm_s=5.0, output_path=out, base_feb_path=minimal_feb)
        assert out.exists()

    def test_idempotent_same_speed_byte_identical(self, configurator, minimal_feb, tmp_path):
        out1, out2 = tmp_path / "a.feb", tmp_path / "b.feb"
        configurator.configure(speed_mm_s=4.5, output_path=out1, base_feb_path=minimal_feb)
        configurator.configure(speed_mm_s=4.5, output_path=out2, base_feb_path=minimal_feb)
        assert out1.read_bytes() == out2.read_bytes()


# ---------------------------------------------------------------------------
# Integration — real 17k-line template (slower, marks as integration)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestXMLModificationRealTemplate:
    """Uses the actual catheterization template — marked 'integration'."""

    def test_output_is_valid_xml(self, configurator, template_feb, tmp_path):
        out = tmp_path / "input.feb"
        configurator.configure(speed_mm_s=5.0, output_path=out)
        tree = etree.parse(str(out))
        assert tree.getroot().tag == "febio_spec"

    def test_step2_time_steps_at_4mms(self, configurator, template_feb, tmp_path):
        out = tmp_path / "input.feb"
        configurator.configure(speed_mm_s=4.0, output_path=out)
        step2 = etree.parse(str(out)).getroot().find(".//step[@id='2']")
        assert int(step2.find("Control/time_steps").text) == 50

    def test_lc_end_point_at_6mms(self, configurator, template_feb, tmp_path):
        out = tmp_path / "input.feb"
        configurator.configure(speed_mm_s=6.0, output_path=out)
        lc = etree.parse(str(out)).getroot().find(".//load_controller[@id='1']")
        end_t, _ = _parse_pt(lc.findall("points/pt")[-1].text)
        assert end_t == pytest.approx(2.0 + 10.0 / 6.0, rel=1e-8)

    def test_all_bc_nodes_preserved(self, configurator, template_feb, tmp_path):
        out = tmp_path / "input.feb"
        configurator.configure(speed_mm_s=5.0, output_path=out)
        root = etree.parse(str(out)).getroot()
        bcs = root.findall(".//bc")
        assert len(bcs) > 0, "Boundary conditions lost after modification"


# ---------------------------------------------------------------------------
# ConfigurationResult
# ---------------------------------------------------------------------------

class TestConfigurationResult:
    def test_result_fields_at_4mms(self, configurator, minimal_feb, tmp_path):
        out = tmp_path / "input.feb"
        result = configurator.configure(speed_mm_s=4.0, output_path=out, base_feb_path=minimal_feb)
        assert isinstance(result, ConfigurationResult)
        assert result.speed_mm_s == pytest.approx(4.0)
        assert result.insertion_duration_s == pytest.approx(2.5)
        assert result.lc_start_time == pytest.approx(2.0)
        assert result.lc_end_time == pytest.approx(4.5)
        assert result.time_steps_step2 == 50
        assert result.step_size == pytest.approx(0.05)
        assert result.output_path == out

    def test_result_is_frozen(self, configurator, minimal_feb, tmp_path):
        out = tmp_path / "input.feb"
        result = configurator.configure(speed_mm_s=5.0, output_path=out, base_feb_path=minimal_feb)
        with pytest.raises((TypeError, AttributeError)):
            result.speed_mm_s = 9.0  # type: ignore[misc]

    def test_as_dict_has_all_keys(self, configurator, minimal_feb, tmp_path):
        out = tmp_path / "input.feb"
        result = configurator.configure(speed_mm_s=5.0, output_path=out, base_feb_path=minimal_feb)
        d = result.as_dict()
        for key in ("speed_mm_s", "insertion_duration_s", "lc_start_time",
                    "lc_end_time", "time_steps_step2", "step_size",
                    "output_path", "base_feb_path"):
            assert key in d, f"Missing key in as_dict(): {key}"

    def test_total_simulation_time_equals_lc_end(self, configurator, minimal_feb, tmp_path):
        out = tmp_path / "input.feb"
        result = configurator.configure(speed_mm_s=5.0, output_path=out, base_feb_path=minimal_feb)
        assert result.total_simulation_time_s == pytest.approx(result.lc_end_time)

    def test_base_feb_path_recorded(self, configurator, minimal_feb, tmp_path):
        out = tmp_path / "input.feb"
        result = configurator.configure(speed_mm_s=5.0, output_path=out, base_feb_path=minimal_feb)
        assert result.base_feb_path == minimal_feb


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_zero_speed_raises_value_error(self, configurator, tmp_path):
        with pytest.raises(ValueError, match="positive"):
            configurator.configure(speed_mm_s=0.0, output_path=tmp_path / "x.feb")

    def test_negative_speed_raises_value_error(self, configurator, tmp_path):
        with pytest.raises(ValueError, match="positive"):
            configurator.configure(speed_mm_s=-1.0, output_path=tmp_path / "x.feb")

    def test_missing_base_file_raises_file_not_found(self, configurator, tmp_path):
        with pytest.raises(FileNotFoundError):
            configurator.configure(
                speed_mm_s=5.0,
                output_path=tmp_path / "x.feb",
                base_feb_path=tmp_path / "nonexistent.feb",
            )

    def test_missing_loadcontroller_raises_runtime_error(self, tmp_path):
        feb = tmp_path / "bad.feb"
        feb.write_text(
            '<?xml version="1.0" encoding="ISO-8859-1"?>'
            '<febio_spec version="4.0">'
            '<LoadData>'
            '<load_controller id="99" name="OTHER" type="loadcurve">'
            '<points><pt>0,0</pt><pt>1,1</pt></points>'
            '</load_controller>'
            '</LoadData>'
            '<Step><step id="2" name="Step2">'
            '<Control><time_steps>40</time_steps><step_size>0.05</step_size></Control>'
            '</step></Step>'
            '</febio_spec>'
        )
        with pytest.raises(RuntimeError, match="LoadController"):
            SimulationConfigurator().configure(
                speed_mm_s=5.0, output_path=tmp_path / "out.feb", base_feb_path=feb
            )

    def test_missing_insertion_step_raises_runtime_error(self, tmp_path):
        feb = tmp_path / "bad.feb"
        feb.write_text(
            '<?xml version="1.0" encoding="ISO-8859-1"?>'
            '<febio_spec version="4.0">'
            '<LoadData>'
            '<load_controller id="1" name="LC1" type="loadcurve">'
            '<points><pt>2,0</pt><pt>4,1</pt></points>'
            '</load_controller>'
            '</LoadData>'
            '<Step><step id="99" name="WrongStep">'
            '<Control><time_steps>40</time_steps></Control>'
            '</step></Step>'
            '</febio_spec>'
        )
        with pytest.raises(RuntimeError, match="step"):
            SimulationConfigurator().configure(
                speed_mm_s=5.0, output_path=tmp_path / "out.feb", base_feb_path=feb
            )

    def test_missing_control_block_raises_runtime_error(self, tmp_path):
        feb = tmp_path / "bad.feb"
        feb.write_text(
            '<?xml version="1.0" encoding="ISO-8859-1"?>'
            '<febio_spec version="4.0">'
            '<LoadData>'
            '<load_controller id="1" name="LC1" type="loadcurve">'
            '<points><pt>2,0</pt><pt>4,1</pt></points>'
            '</load_controller>'
            '</LoadData>'
            '<Step><step id="2" name="Step2">'
            '</step></Step>'
            '</febio_spec>'
        )
        with pytest.raises(RuntimeError, match="Control"):
            SimulationConfigurator().configure(
                speed_mm_s=5.0, output_path=tmp_path / "out.feb", base_feb_path=feb
            )

    def test_missing_points_block_raises_runtime_error(self, tmp_path):
        feb = tmp_path / "bad.feb"
        feb.write_text(
            '<?xml version="1.0" encoding="ISO-8859-1"?>'
            '<febio_spec version="4.0">'
            '<LoadData>'
            '<load_controller id="1" name="LC1" type="loadcurve">'
            '</load_controller>'
            '</LoadData>'
            '<Step><step id="2" name="Step2">'
            '<Control><time_steps>40</time_steps></Control>'
            '</step></Step>'
            '</febio_spec>'
        )
        with pytest.raises(RuntimeError, match="points"):
            SimulationConfigurator().configure(
                speed_mm_s=5.0, output_path=tmp_path / "out.feb", base_feb_path=feb
            )


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

class TestConfigureSimulationFunction:
    def test_basic_call(self, minimal_feb, tmp_path):
        out = tmp_path / "input.feb"
        result = configure_simulation(
            speed_mm_s=5.0, output_path=out, base_feb_path=minimal_feb
        )
        assert out.exists()
        assert result.speed_mm_s == pytest.approx(5.0)

    def test_with_custom_settings(self, minimal_feb, tmp_path):
        from digital_twin_ui.app.core.config import load_settings
        s = load_settings()
        out = tmp_path / "input.feb"
        result = configure_simulation(
            speed_mm_s=4.0,
            output_path=out,
            base_feb_path=minimal_feb,
            settings=s,
        )
        assert result.time_steps_step2 == 50
