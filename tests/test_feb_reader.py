"""
Tests for FEB File Reader (simulation/feb_reader.py).

Strategy
--------
- Minimal synthetic FEB XML strings are used (fast, no real file I/O dependency).
- One fixture generates a valid 10-step FEB file to mirror the catheter templates.
- Edge cases: missing file, non-XML content, wrong step count, no load controllers.

Coverage
--------
- read_feb() returns correct FebInfo for valid files
- step count, step ids, step names, lc_ids, time_steps populated correctly
- load controller count and ids populated correctly
- version string extracted
- FileNotFoundError on missing file
- ValueError on invalid XML
- validate_feb_steps() passes when step count matches
- validate_feb_steps() raises ValueError when count is wrong
- as_dict() is JSON-serialisable
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

try:
    from lxml import etree  # noqa: F401
    _LXML_AVAILABLE = True
except ImportError:
    _LXML_AVAILABLE = False

from digital_twin_ui.simulation.feb_reader import (
    FebInfo,
    FebStepInfo,
    read_feb,
    validate_feb_steps,
)

pytestmark = pytest.mark.skipif(
    not _LXML_AVAILABLE, reason="lxml not installed"
)


# ---------------------------------------------------------------------------
# Helpers to build synthetic FEB XML
# ---------------------------------------------------------------------------

def _make_steps_xml(n: int, start_lc: int = 1) -> str:
    """Return <Step> wrapper containing n <step id=i> elements with LCs."""
    inner = ""
    for i in range(1, n + 1):
        lc = start_lc + i - 1
        inner += f"""
    <step id="{i}" name="Step{i}">
      <Control>
        <time_steps>40</time_steps>
        <step_size>0.1</step_size>
      </Control>
      <Boundary>
        <bc name="insert{i}" type="prescribed displacement">
          <value lc="{lc}">10</value>
        </bc>
      </Boundary>
    </step>"""
    return f"  <Step>{inner}\n  </Step>"


def _make_lcs_xml(n: int) -> str:
    """Return <LoadData> wrapper with n load_controller elements."""
    inner = ""
    for i in range(1, n + 1):
        t_start = (i - 1) * 2
        t_end = i * 2
        inner += f"""
    <load_controller id="{i}" name="LC{i}" type="loadcurve">
      <interpolate>SMOOTH STEP</interpolate>
      <points>
        <pt>{t_start},0</pt>
        <pt>{t_end},1</pt>
        <pt>{t_end + 1},1</pt>
      </points>
    </load_controller>"""
    return f"  <LoadData>{inner}\n  </LoadData>"


def _make_feb(n_steps: int, n_lcs: int | None = None, version: str = "4.0") -> str:
    """Build a complete minimal FEB XML string with n_steps steps."""
    if n_lcs is None:
        n_lcs = n_steps
    return (
        f'<?xml version="1.0" encoding="ISO-8859-1"?>\n'
        f'<febio_spec version="{version}">\n'
        f"{_make_steps_xml(n_steps)}\n"
        f"{_make_lcs_xml(n_lcs)}\n"
        f"</febio_spec>\n"
    )


def _write_feb(tmp_path: Path, n_steps: int, n_lcs: int | None = None,
               filename: str = "test.feb", version: str = "4.0") -> Path:
    """Write a synthetic FEB file and return its path."""
    path = tmp_path / filename
    path.write_text(_make_feb(n_steps, n_lcs, version=version), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# read_feb — basic functionality
# ---------------------------------------------------------------------------


class TestReadFeb:
    def test_returns_feb_info(self, tmp_path):
        path = _write_feb(tmp_path, n_steps=3)
        info = read_feb(path)
        assert isinstance(info, FebInfo)

    def test_step_count(self, tmp_path):
        path = _write_feb(tmp_path, n_steps=5)
        info = read_feb(path)
        assert info.n_steps == 5

    def test_step_count_10(self, tmp_path):
        """Primary case: catheter templates have 10 steps."""
        path = _write_feb(tmp_path, n_steps=10)
        info = read_feb(path)
        assert info.n_steps == 10

    def test_step_ids(self, tmp_path):
        path = _write_feb(tmp_path, n_steps=3)
        info = read_feb(path)
        ids = [s.step_id for s in info.steps]
        assert ids == [1, 2, 3]

    def test_step_names(self, tmp_path):
        path = _write_feb(tmp_path, n_steps=3)
        info = read_feb(path)
        names = [s.name for s in info.steps]
        assert names == ["Step1", "Step2", "Step3"]

    def test_step_lc_ids(self, tmp_path):
        path = _write_feb(tmp_path, n_steps=3)
        info = read_feb(path)
        # Each step i references LC i
        assert info.steps[0].lc_ids == [1]
        assert info.steps[1].lc_ids == [2]
        assert info.steps[2].lc_ids == [3]

    def test_step_time_steps(self, tmp_path):
        path = _write_feb(tmp_path, n_steps=3)
        info = read_feb(path)
        for s in info.steps:
            assert s.time_steps == 40

    def test_lc_count(self, tmp_path):
        path = _write_feb(tmp_path, n_steps=3, n_lcs=3)
        info = read_feb(path)
        assert info.n_load_controllers == 3

    def test_lc_ids_list(self, tmp_path):
        path = _write_feb(tmp_path, n_steps=3, n_lcs=3)
        info = read_feb(path)
        assert info.lc_ids == [1, 2, 3]

    def test_version_extracted(self, tmp_path):
        path = _write_feb(tmp_path, n_steps=2, version="4.0")
        info = read_feb(path)
        assert info.version == "4.0"

    def test_path_is_resolved(self, tmp_path):
        path = _write_feb(tmp_path, n_steps=2)
        info = read_feb(path)
        assert info.path == path.resolve()

    def test_accepts_str_path(self, tmp_path):
        path = _write_feb(tmp_path, n_steps=2)
        info = read_feb(str(path))
        assert info.n_steps == 2

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_feb(tmp_path / "nonexistent.feb")

    def test_invalid_xml_raises_value_error(self, tmp_path):
        bad = tmp_path / "bad.feb"
        bad.write_text("not xml at all <<<", encoding="utf-8")
        with pytest.raises(ValueError, match="Cannot parse"):
            read_feb(bad)

    def test_empty_file_raises_value_error(self, tmp_path):
        empty = tmp_path / "empty.feb"
        empty.write_text("", encoding="utf-8")
        with pytest.raises(ValueError):
            read_feb(empty)

    def test_zero_steps(self, tmp_path):
        """A file with no <step> elements returns n_steps=0."""
        xml = (
            '<?xml version="1.0" encoding="ISO-8859-1"?>\n'
            '<febio_spec version="4.0">\n'
            '  <LoadData>\n'
            '    <load_controller id="1" type="loadcurve"><points><pt>0,0</pt></points></load_controller>\n'
            '  </LoadData>\n'
            '</febio_spec>\n'
        )
        path = tmp_path / "nosteps.feb"
        path.write_text(xml, encoding="utf-8")
        info = read_feb(path)
        assert info.n_steps == 0
        assert info.n_load_controllers == 1

    def test_no_load_controllers(self, tmp_path):
        """A file with no load controllers has n_load_controllers=0."""
        xml = (
            '<?xml version="1.0" encoding="ISO-8859-1"?>\n'
            '<febio_spec version="4.0">\n'
            '  <Step>\n'
            '    <step id="1"><Control><time_steps>10</time_steps></Control></step>\n'
            '  </Step>\n'
            '</febio_spec>\n'
        )
        path = tmp_path / "nolc.feb"
        path.write_text(xml, encoding="utf-8")
        info = read_feb(path)
        assert info.n_steps == 1
        assert info.n_load_controllers == 0
        assert info.lc_ids == []

    def test_step_without_name_attribute(self, tmp_path):
        xml = (
            '<?xml version="1.0" encoding="ISO-8859-1"?>\n'
            '<febio_spec version="4.0">\n'
            '  <Step>\n'
            '    <step id="1"><Control><time_steps>10</time_steps></Control></step>\n'
            '  </Step>\n'
            '</febio_spec>\n'
        )
        path = tmp_path / "noname.feb"
        path.write_text(xml, encoding="utf-8")
        info = read_feb(path)
        assert info.steps[0].name is None

    def test_step_without_time_steps(self, tmp_path):
        xml = (
            '<?xml version="1.0" encoding="ISO-8859-1"?>\n'
            '<febio_spec version="4.0">\n'
            '  <Step>\n'
            '    <step id="1" name="S1"><Control></Control></step>\n'
            '  </Step>\n'
            '</febio_spec>\n'
        )
        path = tmp_path / "nots.feb"
        path.write_text(xml, encoding="utf-8")
        info = read_feb(path)
        assert info.steps[0].time_steps is None


# ---------------------------------------------------------------------------
# read_feb — as_dict
# ---------------------------------------------------------------------------


class TestFebInfoAsDict:
    def test_as_dict_is_json_serialisable(self, tmp_path):
        path = _write_feb(tmp_path, n_steps=3)
        info = read_feb(path)
        d = info.as_dict()
        serialised = json.dumps(d)
        assert isinstance(serialised, str)

    def test_as_dict_keys(self, tmp_path):
        path = _write_feb(tmp_path, n_steps=2)
        d = read_feb(path).as_dict()
        assert "path" in d
        assert "version" in d
        assert "n_steps" in d
        assert "steps" in d
        assert "n_load_controllers" in d
        assert "lc_ids" in d

    def test_as_dict_step_keys(self, tmp_path):
        path = _write_feb(tmp_path, n_steps=1)
        d = read_feb(path).as_dict()
        step = d["steps"][0]
        assert set(step.keys()) == {"step_id", "name", "lc_ids", "time_steps"}


# ---------------------------------------------------------------------------
# validate_feb_steps — passing cases
# ---------------------------------------------------------------------------


class TestValidateFebStepsPass:
    def test_returns_feb_info(self, tmp_path):
        path = _write_feb(tmp_path, n_steps=10)
        result = validate_feb_steps(path, required_steps=10)
        assert isinstance(result, FebInfo)
        assert result.n_steps == 10

    def test_exact_match_3(self, tmp_path):
        path = _write_feb(tmp_path, n_steps=3)
        validate_feb_steps(path, required_steps=3)  # no exception

    def test_exact_match_10(self, tmp_path):
        """Primary guard for catheter simulations."""
        path = _write_feb(tmp_path, n_steps=10)
        validate_feb_steps(path, required_steps=10)  # no exception

    def test_accepts_str_path(self, tmp_path):
        path = _write_feb(tmp_path, n_steps=10)
        validate_feb_steps(str(path), required_steps=10)  # no exception

    def test_returns_same_info_as_read_feb(self, tmp_path):
        path = _write_feb(tmp_path, n_steps=5)
        info_direct = read_feb(path)
        info_via_validate = validate_feb_steps(path, required_steps=5)
        assert info_direct.n_steps == info_via_validate.n_steps
        assert info_direct.lc_ids == info_via_validate.lc_ids


# ---------------------------------------------------------------------------
# validate_feb_steps — failure cases
# ---------------------------------------------------------------------------


class TestValidateFebStepsFail:
    def test_wrong_count_too_few(self, tmp_path):
        path = _write_feb(tmp_path, n_steps=5)
        with pytest.raises(ValueError, match="expected 10 step"):
            validate_feb_steps(path, required_steps=10)

    def test_wrong_count_too_many(self, tmp_path):
        path = _write_feb(tmp_path, n_steps=15)
        with pytest.raises(ValueError, match="expected 10 step"):
            validate_feb_steps(path, required_steps=10)

    def test_error_message_contains_actual_count(self, tmp_path):
        path = _write_feb(tmp_path, n_steps=2)
        with pytest.raises(ValueError, match="found 2"):
            validate_feb_steps(path, required_steps=10)

    def test_error_message_contains_file_path(self, tmp_path):
        path = _write_feb(tmp_path, n_steps=2)
        with pytest.raises(ValueError, match=str(path.resolve())):
            validate_feb_steps(path, required_steps=10)

    def test_zero_steps_vs_required_10(self, tmp_path):
        xml = (
            '<?xml version="1.0" encoding="ISO-8859-1"?>\n'
            '<febio_spec version="4.0"></febio_spec>\n'
        )
        path = tmp_path / "empty.feb"
        path.write_text(xml, encoding="utf-8")
        with pytest.raises(ValueError, match="found 0"):
            validate_feb_steps(path, required_steps=10)

    def test_missing_file_raises_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            validate_feb_steps(tmp_path / "missing.feb", required_steps=10)

    def test_required_steps_1_with_10_step_file(self, tmp_path):
        """Guard also catches files that are too big for the expected template."""
        path = _write_feb(tmp_path, n_steps=10)
        with pytest.raises(ValueError, match="expected 1 step"):
            validate_feb_steps(path, required_steps=1)
