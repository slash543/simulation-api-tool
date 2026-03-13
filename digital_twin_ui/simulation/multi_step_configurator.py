"""
Multi-Step Configurator
=======================
Modifies a multi-step FEB file (10 load controllers / 10 steps) to apply
per-step insertion speeds and a uniform dwell time.

Physics
-------
For each step i (0-indexed):

    ramp_i        = displacement_mm[i] / speed_mm_s[i]
    step_duration = ramp_i + dwell_time_s
    time_steps_i  = max(10, ceil(step_duration / base_step_size))

Load controllers are rebuilt sequentially:
    LC[i].t_start = LC[i-1].t_end
    LC[i] points  = (t_start, 0), (t_start + ramp, 1), (t_start + ramp + dwell, 1)

The last LC (LC10) preserves FEBio's 4-point pattern:
    (t_start, 0), (t_start, 0), (t_start + ramp, 1), (t_start + ramp + dwell, 1)

All other XML content, whitespace, comments, and encoding are preserved.

Usage
-----
    from digital_twin_ui.simulation.multi_step_configurator import MultiStepConfigurator
    from digital_twin_ui.simulation.template_registry import TemplateRegistry

    registry = TemplateRegistry()
    tc = registry.get("DT_BT_14Fr_FO_10E_IR12")

    cfg = MultiStepConfigurator(tc)
    result = cfg.configure(
        speeds_mm_s=[15.0] * 10,
        dwell_time_s=1.0,
        output_path=Path("runs/run_0001/input.feb"),
    )
    print(result.total_duration_s)
    print(result.step_durations)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lxml import etree

from digital_twin_ui.app.core.logging import get_logger
from digital_twin_ui.simulation.feb_reader import validate_feb_steps
from digital_twin_ui.simulation.template_registry import TemplateConfig

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class MultiStepConfigResult:
    """
    Immutable record of what was written to the output .feb file.

    Attributes:
        speeds_mm_s:       Per-step insertion speeds used.
        dwell_time_s:      Dwell time appended after each ramp.
        template_name:     Name of the source template.
        total_duration_s:  Wall-clock end time of the last LC.
        step_durations:    Duration (ramp + dwell) for each step in seconds.
        lc_timing:         List of dicts with t_start, ramp_s, dwell_s,
                           t_ramp_end, t_end for each step (in step order).
        output_path:       Destination file that was written.
    """

    speeds_mm_s: list[float]
    dwell_time_s: float
    template_name: str
    total_duration_s: float
    step_durations: list[float]
    lc_timing: list[dict[str, float]]
    output_path: Path

    def as_dict(self) -> dict[str, Any]:
        return {
            "speeds_mm_s": self.speeds_mm_s,
            "dwell_time_s": self.dwell_time_s,
            "template_name": self.template_name,
            "total_duration_s": self.total_duration_s,
            "step_durations": self.step_durations,
            "lc_timing": self.lc_timing,
            "output_path": str(self.output_path),
        }


# ---------------------------------------------------------------------------
# Configurator
# ---------------------------------------------------------------------------


class MultiStepConfigurator:
    """
    Reads a multi-step .feb template and produces a speed-modified copy.

    The configurator is stateless per run — a single instance can be reused
    across many DOE samples.

    Args:
        template: A :class:`~digital_twin_ui.simulation.template_registry.TemplateConfig`
                  describing the FEB template to use.
    """

    def __init__(self, template: TemplateConfig) -> None:
        self._template = template

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def configure(
        self,
        speeds_mm_s: list[float],
        dwell_time_s: float,
        output_path: Path,
    ) -> MultiStepConfigResult:
        """
        Generate a modified .feb file with per-step speeds applied.

        Args:
            speeds_mm_s:  One speed per step (length must equal template.n_steps).
            dwell_time_s: Dwell time in seconds appended after each ramp.
            output_path:  Destination path for the modified file.

        Returns:
            :class:`MultiStepConfigResult`

        Raises:
            ValueError:      If len(speeds_mm_s) != template.n_steps or any
                             speed is non-positive.
            FileNotFoundError: If the template .feb file does not exist.
            RuntimeError:    If required XML elements are missing.
        """
        n = self._template.n_steps
        if len(speeds_mm_s) != n:
            raise ValueError(
                f"Template '{self._template.name}' requires {n} speeds, "
                f"got {len(speeds_mm_s)}"
            )
        for i, s in enumerate(speeds_mm_s):
            if s <= 0:
                raise ValueError(
                    f"speeds_mm_s[{i}] must be positive, got {s}"
                )

        feb_path = self._template.feb_path
        if not feb_path.exists():
            raise FileNotFoundError(
                f"Template .feb file not found: {feb_path}"
            )

        # Validate that the file has exactly n_steps steps before modifying it.
        # Each <step> in the FEB XML is one analysis step (also called an event).
        validate_feb_steps(feb_path, required_steps=n)

        logger.info(
            "MultiStepConfigurator: configuring '{name}' with {n} speeds",
            name=self._template.name,
            n=n,
        )

        result = self._apply(speeds_mm_s, dwell_time_s, output_path, feb_path)

        logger.info(
            "MultiStepConfigurator: done — total_duration={dur:.2f}s",
            dur=result.total_duration_s,
        )
        return result

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    def _apply(
        self,
        speeds_mm_s: list[float],
        dwell_time_s: float,
        output_path: Path,
        feb_path: Path,
    ) -> MultiStepConfigResult:
        """Parse, modify, and write the FEB file."""
        displacements = self._template.displacements_mm
        base_step_size = self._template.base_step_size

        # --- Parse XML ---
        parser = etree.XMLParser(remove_blank_text=False, remove_comments=False)
        tree = etree.parse(str(feb_path), parser)
        root = tree.getroot()

        # --- Extract step → LC id mapping ---
        step_lc_ids = self._extract_step_lc_mapping(root)
        if len(step_lc_ids) != self._template.n_steps:
            raise RuntimeError(
                f"Expected {self._template.n_steps} steps with LC references, "
                f"found {len(step_lc_ids)} in {feb_path}"
            )

        # --- Sort LCs chronologically by their current first point time ---
        lc_elements = self._collect_lc_elements(root)
        lc_order = self._sort_lcs_chronologically(lc_elements)

        # Determine which LC is the "last" one (gets 4-point pattern)
        last_lc_id = lc_order[-1]

        # --- Compute timing ---
        step_ramps: list[float] = []
        step_durations_list: list[float] = []
        time_steps_list: list[int] = []
        for i in range(self._template.n_steps):
            ramp = displacements[i] / speeds_mm_s[i]
            duration = ramp + dwell_time_s
            ts = max(10, math.ceil(duration / base_step_size))
            step_ramps.append(ramp)
            step_durations_list.append(duration)
            time_steps_list.append(ts)

        # --- Rebuild LC time points sequentially ---
        lc_timing: list[dict[str, float]] = []
        t_cursor = self._get_lc_first_time(lc_elements[lc_order[0]])

        for step_idx, lc_id in enumerate(lc_order):
            ramp = step_ramps[step_idx]
            dwell = dwell_time_s
            t_start = t_cursor
            t_ramp_end = t_start + ramp
            t_end = t_ramp_end + dwell

            is_last = (lc_id == last_lc_id)
            self._rebuild_lc(
                lc_elements[lc_id],
                t_start=t_start,
                t_ramp_end=t_ramp_end,
                t_end=t_end,
                is_last=is_last,
            )

            lc_timing.append({
                "lc_id": lc_id,
                "step_index": step_idx,
                "t_start": t_start,
                "ramp_s": ramp,
                "dwell_s": dwell,
                "t_ramp_end": t_ramp_end,
                "t_end": t_end,
            })

            t_cursor = t_end

        total_duration = t_cursor

        # --- Update time_steps in each Step's Control block ---
        step_elements = self._collect_step_elements(root)
        for step_idx, lc_id in enumerate(step_lc_ids):
            # step_lc_ids[i] = LC id used by step i (in step order)
            # We need the position of this lc_id in the chronological order
            # to find the right time_steps value
            if lc_id in lc_order:
                chron_idx = lc_order.index(lc_id)
                ts = time_steps_list[chron_idx]
            else:
                ts = time_steps_list[step_idx]
            step_id = step_idx + 1  # step IDs are 1-based
            self._update_step_time_steps(step_elements, step_id, ts)

        # --- Write output ---
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tree.write(
            str(output_path),
            xml_declaration=True,
            encoding="ISO-8859-1",
            pretty_print=True,
        )

        return MultiStepConfigResult(
            speeds_mm_s=list(speeds_mm_s),
            dwell_time_s=dwell_time_s,
            template_name=self._template.name,
            total_duration_s=total_duration,
            step_durations=step_durations_list,
            lc_timing=lc_timing,
            output_path=output_path,
        )

    # ------------------------------------------------------------------
    # XML helpers
    # ------------------------------------------------------------------

    def _extract_step_lc_mapping(self, root: etree._Element) -> list[int]:
        """
        Parse each <Step> in document order and extract the LC id used by
        its <Boundary> prescribed displacement.

        Returns a list of LC ids, one per step (in step document order).

        In some files (e.g. IR12) the LC ids do NOT match step order, so we
        must read ``<value lc="N">`` inside each Step's Boundary block.
        """
        step_lc_ids: list[int] = []
        steps = root.findall(".//step")
        for step_elem in steps:
            # Look for any <value lc="N"> inside this step
            value_elems = step_elem.findall(".//value[@lc]")
            if value_elems:
                # Take the first one (all BCs in the step should use the same LC)
                lc_id = int(value_elems[0].get("lc"))
                step_lc_ids.append(lc_id)
            else:
                # Fallback: try to infer from prescribe elements
                prescribe_elems = step_elem.findall(".//prescribe")
                if prescribe_elems:
                    for pe in prescribe_elems:
                        val = pe.find("value")
                        if val is not None and val.get("lc"):
                            step_lc_ids.append(int(val.get("lc")))
                            break

        return step_lc_ids

    def _collect_lc_elements(
        self, root: etree._Element
    ) -> dict[int, etree._Element]:
        """Return {lc_id: element} for all load_controller elements."""
        lcs: dict[int, etree._Element] = {}
        for elem in root.findall(".//load_controller"):
            id_str = elem.get("id")
            if id_str is not None:
                lcs[int(id_str)] = elem
        return lcs

    def _collect_step_elements(
        self, root: etree._Element
    ) -> dict[int, etree._Element]:
        """Return {step_id: element} for all step elements."""
        steps: dict[int, etree._Element] = {}
        for elem in root.findall(".//step"):
            id_str = elem.get("id")
            if id_str is not None:
                steps[int(id_str)] = elem
        return steps

    @staticmethod
    def _get_lc_first_time(lc_elem: etree._Element) -> float:
        """Extract the time value of the first <pt> in a load_controller."""
        points_elem = lc_elem.find("points")
        if points_elem is None:
            return 0.0
        pts = points_elem.findall("pt")
        if not pts:
            return 0.0
        text = (pts[0].text or "").strip()
        parts = text.split(",")
        if len(parts) >= 1:
            try:
                return float(parts[0].strip())
            except ValueError:
                return 0.0
        return 0.0

    def _sort_lcs_chronologically(
        self, lc_elements: dict[int, etree._Element]
    ) -> list[int]:
        """
        Sort LC ids by the time value of their first <pt> element (ascending).
        This gives a reliable chronological order regardless of id numbering.
        """
        timed = [
            (self._get_lc_first_time(elem), lc_id)
            for lc_id, elem in lc_elements.items()
        ]
        timed.sort(key=lambda x: (x[0], x[1]))
        return [lc_id for _, lc_id in timed]

    @staticmethod
    def _rebuild_lc(
        lc_elem: etree._Element,
        t_start: float,
        t_ramp_end: float,
        t_end: float,
        is_last: bool,
    ) -> None:
        """
        Replace all <pt> elements inside the load controller's <points> block.

        Standard pattern (3 points):
            (t_start, 0), (t_ramp_end, 1), (t_end, 1)

        Last-LC pattern (4 points, matches original FEB structure):
            (t_start, 0), (t_start, 0), (t_ramp_end, 1), (t_end, 1)
        """
        points_elem = lc_elem.find("points")
        if points_elem is None:
            raise RuntimeError(
                f"<points> missing in load_controller id='{lc_elem.get('id')}'"
            )

        # Remove existing <pt> elements
        for pt in points_elem.findall("pt"):
            points_elem.remove(pt)

        def _fmt(v: float) -> str:
            return f"{v:.10g}"

        def _make_pt(t: float, val: float) -> etree._Element:
            pt = etree.SubElement(points_elem, "pt")
            pt.text = f"{_fmt(t)},{_fmt(val)}"
            return pt

        if is_last:
            _make_pt(t_start, 0.0)
            _make_pt(t_start, 0.0)
            _make_pt(t_ramp_end, 1.0)
            _make_pt(t_end, 1.0)
        else:
            _make_pt(t_start, 0.0)
            _make_pt(t_ramp_end, 1.0)
            _make_pt(t_end, 1.0)

    @staticmethod
    def _update_step_time_steps(
        step_elements: dict[int, etree._Element],
        step_id: int,
        time_steps: int,
    ) -> None:
        """Update <time_steps> inside the given step's <Control> block."""
        step_elem = step_elements.get(step_id)
        if step_elem is None:
            raise RuntimeError(
                f"<Step id='{step_id}'> not found in FEB file"
            )

        control_elem = step_elem.find("Control")
        if control_elem is None:
            raise RuntimeError(
                f"<Control> block missing in Step id='{step_id}'"
            )

        ts_elem = control_elem.find("time_steps")
        if ts_elem is None:
            raise RuntimeError(
                f"<time_steps> missing in Step id='{step_id}' Control"
            )

        old = ts_elem.text
        ts_elem.text = str(time_steps)

        logger.debug(
            "Step id={id}: time_steps {old} → {new}",
            id=step_id,
            old=old,
            new=time_steps,
        )
