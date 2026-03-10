"""
Simulation Configurator
=======================
Modifies a simulation input file (.feb) to apply a new insertion speed.

Physics
-------
The insertion step (Step 2) prescribes a fixed displacement of D mm over a
load curve that spans from t_start to t_end.  Speed is therefore:

    speed = D / (t_end - t_start)
    → t_end = t_start + D / speed

Two XML locations are updated:
  1. LoadController LC<id> — end-point time coordinate of the load curve
  2. Step <insertion_step_id> — time_steps count  (step_size kept fixed)

All other content, whitespace, comments and encoding are preserved exactly.

Usage
-----
    from digital_twin_ui.simulation.simulation_configurator import (
        SimulationConfigurator, ConfigurationParams, ConfigurationResult
    )

    configurator = SimulationConfigurator(settings)
    result = configurator.configure(
        speed_mm_s=4.5,
        output_path=Path("runs/run_0001/input.feb"),
    )
    print(result.time_steps_step2)   # 44
    print(result.lc_end_time)        # 4.222...
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lxml import etree

from digital_twin_ui.app.core.config import Settings, get_settings
from digital_twin_ui.app.core.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConfigurationParams:
    """Input parameters for a single configuration operation."""

    speed_mm_s: float
    """Insertion speed in mm/s. Must be in (0, ∞)."""

    base_feb_path: Path
    """Source template .feb file to read from."""

    output_path: Path
    """Destination path for the modified .feb file."""

    displacement_mm: float = 10.0
    """Total prescribed displacement magnitude (mm).  Matches FEB value attribute."""

    loadcurve_id: int = 1
    """id attribute of the LoadController element to modify."""

    loadcurve_start_time: float = 2.0
    """Global time at which the load curve (and insertion) begins."""

    insertion_step_id: int = 2
    """id attribute of the <step> element whose time_steps will be updated."""

    step_size: float = 0.05
    """Solver step size (s) — kept fixed; only time_steps count changes."""


@dataclass(frozen=True)
class ConfigurationResult:
    """Immutable record of what was written to the output .feb file."""

    speed_mm_s: float
    insertion_duration_s: float
    lc_start_time: float
    lc_end_time: float
    time_steps_step2: int
    step_size: float
    output_path: Path
    base_feb_path: Path

    # Derived
    @property
    def total_simulation_time_s(self) -> float:
        """Approximate total simulation time (Step 1 + Step 2)."""
        return self.lc_end_time  # LC start is end of Step1

    def as_dict(self) -> dict[str, Any]:
        return {
            "speed_mm_s": self.speed_mm_s,
            "insertion_duration_s": self.insertion_duration_s,
            "lc_start_time": self.lc_start_time,
            "lc_end_time": self.lc_end_time,
            "time_steps_step2": self.time_steps_step2,
            "step_size": self.step_size,
            "output_path": str(self.output_path),
            "base_feb_path": str(self.base_feb_path),
        }


# ---------------------------------------------------------------------------
# Core configurator
# ---------------------------------------------------------------------------

class SimulationConfigurator:
    """
    Reads a base .feb file and produces a speed-modified copy.

    The configurator is stateless with respect to individual runs — a single
    instance can be reused for the entire DOE campaign.

    Args:
        settings: Application settings.  If None, the cached singleton is used.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._cfg = settings or get_settings()
        self._sim = self._cfg.simulation

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def configure(
        self,
        speed_mm_s: float,
        output_path: Path,
        base_feb_path: Path | None = None,
    ) -> ConfigurationResult:
        """
        Generate a modified .feb file for the given insertion speed.

        Args:
            speed_mm_s:    Target insertion speed (mm/s).
            output_path:   Where to write the modified file.
            base_feb_path: Override for the base template.  Defaults to
                           settings.base_feb_path_abs.

        Returns:
            ConfigurationResult with all modified values for audit / logging.

        Raises:
            ValueError:  If speed_mm_s ≤ 0.
            FileNotFoundError: If the base .feb file does not exist.
            RuntimeError: If required XML elements are missing in the template.
        """
        if speed_mm_s <= 0:
            raise ValueError(f"speed_mm_s must be positive, got {speed_mm_s}")

        source = Path(base_feb_path) if base_feb_path else self._cfg.base_feb_path_abs
        if not source.exists():
            raise FileNotFoundError(f"Base simulation file not found: {source}")

        params = ConfigurationParams(
            speed_mm_s=speed_mm_s,
            base_feb_path=source,
            output_path=output_path,
            displacement_mm=self._sim.displacement_mm,
            loadcurve_id=self._sim.loadcurve_id,
            loadcurve_start_time=self._sim.loadcurve_start_time,
            insertion_step_id=self._sim.insertion_step_id,
            step_size=self._sim.default_step_size,
        )

        logger.info(
            "Configuring simulation file for speed={speed} mm/s",
            speed=speed_mm_s,
            source=str(source),
            output=str(output_path),
        )

        result = self._apply(params)

        logger.info(
            "Configuration complete: duration={dur:.4f}s, "
            "lc_end={lc_end:.4f}, time_steps={ts}",
            dur=result.insertion_duration_s,
            lc_end=result.lc_end_time,
            ts=result.time_steps_step2,
        )
        return result

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    def _apply(self, params: ConfigurationParams) -> ConfigurationResult:
        """Parse, modify and write the .feb file.  Returns ConfigurationResult."""

        # --- Compute target values ---
        insertion_duration = params.displacement_mm / params.speed_mm_s
        lc_end_time = params.loadcurve_start_time + insertion_duration
        time_steps = self._compute_time_steps(insertion_duration, params.step_size)

        # --- Parse XML (preserve whitespace, encoding, comments) ---
        parser = etree.XMLParser(remove_blank_text=False, remove_comments=False)
        tree = etree.parse(str(params.base_feb_path), parser)
        root = tree.getroot()

        # --- Apply modifications ---
        self._update_loadcurve(root, params, lc_end_time)
        self._update_step_time_steps(root, params, time_steps)

        # --- Write output ---
        output_path = Path(params.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tree.write(
            str(output_path),
            xml_declaration=True,
            encoding="ISO-8859-1",
            pretty_print=False,   # preserve original whitespace exactly
        )

        return ConfigurationResult(
            speed_mm_s=params.speed_mm_s,
            insertion_duration_s=insertion_duration,
            lc_start_time=params.loadcurve_start_time,
            lc_end_time=lc_end_time,
            time_steps_step2=time_steps,
            step_size=params.step_size,
            output_path=output_path,
            base_feb_path=params.base_feb_path,
        )

    # ------------------------------------------------------------------
    # XML helpers
    # ------------------------------------------------------------------

    def _update_loadcurve(
        self,
        root: etree._Element,
        params: ConfigurationParams,
        lc_end_time: float,
    ) -> None:
        """
        Update the end-point time coordinate of the target LoadController.

        The load curve has exactly two <pt> entries:
            <pt>t_start, 0</pt>   ← left alone
            <pt>t_end,   1</pt>   ← updated to new lc_end_time
        """
        lc_id = str(params.loadcurve_id)
        lc_elem = root.find(f".//load_controller[@id='{lc_id}']")
        if lc_elem is None:
            raise RuntimeError(
                f"LoadController id='{lc_id}' not found in {params.base_feb_path}"
            )

        points_elem = lc_elem.find("points")
        if points_elem is None:
            raise RuntimeError(
                f"<points> element missing inside load_controller id='{lc_id}'"
            )

        pt_list = points_elem.findall("pt")
        if len(pt_list) < 2:
            raise RuntimeError(
                f"Expected ≥2 <pt> elements in load_controller id='{lc_id}', "
                f"found {len(pt_list)}"
            )

        # The last <pt> is the end point: "t_end,1"
        end_pt = pt_list[-1]
        old_text = end_pt.text or ""
        _, value_part = _parse_pt(old_text)

        new_text = _format_pt(lc_end_time, value_part)
        end_pt.text = new_text

        logger.debug(
            "LoadController id={id}: pt updated '{old}' → '{new}'",
            id=lc_id,
            old=old_text.strip(),
            new=new_text,
        )

    def _update_step_time_steps(
        self,
        root: etree._Element,
        params: ConfigurationParams,
        time_steps: int,
    ) -> None:
        """
        Update <time_steps> inside the insertion step's <Control> block.
        """
        step_id = str(params.insertion_step_id)
        step_elem = root.find(f".//step[@id='{step_id}']")
        if step_elem is None:
            raise RuntimeError(
                f"<step id='{step_id}'> not found in {params.base_feb_path}"
            )

        control_elem = step_elem.find("Control")
        if control_elem is None:
            raise RuntimeError(
                f"<Control> block missing inside step id='{step_id}'"
            )

        ts_elem = control_elem.find("time_steps")
        if ts_elem is None:
            raise RuntimeError(
                f"<time_steps> missing inside step id='{step_id}' Control"
            )

        old_val = ts_elem.text
        ts_elem.text = str(time_steps)

        logger.debug(
            "Step id={id}: time_steps updated {old} → {new}",
            id=step_id,
            old=old_val,
            new=time_steps,
        )

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_time_steps(insertion_duration: float, step_size: float) -> int:
        """
        Compute the integer time_steps count that covers insertion_duration.

        Rounds to nearest integer; minimum of 1 to avoid invalid FEB files.
        """
        return max(1, round(insertion_duration / step_size))


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _parse_pt(text: str) -> tuple[float, float]:
    """
    Parse a loadcurve point string like "4,1" or " 4.5 , 1 ".

    Returns:
        (time_value, scale_value) as floats.

    Raises:
        ValueError: If the text cannot be parsed as two comma-separated floats.
    """
    text = text.strip()
    parts = text.split(",")
    if len(parts) != 2:
        raise ValueError(
            f"Expected 'time,value' in <pt> element, got: {text!r}"
        )
    return float(parts[0].strip()), float(parts[1].strip())


def _format_pt(time_val: float, scale_val: float) -> str:
    """
    Format a loadcurve point as "t,v" with up to 10 significant figures,
    stripping unnecessary trailing zeros.

    Examples:
        _format_pt(4.5, 1.0)   → "4.5,1"
        _format_pt(4.0, 0.0)   → "4,0"
        _format_pt(4.166666667, 1.0) → "4.166666667,1"
    """
    def _fmt(v: float) -> str:
        # Use up to 10 significant figures, strip trailing zeros
        s = f"{v:.10g}"
        return s

    return f"{_fmt(time_val)},{_fmt(scale_val)}"


# ---------------------------------------------------------------------------
# Convenience function (used by runner / tasks)
# ---------------------------------------------------------------------------

def configure_simulation(
    speed_mm_s: float,
    output_path: Path,
    base_feb_path: Path | None = None,
    settings: Settings | None = None,
) -> ConfigurationResult:
    """
    Module-level convenience wrapper around SimulationConfigurator.

    Args:
        speed_mm_s:    Insertion speed in mm/s.
        output_path:   Destination path for the modified file.
        base_feb_path: Override base template path (optional).
        settings:      Application settings (optional; uses cached singleton).

    Returns:
        ConfigurationResult
    """
    configurator = SimulationConfigurator(settings=settings)
    return configurator.configure(
        speed_mm_s=speed_mm_s,
        output_path=output_path,
        base_feb_path=base_feb_path,
    )
