"""
FEB File Reader
===============
Reads and inspects FEBio .feb XML files without modifying them.

In FEBio terminology, each ``<step>`` element in the XML represents one
analysis step — also referred to as an *event*.  The catheter-insertion
simulations require exactly 10 such steps.

Provides
--------
- :class:`FebStepInfo` — details for one ``<step>`` element
- :class:`FebInfo`     — full structural summary of a .feb file
- :func:`read_feb`     — parse a .feb file and return a :class:`FebInfo`
- :func:`validate_feb_steps` — raise :exc:`ValueError` when the step count
  does not match the expected value; use this as a pre-run guard

Usage
-----
    from digital_twin_ui.simulation.feb_reader import read_feb, validate_feb_steps

    # Inspect a file
    info = read_feb("base_configuration/ball_tip_14FR_ir12.feb")
    print(info.n_steps)          # 10
    print(info.n_load_controllers)  # 10
    for s in info.steps:
        print(s.step_id, s.name, s.lc_ids, s.time_steps)

    # Validate before running (raises ValueError if not 10 steps)
    validate_feb_steps("base_configuration/ball_tip_14FR_ir12.feb", required_steps=10)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lxml import etree

from digital_twin_ui.app.core.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FebStepInfo:
    """
    Summary of a single ``<step>`` element.

    Attributes:
        step_id:    Integer id attribute of the ``<step>`` element.
        name:       Name attribute (may be None if not set in the file).
        lc_ids:     Sorted list of load controller ids referenced by boundary
                    conditions inside this step (via ``<value lc="N">``).
        time_steps: Value of ``<time_steps>`` inside ``<Control>``, or None if
                    the element is absent.
    """

    step_id: int
    name: str | None
    lc_ids: list[int]
    time_steps: int | None


@dataclass(frozen=True)
class FebInfo:
    """
    Structural summary of a .feb file relevant to simulation configuration.

    Only elements needed for validation and configuration are inspected;
    geometry and mesh data are not held in memory after this call.

    Attributes:
        path:               Resolved absolute path to the file.
        version:            FEBio spec version string (e.g. ``"4.0"``).
        n_steps:            Number of ``<step>`` elements found (= number of
                            events / analysis steps).
        steps:              Detailed info for each step, in document order.
        n_load_controllers: Total number of ``<load_controller>`` elements.
        lc_ids:             Sorted list of load controller ids.
    """

    path: Path
    version: str
    n_steps: int
    steps: list[FebStepInfo]
    n_load_controllers: int
    lc_ids: list[int]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "path": str(self.path),
            "version": self.version,
            "n_steps": self.n_steps,
            "steps": [
                {
                    "step_id": s.step_id,
                    "name": s.name,
                    "lc_ids": s.lc_ids,
                    "time_steps": s.time_steps,
                }
                for s in self.steps
            ],
            "n_load_controllers": self.n_load_controllers,
            "lc_ids": self.lc_ids,
        }


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


def read_feb(path: Path | str) -> FebInfo:
    """
    Parse a .feb XML file and return a :class:`FebInfo` summary.

    Only the structural elements needed for simulation configuration are
    inspected.  Geometry/mesh data is not retained after this call.

    Args:
        path: Path to the .feb file (absolute or relative).

    Returns:
        :class:`FebInfo` with step count, load controller ids, etc.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError:        If the file cannot be parsed as XML.
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f".feb file not found: {path}")

    logger.debug("Reading .feb file: {p}", p=str(path))

    try:
        parser = etree.XMLParser(remove_blank_text=False, remove_comments=False)
        tree = etree.parse(str(path), parser)
    except etree.XMLSyntaxError as exc:
        raise ValueError(
            f"Cannot parse .feb file as XML: {path}\n{exc}"
        ) from exc

    root = tree.getroot()

    # --- Spec version ---
    version = root.get("version", "unknown")

    # --- Step elements ---
    # In FEBio, each <step> is one analysis step / event.
    step_elems = root.findall(".//step")
    steps: list[FebStepInfo] = []
    for elem in step_elems:
        id_str = elem.get("id")
        step_id = int(id_str) if id_str is not None else 0
        name = elem.get("name")

        # Collect LC ids referenced by <value lc="N"> inside this step
        lc_ids: list[int] = []
        for val_elem in elem.findall(".//value[@lc]"):
            try:
                lc_ids.append(int(val_elem.get("lc")))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                pass

        # <time_steps> inside <Control>
        ts_elem = elem.find(".//Control/time_steps")
        time_steps: int | None = None
        if ts_elem is not None and ts_elem.text:
            try:
                time_steps = int(ts_elem.text.strip())
            except ValueError:
                pass

        steps.append(FebStepInfo(
            step_id=step_id,
            name=name,
            lc_ids=sorted(set(lc_ids)),
            time_steps=time_steps,
        ))

    # --- Load controller elements ---
    lc_elems = root.findall(".//load_controller")
    lc_ids_list: list[int] = []
    for lc_elem in lc_elems:
        id_str = lc_elem.get("id")
        if id_str is not None:
            try:
                lc_ids_list.append(int(id_str))
            except ValueError:
                pass

    info = FebInfo(
        path=path,
        version=version,
        n_steps=len(steps),
        steps=steps,
        n_load_controllers=len(lc_elems),
        lc_ids=sorted(lc_ids_list),
    )

    logger.info(
        "FEB file read: {p} — version={v}, steps={s}, load_controllers={lc}",
        p=path.name,
        v=info.version,
        s=info.n_steps,
        lc=info.n_load_controllers,
    )

    return info


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


def validate_feb_steps(path: Path | str, required_steps: int) -> FebInfo:
    """
    Read a .feb file and verify it contains exactly *required_steps* steps.

    Each ``<step>`` element counts as one step (= one event in FEBio terms).
    The catheter-insertion templates require exactly 10 steps.

    Args:
        path:           Path to the .feb file.
        required_steps: Expected number of ``<step>`` elements.

    Returns:
        :class:`FebInfo` (so callers can reuse it without re-parsing).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError:        If the step count does not equal *required_steps*.

    Example::

        info = validate_feb_steps(
            "base_configuration/ball_tip_14FR_ir12.feb",
            required_steps=10,
        )
        # Raises ValueError if the file has ≠ 10 <step> elements
    """
    info = read_feb(path)

    if info.n_steps != required_steps:
        raise ValueError(
            f"FEB file validation failed: expected {required_steps} step(s) "
            f"but found {info.n_steps}.\n"
            f"File: {info.path}\n"
            f"Steps found: "
            + ", ".join(
                f"id={s.step_id}" + (f" name={s.name!r}" if s.name else "")
                for s in info.steps
            )
        )

    logger.info(
        "FEB validation passed: {p} has {n} step(s) as required",
        p=Path(path).name,
        n=info.n_steps,
    )
    return info
