"""
Catheter Catalogue
==================
Loads ``config/catheter_catalogue.yaml`` — the single source of truth for all
catheter designs — and provides typed lookups for the API, tasks, and MCP tools.

Selection hierarchy exposed to the user:
    1. Tip design  (e.g. "ball_tip" → "Ball Tip")
    2. Configuration — size × urethra model  (e.g. "14Fr_IR12")
    3. Per-step insertion speeds  (10 values for the 10-step analysis)

Adding a new combination only requires dropping a .feb file in
``base_configuration/`` and adding the entry in ``catheter_catalogue.yaml``.
No Python code changes are needed.

Usage
-----
    from digital_twin_ui.simulation.catheter_catalogue import get_catalogue

    cat = get_catalogue()

    # List designs for the UI / agent
    for d in cat.designs:
        print(d.name, d.label, [c.key for c in d.configurations])

    # Resolve to a TemplateConfig for MultiStepConfigurator
    tc = cat.resolve(design="ball_tip", configuration="14Fr_IR12")
    print(tc.feb_path)     # .../base_configuration/ball_tip_14FR_ir12.feb

    # Shared simulation parameters
    params = cat.simulation_params
    print(params.displacements_mm)   # [64.0, 46.0, 28.0, ...]
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from digital_twin_ui.app.core.config import get_settings
from digital_twin_ui.app.core.logging import get_logger
from digital_twin_ui.simulation.template_registry import SpeedRange, TemplateConfig

logger = get_logger(__name__)

_CATALOGUE_FILE = "config/catheter_catalogue.yaml"
_FEB_SUBDIR = "base_configuration"

# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

# Pattern: <design_key>_<size>Fr[optional extra words]_ir<ir_value>
# Fully case-insensitive — handles Ball_tip_14Fr_IR12 and ball_tip_14FR_ir12.
# Design key is normalised to lowercase so Ball_tip and ball_tip are identical.
# Examples handled:
#   Ball_tip_14Fr_IR12             → ('ball_tip',          '14', '12')
#   ball_tip_14FR_ir12             → ('ball_tip',          '14', '12')
#   nelaton_tip_16Fr_ir25          → ('nelaton_tip',        '16', '25')
#   vapro_introducer_14Fr_tip_ir12 → ('vapro_introducer',   '14', '12')
_FEB_RE = re.compile(r"^(.+?)_(\d+)fr.*?_ir(\d+)$", re.IGNORECASE)


def _parse_feb_filename(stem: str) -> tuple[str, str, str] | None:
    """Parse a .feb filename stem to (design_key, size_fr, ir_value) or None.

    The design_key is always returned in lowercase so filenames with mixed
    capitalisation (e.g. ``Ball_tip``) are treated identically to their
    all-lowercase equivalents (``ball_tip``).
    """
    m = _FEB_RE.match(stem)
    if m:
        design_key = m.group(1).lower()   # normalise: Ball_tip → ball_tip
        return design_key, m.group(2), m.group(3)
    return None


def _design_label_from_key(design_key: str) -> str:
    """Convert a snake_case design key to a human-readable label.

    Examples:
        'ball_tip'          → 'Ball Tip'
        'vapro_introducer'  → 'Vapro Introducer'
    """
    return " ".join(w.capitalize() for w in design_key.split("_"))


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CatalogueConfiguration:
    """One specific catheter size + urethra-model combination."""

    key: str        # e.g. "14Fr_IR12"
    label: str      # e.g. "14Fr catheter — IR12 urethra model"
    feb_file: str   # filename in base_configuration/


@dataclass(frozen=True)
class CatalogueDesign:
    """One catheter tip type with all its available configurations."""

    name: str                                    # e.g. "ball_tip"
    label: str                                   # e.g. "Ball Tip"
    configurations: list[CatalogueConfiguration] # ordered as in YAML

    def get_configuration(self, key: str) -> CatalogueConfiguration:
        """Return configuration by key, raising KeyError if not found."""
        for c in self.configurations:
            if c.key == key:
                return c
        available = [c.key for c in self.configurations]
        raise KeyError(
            f"Configuration '{key}' not found for design '{self.name}'. "
            f"Available: {available}"
        )


@dataclass(frozen=True)
class SimulationParams:
    """Shared simulation parameters that apply to all base_configuration files."""

    n_steps: int
    base_step_size: float
    default_dwell_time_s: float
    displacements_mm: list[float]
    speed_min_mm_s: float
    speed_max_mm_s: float
    default_uniform_speed_mm_s: float


# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------


class CatheterCatalogue:
    """
    Parses ``config/catheter_catalogue.yaml`` and provides typed lookups.

    Args:
        catalogue_path: Override the YAML file path.
        project_root:   Project root for resolving paths.  Defaults to settings.
    """

    def __init__(
        self,
        catalogue_path: Path | None = None,
        project_root: Path | None = None,
    ) -> None:
        settings = get_settings()
        self._project_root = project_root or settings.project_root
        self._path = catalogue_path or (self._project_root / _CATALOGUE_FILE)
        self._designs: list[CatalogueDesign] = []
        self._params: SimulationParams | None = None
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def designs(self) -> list[CatalogueDesign]:
        """All registered catheter designs (in catalogue order)."""
        return self._designs

    @property
    def simulation_params(self) -> SimulationParams:
        """Shared simulation parameters for all designs."""
        assert self._params is not None
        return self._params

    def get_design(self, name: str) -> CatalogueDesign:
        """
        Return design by name.

        Raises:
            KeyError: If the design name is not in the catalogue.
        """
        for d in self._designs:
            if d.name == name:
                return d
        available = [d.name for d in self._designs]
        raise KeyError(
            f"Catheter design '{name}' not found. Available: {available}"
        )

    def resolve(self, design: str, configuration: str) -> TemplateConfig:
        """
        Return a :class:`TemplateConfig` suitable for :class:`MultiStepConfigurator`.

        ``feb_path`` points to ``base_configuration/<feb_file>``.

        Raises:
            KeyError: If design or configuration is not found.
            FileNotFoundError: If the .feb file is missing from base_configuration/.
        """
        d = self.get_design(design)
        cfg = d.get_configuration(configuration)
        params = self.simulation_params

        feb_path = self._project_root / _FEB_SUBDIR / cfg.feb_file
        if not feb_path.exists():
            # Case-insensitive fallback for Linux (e.g. Ball_tip_14Fr_IR12.feb vs ball_tip_14FR_ir12.feb)
            feb_dir = self._project_root / _FEB_SUBDIR
            matches = [p for p in feb_dir.glob("*.feb") if p.name.lower() == cfg.feb_file.lower()]
            if matches:
                feb_path = matches[0]
                logger.debug(
                    "Case-insensitive match for '{wanted}' → '{found}'",
                    wanted=cfg.feb_file,
                    found=feb_path.name,
                )
            else:
                raise FileNotFoundError(
                    f"FEB file '{cfg.feb_file}' not found in {_FEB_SUBDIR}/. "
                    f"Expected: {feb_path}"
                )

        return TemplateConfig(
            name=f"{design}__{configuration}",
            label=f"{d.label} — {cfg.label}",
            feb_file=feb_path.name,  # use actual on-disk name (may differ in case from YAML)
            n_steps=params.n_steps,
            base_step_size=params.base_step_size,
            default_dwell_time_s=params.default_dwell_time_s,
            displacements_mm=list(params.displacements_mm),
            speed_range=SpeedRange(
                min_mm_s=params.speed_min_mm_s,
                max_mm_s=params.speed_max_mm_s,
            ),
            _project_root=self._project_root,
            _feb_subdir=_FEB_SUBDIR,
        )

    def uniform_speeds(self, speed_mm_s: float | None = None) -> list[float]:
        """
        Return a uniform speed vector of length n_steps.

        Args:
            speed_mm_s: Speed for all steps.  If None, uses
                        ``simulation_params.default_uniform_speed_mm_s``.
        """
        s = speed_mm_s if speed_mm_s is not None else self.simulation_params.default_uniform_speed_mm_s
        return [s] * self.simulation_params.n_steps

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._path.exists():
            raise FileNotFoundError(
                f"Catheter catalogue not found: {self._path}"
            )

        with self._path.open("r", encoding="utf-8") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh) or {}

        # Parse designs
        designs_raw: dict[str, Any] = raw.get("designs", {})
        for name, d_raw in designs_raw.items():
            configs: list[CatalogueConfiguration] = []
            for key, c_raw in (d_raw.get("configurations") or {}).items():
                configs.append(
                    CatalogueConfiguration(
                        key=key,
                        label=str(c_raw.get("label", key)),
                        feb_file=str(c_raw["feb_file"]),
                    )
                )
            self._designs.append(
                CatalogueDesign(
                    name=name,
                    label=str(d_raw.get("label", name)),
                    configurations=configs,
                )
            )

        # Parse shared simulation params
        sim: dict[str, Any] = raw.get("simulation", {})
        speed_raw = sim.get("speed_range", {})
        self._params = SimulationParams(
            n_steps=int(sim.get("n_steps", 10)),
            base_step_size=float(sim.get("base_step_size", 0.1)),
            default_dwell_time_s=float(sim.get("default_dwell_time_s", 1.0)),
            displacements_mm=[float(x) for x in sim.get("displacements_mm", [28.0])],
            speed_min_mm_s=float(speed_raw.get("min_mm_s", 10.0)),
            speed_max_mm_s=float(speed_raw.get("max_mm_s", 25.0)),
            default_uniform_speed_mm_s=float(
                sim.get("default_uniform_speed_mm_s", 15.0)
            ),
        )

        logger.info(
            "CatheterCatalogue loaded {n} designs from YAML: {names}",
            n=len(self._designs),
            names=", ".join(d.name for d in self._designs),
        )

        # Auto-discover any additional .feb files not covered by the YAML
        self._auto_discover()

    def _auto_discover(self) -> None:
        """Scan *base_configuration/* and register any .feb files not in the YAML.

        Filename convention (case-insensitive on Fr/ir):
            ``<design_key>_<size>Fr[_<extra>]_ir<ir_value>.feb``

        Examples::
            ball_tip_14FR_ir12.feb         → ball_tip / 14Fr_IR12
            vapro_introducer_14Fr_tip_ir12.feb → vapro_introducer / 14Fr_IR12

        Any file whose stem cannot be parsed is silently skipped.
        Already-registered files are skipped by filename comparison.
        New designs not yet in the YAML are created automatically.
        """
        feb_dir = self._project_root / _FEB_SUBDIR
        if not feb_dir.exists():
            return

        # Case-insensitive set of already-registered filenames
        registered: set[str] = {
            c.feb_file.lower()
            for d in self._designs
            for c in d.configurations
        }

        # Map design_name → list index for O(1) merge
        design_index: dict[str, int] = {d.name: i for i, d in enumerate(self._designs)}

        n_new = 0
        for feb_path in sorted(feb_dir.glob("*.feb")):
            if feb_path.name.lower() in registered:
                continue  # already in catalogue

            parsed = _parse_feb_filename(feb_path.stem)
            if parsed is None:
                logger.debug(
                    "Auto-discover: skipping unrecognized filename {f}",
                    f=feb_path.name,
                )
                continue

            design_key, size_fr, ir_val = parsed
            config_key = f"{size_fr}Fr_IR{ir_val}"
            new_cfg = CatalogueConfiguration(
                key=config_key,
                label=f"{size_fr}Fr catheter — IR{ir_val} urethra model",
                feb_file=feb_path.name,
            )

            if design_key in design_index:
                idx = design_index[design_key]
                d = self._designs[idx]
                if config_key not in {c.key for c in d.configurations}:
                    # CatalogueDesign is frozen — replace with an updated copy
                    self._designs[idx] = CatalogueDesign(
                        name=d.name,
                        label=d.label,
                        configurations=list(d.configurations) + [new_cfg],
                    )
            else:
                new_design = CatalogueDesign(
                    name=design_key,
                    label=_design_label_from_key(design_key),
                    configurations=[new_cfg],
                )
                self._designs.append(new_design)
                design_index[design_key] = len(self._designs) - 1

            registered.add(feb_path.name.lower())
            n_new += 1
            logger.info("Auto-discovered: {f}", f=feb_path.name)

        if n_new:
            logger.info(
                "Auto-discover complete: {n} new .feb file(s) registered",
                n=n_new,
            )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_catalogue_singleton: CatheterCatalogue | None = None


def get_catalogue() -> CatheterCatalogue:
    """Return (and cache) the global CatheterCatalogue singleton."""
    global _catalogue_singleton
    if _catalogue_singleton is None:
        _catalogue_singleton = CatheterCatalogue()
    return _catalogue_singleton


def reset_catalogue_singleton() -> None:
    """Reset the singleton so the next call to get_catalogue() reloads from disk.

    Intended for testing and for scenarios where new .feb files have been added
    and the running process should pick them up without restarting.
    """
    global _catalogue_singleton
    _catalogue_singleton = None
