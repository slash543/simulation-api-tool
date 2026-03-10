"""
Template Registry
=================
Scans ``config/templates/`` for YAML template definitions and provides
a typed registry so callers can look up simulation templates by name.

Each YAML file describes one FEB template: the file to use, how many
steps it has, what speed range is valid, and the per-step displacements.

Usage
-----
    from digital_twin_ui.simulation.template_registry import TemplateRegistry

    registry = TemplateRegistry()
    tc = registry.get("DT_BT_14Fr_FO_10E_IR12")

    print(tc.name)               # "DT_BT_14Fr_FO_10E_IR12"
    print(tc.is_multi_step)      # True
    print(tc.feb_path)           # absolute Path to .feb file
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from digital_twin_ui.app.core.config import get_settings
from digital_twin_ui.app.core.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# TemplateConfig dataclass
# ---------------------------------------------------------------------------


@dataclass
class SpeedRange:
    """Speed bounds for a template."""

    min_mm_s: float
    max_mm_s: float


@dataclass
class TemplateConfig:
    """
    All configuration for one FEB simulation template.

    Attributes:
        name:               Unique identifier string (matches YAML filename stem).
        label:              Human-readable display name.
        feb_file:           Filename of the .feb template (resolved relative to
                            project_root/templates/).
        n_steps:            Number of load-curve steps in the FEB file.
        base_step_size:     Solver step size in seconds (kept fixed per step).
        default_dwell_time_s: Default dwell time appended after each ramp.
        displacements_mm:   Prescribed displacement for each step (mm).
        speed_range:        Valid insertion speed range.
        _project_root:      Used internally for path resolution (not public).
    """

    name: str
    label: str
    feb_file: str
    n_steps: int
    base_step_size: float
    default_dwell_time_s: float
    displacements_mm: list[float]
    speed_range: SpeedRange
    _project_root: Path = field(default_factory=Path.cwd, repr=False, compare=False)

    @property
    def is_multi_step(self) -> bool:
        """True if the template has more than one insertion step."""
        return self.n_steps > 1

    @property
    def feb_path(self) -> Path:
        """Absolute path to the .feb template file."""
        return self._project_root / "templates" / self.feb_file

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary of this config."""
        return {
            "name": self.name,
            "label": self.label,
            "feb_file": self.feb_file,
            "n_steps": self.n_steps,
            "base_step_size": self.base_step_size,
            "default_dwell_time_s": self.default_dwell_time_s,
            "displacements_mm": self.displacements_mm,
            "speed_range": {
                "min_mm_s": self.speed_range.min_mm_s,
                "max_mm_s": self.speed_range.max_mm_s,
            },
            "is_multi_step": self.is_multi_step,
            "feb_path": str(self.feb_path),
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TemplateRegistry:
    """
    Scans ``config/templates/`` at construction time and exposes a lookup
    by template name.

    Args:
        templates_dir: Override the directory to scan.  If None, uses
                       ``<project_root>/config/templates/``.
        project_root:  Project root used for resolving feb_path.  If None,
                       uses the value from :func:`get_settings`.
    """

    def __init__(
        self,
        templates_dir: Path | None = None,
        project_root: Path | None = None,
    ) -> None:
        settings = get_settings()
        self._project_root = project_root or settings.project_root
        self._templates_dir = templates_dir or (
            self._project_root / "config" / "templates"
        )
        self._registry: dict[str, TemplateConfig] = {}
        self._scan()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, name: str) -> TemplateConfig:
        """
        Return the TemplateConfig for *name*.

        Args:
            name: Template name (e.g. ``"DT_BT_14Fr_FO_10E_IR12"``).

        Returns:
            :class:`TemplateConfig`

        Raises:
            KeyError: If no template with that name was found.
        """
        if name not in self._registry:
            available = ", ".join(sorted(self._registry))
            raise KeyError(
                f"Template '{name}' not found. Available: {available}"
            )
        return self._registry[name]

    def list_templates(self) -> list[str]:
        """Return sorted list of registered template names."""
        return sorted(self._registry)

    def all_configs(self) -> list[TemplateConfig]:
        """Return all registered TemplateConfig objects sorted by name."""
        return [self._registry[n] for n in self.list_templates()]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _scan(self) -> None:
        """Load all *.yaml files from the templates directory."""
        if not self._templates_dir.exists():
            logger.warning(
                "Templates directory not found: {d}", d=str(self._templates_dir)
            )
            return

        yaml_files = sorted(self._templates_dir.glob("*.yaml"))
        if not yaml_files:
            logger.warning(
                "No template YAML files found in {d}", d=str(self._templates_dir)
            )
            return

        for path in yaml_files:
            try:
                tc = self._load_one(path)
                self._registry[tc.name] = tc
                logger.debug(
                    "Loaded template '{n}' from {p}", n=tc.name, p=str(path)
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Failed to load template YAML {p}: {e}", p=str(path), e=exc
                )

        logger.info(
            "TemplateRegistry loaded {n} templates: {names}",
            n=len(self._registry),
            names=", ".join(sorted(self._registry)),
        )

    def _load_one(self, path: Path) -> TemplateConfig:
        """Parse one YAML file into a TemplateConfig."""
        with path.open("r", encoding="utf-8") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh) or {}

        speed_raw = raw.get("speed_range", {})
        speed_range = SpeedRange(
            min_mm_s=float(speed_raw.get("min_mm_s", 0.0)),
            max_mm_s=float(speed_raw.get("max_mm_s", 100.0)),
        )

        return TemplateConfig(
            name=str(raw["name"]),
            label=str(raw.get("label", raw["name"])),
            feb_file=str(raw["feb_file"]),
            n_steps=int(raw.get("n_steps", 1)),
            base_step_size=float(raw.get("base_step_size", 0.05)),
            default_dwell_time_s=float(raw.get("default_dwell_time_s", 0.0)),
            displacements_mm=[float(d) for d in raw.get("displacements_mm", [10.0])],
            speed_range=speed_range,
            _project_root=self._project_root,
        )


# ---------------------------------------------------------------------------
# Module-level singleton accessor
# ---------------------------------------------------------------------------

_registry_singleton: TemplateRegistry | None = None


def get_template_registry() -> TemplateRegistry:
    """
    Return (and cache) the global TemplateRegistry singleton.

    Thread-safety note: In practice the registry is initialised at application
    startup before concurrent requests arrive, so no lock is needed.
    """
    global _registry_singleton
    if _registry_singleton is None:
        _registry_singleton = TemplateRegistry()
    return _registry_singleton
