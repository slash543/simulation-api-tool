"""
Shared pytest fixtures for the Digital Twin UI test suite.

Fixtures defined here are auto-discovered by pytest and available in
every test module without explicit import.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
TEMPLATE_FEB = PROJECT_ROOT / "templates" / "sample_catheterization.feb"


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Absolute path to the project root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def template_feb() -> Path:
    """
    Path to the base simulation template file.
    Skips entire test session if the template is missing.
    """
    if not TEMPLATE_FEB.exists():
        pytest.skip(f"Simulation template not found: {TEMPLATE_FEB}")
    return TEMPLATE_FEB


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    """Temporary run directory, mimicking runs/run_XXXX layout."""
    d = tmp_path / "runs" / "run_0001"
    d.mkdir(parents=True)
    return d


# ---------------------------------------------------------------------------
# Minimal synthetic FEB fixture (no real template needed for unit tests)
# ---------------------------------------------------------------------------

MINIMAL_FEB_TEMPLATE = """\
<?xml version="1.0" encoding="ISO-8859-1"?>
<febio_spec version="4.0">
  <Material>
    <material id="1" name="urethra" type="neo-Hookean">
      <E>1</E><v>0.49</v>
    </material>
  </Material>
  <Step>
    <step id="1" name="Step1">
      <Control>
        <time_steps>40</time_steps>
        <step_size>0.05</step_size>
      </Control>
    </step>
    <step id="2" name="Step2">
      <Control>
        <time_steps>40</time_steps>
        <step_size>0.05</step_size>
      </Control>
      <Boundary>
        <bc name="event2" type="prescribed displacement">
          <dof>z</dof>
          <value lc="1">10</value>
        </bc>
      </Boundary>
    </step>
  </Step>
  <LoadData>
    <load_controller id="1" name="LC1" type="loadcurve">
      <interpolate>SMOOTH STEP</interpolate>
      <extend>CONSTANT</extend>
      <points>
        <pt>2,0</pt>
        <pt>4,1</pt>
      </points>
    </load_controller>
  </LoadData>
</febio_spec>
"""


@pytest.fixture()
def minimal_feb(tmp_path: Path) -> Path:
    """
    A small, synthetic .feb file that has exactly the elements the
    SimulationConfigurator needs — no 17 000-line real mesh required.
    Use this for fast unit tests; use template_feb for integration tests.
    """
    p = tmp_path / "minimal.feb"
    p.write_text(MINIMAL_FEB_TEMPLATE, encoding="ISO-8859-1")
    return p


# ---------------------------------------------------------------------------
# Environment variable helper
# ---------------------------------------------------------------------------

@contextmanager
def patch_env(env: dict[str, str]):
    """
    Context manager that temporarily sets environment variables.

    Usage::

        with patch_env({"DTUI__API__PORT": "9000"}):
            ...
    """
    previous = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        yield
    finally:
        for k, old_val in previous.items():
            if old_val is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old_val


@pytest.fixture()
def clean_settings_cache():
    """
    Clears the lru_cache on get_settings() before and after each test,
    so YAML / env-override tests do not bleed into each other.
    """
    from digital_twin_ui.app.core.config import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture()
def reset_logging_state():
    """
    Resets the _configured flag in the logging module before and after
    each test so logging tests are fully isolated.
    """
    import digital_twin_ui.app.core.logging as log_module
    log_module._configured = False
    yield
    log_module._configured = False
