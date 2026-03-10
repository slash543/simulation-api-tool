"""
Sanity checks for Docker configuration files.

These tests validate:
  - Dockerfile exists and contains required stage names
  - docker-compose.yml exists and is valid YAML with required service keys
  - .dockerignore exists and excludes sensitive/bulky paths
  - .env.example exists and documents required variables

No Docker daemon is required — these are pure file/text checks.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Dockerfile
# ---------------------------------------------------------------------------

class TestDockerfile:
    @pytest.fixture()
    def dockerfile(self) -> str:
        path = PROJECT_ROOT / "Dockerfile"
        assert path.exists(), "Dockerfile not found"
        return path.read_text()

    def test_dockerfile_exists(self) -> None:
        assert (PROJECT_ROOT / "Dockerfile").exists()

    def test_has_base_stage(self, dockerfile: str) -> None:
        assert "FROM python:3.12" in dockerfile

    def test_has_builder_stage(self, dockerfile: str) -> None:
        assert "AS builder" in dockerfile

    def test_has_api_stage(self, dockerfile: str) -> None:
        assert "AS api" in dockerfile

    def test_has_worker_stage(self, dockerfile: str) -> None:
        assert "AS worker" in dockerfile

    def test_exposes_port_8000(self, dockerfile: str) -> None:
        assert "EXPOSE 8000" in dockerfile

    def test_uvicorn_cmd_in_api(self, dockerfile: str) -> None:
        assert "uvicorn" in dockerfile

    def test_celery_cmd_in_worker(self, dockerfile: str) -> None:
        assert "celery" in dockerfile

    def test_non_root_user(self, dockerfile: str) -> None:
        assert "useradd" in dockerfile
        assert "USER appuser" in dockerfile

    def test_healthcheck_api(self, dockerfile: str) -> None:
        assert "HEALTHCHECK" in dockerfile

    def test_pythondontwritebytecode(self, dockerfile: str) -> None:
        assert "PYTHONDONTWRITEBYTECODE" in dockerfile

    def test_requirements_copied(self, dockerfile: str) -> None:
        assert "requirements.txt" in dockerfile

    def test_pth_file_created(self, dockerfile: str) -> None:
        assert "digital_twin_ui.pth" in dockerfile


# ---------------------------------------------------------------------------
# docker-compose.yml
# ---------------------------------------------------------------------------

class TestDockerCompose:
    @pytest.fixture()
    def compose(self) -> dict:
        path = PROJECT_ROOT / "docker-compose.yml"
        assert path.exists(), "docker-compose.yml not found"
        with path.open() as f:
            return yaml.safe_load(f)

    def test_compose_exists(self) -> None:
        assert (PROJECT_ROOT / "docker-compose.yml").exists()

    def test_valid_yaml(self, compose: dict) -> None:
        assert isinstance(compose, dict)

    def test_services_key(self, compose: dict) -> None:
        assert "services" in compose

    def test_redis_service(self, compose: dict) -> None:
        assert "redis" in compose["services"]

    def test_api_service(self, compose: dict) -> None:
        assert "api" in compose["services"]

    def test_worker_service(self, compose: dict) -> None:
        assert "worker" in compose["services"]

    def test_mlflow_service(self, compose: dict) -> None:
        assert "mlflow" in compose["services"]

    def test_flower_service(self, compose: dict) -> None:
        assert "flower" in compose["services"]

    def test_api_exposes_port(self, compose: dict) -> None:
        api_ports = compose["services"]["api"].get("ports", [])
        assert any("8000" in str(p) for p in api_ports)

    def test_mlflow_exposes_port(self, compose: dict) -> None:
        mlflow_ports = compose["services"]["mlflow"].get("ports", [])
        assert any("5000" in str(p) for p in mlflow_ports)

    def test_redis_has_healthcheck(self, compose: dict) -> None:
        assert "healthcheck" in compose["services"]["redis"]

    def test_api_depends_on_redis(self, compose: dict) -> None:
        deps = compose["services"]["api"].get("depends_on", {})
        assert "redis" in deps

    def test_worker_depends_on_redis(self, compose: dict) -> None:
        deps = compose["services"]["worker"].get("depends_on", {})
        assert "redis" in deps

    def test_volumes_defined(self, compose: dict) -> None:
        assert "volumes" in compose
        volumes = compose["volumes"]
        assert "redis_data" in volumes
        assert "mlflow_data" in volumes
        assert "runs_data" in volumes

    def test_flower_has_profile(self, compose: dict) -> None:
        flower = compose["services"]["flower"]
        profiles = flower.get("profiles", [])
        assert "monitoring" in profiles

    def test_worker_has_memory_limit(self, compose: dict) -> None:
        worker = compose["services"]["worker"]
        deploy = worker.get("deploy", {})
        resources = deploy.get("resources", {})
        limits = resources.get("limits", {})
        assert "memory" in limits

    def test_api_build_target(self, compose: dict) -> None:
        build = compose["services"]["api"].get("build", {})
        assert build.get("target") == "api"

    def test_worker_build_target(self, compose: dict) -> None:
        build = compose["services"]["worker"].get("build", {})
        assert build.get("target") == "worker"


# ---------------------------------------------------------------------------
# docker-compose.override.yml
# ---------------------------------------------------------------------------

class TestDockerComposeOverride:
    @pytest.fixture()
    def override(self) -> dict:
        path = PROJECT_ROOT / "docker-compose.override.yml"
        assert path.exists(), "docker-compose.override.yml not found"
        with path.open() as f:
            return yaml.safe_load(f)

    def test_override_exists(self) -> None:
        assert (PROJECT_ROOT / "docker-compose.override.yml").exists()

    def test_valid_yaml(self, override: dict) -> None:
        assert isinstance(override, dict)

    def test_api_service_present(self, override: dict) -> None:
        assert "api" in override.get("services", {})

    def test_worker_service_present(self, override: dict) -> None:
        assert "worker" in override.get("services", {})

    def test_api_has_reload(self, override: dict) -> None:
        api = override["services"]["api"]
        command = api.get("command", "")
        assert "--reload" in str(command)

    def test_source_bind_mounted(self, override: dict) -> None:
        api = override["services"]["api"]
        vols = str(api.get("volumes", ""))
        assert "digital_twin_ui" in vols


# ---------------------------------------------------------------------------
# .dockerignore
# ---------------------------------------------------------------------------

class TestDockerignore:
    @pytest.fixture()
    def dockerignore(self) -> str:
        path = PROJECT_ROOT / ".dockerignore"
        assert path.exists(), ".dockerignore not found"
        return path.read_text()

    def test_dockerignore_exists(self) -> None:
        assert (PROJECT_ROOT / ".dockerignore").exists()

    def test_excludes_venv(self, dockerignore: str) -> None:
        assert ".venv" in dockerignore

    def test_excludes_pycache(self, dockerignore: str) -> None:
        assert "__pycache__" in dockerignore

    def test_excludes_git(self, dockerignore: str) -> None:
        assert ".git" in dockerignore

    def test_excludes_xplt(self, dockerignore: str) -> None:
        assert "*.xplt" in dockerignore

    def test_excludes_runs(self, dockerignore: str) -> None:
        assert "runs/" in dockerignore

    def test_excludes_logs(self, dockerignore: str) -> None:
        assert "logs/" in dockerignore

    def test_excludes_env_files(self, dockerignore: str) -> None:
        assert ".env" in dockerignore


# ---------------------------------------------------------------------------
# .env.example
# ---------------------------------------------------------------------------

class TestEnvExample:
    @pytest.fixture()
    def env_example(self) -> str:
        path = PROJECT_ROOT / ".env.example"
        assert path.exists(), ".env.example not found"
        return path.read_text()

    def test_env_example_exists(self) -> None:
        assert (PROJECT_ROOT / ".env.example").exists()

    def test_documents_api_port(self, env_example: str) -> None:
        assert "DTUI__API__PORT" in env_example

    def test_documents_broker_url(self, env_example: str) -> None:
        assert "DTUI__CELERY__BROKER_URL" in env_example

    def test_documents_mlflow_uri(self, env_example: str) -> None:
        assert "DTUI__MLFLOW__TRACKING_URI" in env_example

    def test_documents_simulator(self, env_example: str) -> None:
        assert "DTUI__SIMULATION__SIMULATOR_EXECUTABLE" in env_example

    def test_documents_log_level(self, env_example: str) -> None:
        assert "DTUI__LOGGING__LEVEL" in env_example
