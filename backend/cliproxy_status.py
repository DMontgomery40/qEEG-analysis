from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from .llm_client import UpstreamError


def _repo_root() -> Path:
    # backend/cliproxy_status.py -> backend/ -> repo root
    return Path(__file__).resolve().parents[1]


def _default_config_candidates() -> list[str]:
    home = Path.home()
    candidates = [
        os.getenv("CLIPROXY_CONFIG", ""),
        str(_repo_root() / ".cli-proxy-api" / "cliproxyapi.conf"),
        str(_repo_root() / ".cli-proxy-api" / "config.yaml"),
        "/opt/homebrew/etc/cliproxyapi.conf",
        str(home / ".cli-proxy-api" / "config.yaml"),
        str(home / ".config" / "cli-proxy-api" / "config.yaml"),
    ]
    return [c for c in candidates if c]


def _classify_error(err: Exception) -> tuple[str, bool]:
    if isinstance(err, UpstreamError) and err.status_code == 401:
        msg = str(err).lower()
        if "missing api key" in msg:
            return "client_api_key_required", False
        return "unauthorized", True

    msg = str(err).lower()
    if "all connection attempts failed" in msg or "connection refused" in msg:
        return "connection_refused", False
    if "timed out" in msg or "timeout" in msg:
        return "timeout", False
    if "name or service not known" in msg or "nodename nor servname provided" in msg:
        return "dns_error", False
    if isinstance(err, UpstreamError) and err.status_code is not None:
        return f"http_{err.status_code}", False
    return "unknown", False


def _find_proxy_binary() -> str | None:
    candidates = [
        shutil.which("cli-proxy-api-plus"),
        str(Path.home() / ".local" / "bin" / "cli-proxy-api-plus"),
        "/opt/homebrew/bin/cli-proxy-api-plus",
        shutil.which("cliproxyapi"),
        str(Path.home() / ".local" / "bin" / "cliproxyapi"),
        "/opt/homebrew/bin/cliproxyapi",
    ]
    for c in candidates:
        if c and Path(c).exists():
            return c
    return None


def suggested_commands(*, base_url: str) -> list[str]:
    cfg = os.getenv("CLIPROXY_CONFIG", str(_repo_root() / ".cli-proxy-api" / "cliproxyapi.conf"))
    proxy_cmd = _find_proxy_binary() or "cli-proxy-api-plus"
    return [
        "Install CLIProxyAPI Plus releases: https://github.com/router-for-me/CLIProxyAPIPlus/releases",
        "brew install cliproxyapi  # fallback mainline (Homebrew)",
        f"{proxy_cmd} -config {cfg}",
        f"{proxy_cmd} -login",
        f"{proxy_cmd} -claude-login",
        f"{proxy_cmd} -codex-login",
        f"{proxy_cmd} -login -project_id YOUR_PROJECT_ID  # Gemini (optional)",
    ]


def status_payload(
    *,
    base_url: str,
    discovered_model_count: int,
    reachable: bool,
    error: Exception | None = None,
) -> dict[str, Any]:
    binary_path = _find_proxy_binary()
    binary_found = binary_path is not None
    brew_found = shutil.which("brew") is not None
    config_candidates = _default_config_candidates()
    existing_configs = [c for c in config_candidates if Path(os.path.expanduser(c)).exists()]

    error_kind = None
    auth_required = False
    error_text = None
    if error is not None:
        error_kind, auth_required = _classify_error(error)
        error_text = str(error)

    return {
        "cliproxy_base_url": base_url,
        "cliproxy_reachable": reachable,
        "cliproxy_auth_required": auth_required,
        "cliproxy_error_kind": error_kind,
        "cliproxy_error": error_text,
        "discovered_model_count": discovered_model_count,
        "cliproxyapi_installed": binary_found,
        "cliproxyapi_path": binary_path,
        "brew_installed": brew_found,
        "cliproxy_config_candidates": config_candidates,
        "cliproxy_config_found": existing_configs,
        "suggested_commands": suggested_commands(base_url=base_url),
    }
