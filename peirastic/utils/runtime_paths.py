"""Reusable runtime paths for data generated outside the source tree."""

from __future__ import annotations

import os
from pathlib import Path

PEIRASTIC_NETFT_WORKSPACE_ENV = "PEIRASTIC_NETFT_WORKSPACE"

_DEFAULT_RUNTIME_ROOT = Path.home() / ".local" / "share" / "peirastic"
_DEFAULT_NETFT_WORKSPACE = _DEFAULT_RUNTIME_ROOT / "netft"


def _expand_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def get_netft_workspace(path_prefix: str | None = None) -> str:
    """Resolve the NetFT workspace directory."""
    if path_prefix:
        return _expand_path(path_prefix)

    env_override = os.environ.get(PEIRASTIC_NETFT_WORKSPACE_ENV)
    if env_override:
        return _expand_path(env_override)

    return str(_DEFAULT_NETFT_WORKSPACE)


def get_default_netft_calib_yaml(path_prefix: str | None = None) -> str:
    """Return the default NetFT calibration YAML path."""
    workspace = Path(get_netft_workspace(path_prefix))
    return str(workspace / "config" / "netft_calib_result.yaml")
