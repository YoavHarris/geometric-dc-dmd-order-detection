"""
Common plotting utilities for refactored plotters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import yaml


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r") as f:
        return yaml.safe_load(f)


def resolve_project_root(cfg: dict[str, Any], config_path: str | Path) -> Path:
    """
    Resolve the project root directory.

    If 'project_root' is in the config, it is resolved relative to the config file location.
    Otherwise, defaults to the current working directory.
    """
    config_path = Path(config_path).resolve()

    if "project_root" in cfg:
        # Resolve relative to the config file
        root_rel = cfg["project_root"]
        return (config_path.parent / root_rel).resolve()

    # Default to current working directory if not specified
    return Path.cwd()


def resolve_path(root: Path, rel: str | Path) -> Path:
    """Resolve a path relative to the project root."""
    return (root / rel).resolve()


def apply_style(plot_cfg: Mapping[str, Any], project_root: Path) -> None:
    """
    Apply matplotlib style based on configuration.

    Supports:
    - style_mode: "single" | "double" | "presentation" | "custom"
    - custom_style_path: path (used if style_mode == "custom")
    - mplstyle_path: path (legacy/direct support)
    """
    # Check for direct path first (legacy/simple support)
    if "mplstyle_path" in plot_cfg:
        style_path = resolve_path(project_root, plot_cfg["mplstyle_path"])
        if not style_path.exists():
            raise FileNotFoundError(f"mplstyle file not found: {style_path}")
        plt.style.use(str(style_path))
        return

    # Check for style_mode
    mode = plot_cfg.get("mplstyle", {}).get("style_mode", "double")

    if mode == "custom":
        custom_path = plot_cfg.get("mplstyle", {}).get("custom_style_path")
        if not custom_path:
            raise ValueError(
                "style_mode is 'custom' but 'custom_style_path' is missing."
            )
        style_path = resolve_path(project_root, custom_path)
    elif mode == "single":
        style_path = resolve_path(
            project_root, "figures/mplstyle_files/chaos_single.mplstyle"
        )
    elif mode == "double":
        style_path = resolve_path(
            project_root, "figures/mplstyle_files/chaos_double.mplstyle"
        )
    elif mode == "presentation":
        style_path = resolve_path(
            project_root, "figures/mplstyle_files/presentation.mplstyle"
        )
    else:
        raise ValueError(f"Unknown style_mode: {mode}")

    if not style_path.exists():
        raise FileNotFoundError(f"mplstyle file not found: {style_path}")

    plt.style.use(str(style_path))
