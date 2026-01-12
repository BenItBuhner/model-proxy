"""
Shared configuration path helpers for Model-Proxy.

Keeps config discovery consistent across CLI and runtime.
"""

import os
from pathlib import Path
from typing import Iterable, List, Optional

_PRIMARY_CONFIG_DIR: Optional[Path] = None


def _unique_paths(paths: Iterable[Path]) -> List[Path]:
    unique: List[Path] = []
    seen = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def get_package_config_dir() -> Path:
    """Return the packaged config directory (read-only in installed builds)."""
    return Path(__file__).resolve().parent.parent.parent / "config"


def get_cwd_config_dir() -> Path:
    """Return the config directory under the current working directory."""
    return Path.cwd() / "config"


def get_user_config_dir() -> Path:
    """Return the per-user config directory."""
    if os.name == "nt":
        base = os.getenv("LOCALAPPDATA") or os.getenv("APPDATA")
        if base:
            return Path(base) / "model-proxy" / "config"
    return Path.home() / ".model-proxy" / "config"


def _is_writable_dir(path: Path, create: bool) -> bool:
    """Best-effort check if a directory is writable (and deletable)."""
    try:
        if create:
            path.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            return False
        test_file = path / ".model_proxy_write_test"
        with open(test_file, "w", encoding="utf-8") as handle:
            handle.write("ok")
        test_file.unlink()
        return True
    except Exception:
        return False


def get_config_search_paths() -> List[Path]:
    """
    Return config roots in precedence order for reads.

    Order: primary config dir, fallback config dirs, packaged defaults.
    """
    primary = get_writable_config_dir()
    roots = [
        primary,
        get_cwd_config_dir(),
        get_user_config_dir(),
        get_package_config_dir(),
    ]
    return _unique_paths(roots)


def get_writable_config_dir() -> Path:
    """
    Return the preferred config root for writes.

    Prefer an existing cwd config; otherwise use the user config dir.
    """
    global _PRIMARY_CONFIG_DIR
    if _PRIMARY_CONFIG_DIR is not None:
        return _PRIMARY_CONFIG_DIR

    cwd_config = get_cwd_config_dir()
    if _is_writable_dir(cwd_config, create=False):
        _PRIMARY_CONFIG_DIR = cwd_config
        return cwd_config

    user_config = get_user_config_dir()
    if _is_writable_dir(user_config, create=True):
        _PRIMARY_CONFIG_DIR = user_config
        return user_config

    if cwd_config.exists():
        _PRIMARY_CONFIG_DIR = cwd_config
        return cwd_config

    _PRIMARY_CONFIG_DIR = user_config
    return user_config


def find_config_file(relative_path: Path) -> Optional[Path]:
    """Find a config file by searching all config roots in order."""
    for root in get_config_search_paths():
        candidate = root / relative_path
        if candidate.exists():
            return candidate
    return None
