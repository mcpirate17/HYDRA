import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_DEFAULT_CACHE_PATH = "~/.cache/hydra/la3_tuning_cache.json"


@dataclass(frozen=True)
class _CacheKey:
    kind: str
    key: str


_CACHE: dict[str, dict[str, Any]] | None = None


def _enabled() -> bool:
    return os.getenv("HYDRA_LA3_TUNING_CACHE", "1") not in ("0", "", "false", "False")


def _cache_path() -> Path:
    return Path(os.path.expanduser(os.getenv("HYDRA_LA3_TUNING_CACHE_PATH", _DEFAULT_CACHE_PATH)))


def _ensure_loaded() -> dict[str, dict[str, Any]]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    if not _enabled():
        _CACHE = {}
        return _CACHE

    path = _cache_path()
    try:
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            _CACHE = {k: v for k, v in data.items() if isinstance(v, dict)}
        else:
            _CACHE = {}
    except FileNotFoundError:
        _CACHE = {}
    except Exception:
        _CACHE = {}

    return _CACHE


def _atomic_write_json(path: Path, data: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    tmp.replace(path)


def get(kind: str, key: str) -> Any | None:
    cache = _ensure_loaded()
    bucket = cache.get(kind)
    if not isinstance(bucket, dict):
        return None
    return bucket.get(key)


def set(kind: str, key: str, value: Any) -> None:
    if not _enabled():
        return
    cache = _ensure_loaded()
    bucket = cache.get(kind)
    if not isinstance(bucket, dict):
        bucket = {}
        cache[kind] = bucket

    # Only write if it changes something.
    if bucket.get(key) == value:
        return
    bucket[key] = value

    try:
        _atomic_write_json(_cache_path(), cache)
    except Exception:
        # Best-effort only.
        return


def make_key(*parts: Any) -> str:
    # Stable, human-readable.
    return "|".join(str(p) for p in parts)
