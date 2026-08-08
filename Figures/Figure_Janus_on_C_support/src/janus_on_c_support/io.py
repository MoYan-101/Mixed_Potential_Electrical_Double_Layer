"""Traceable, strict, and non-overwriting result I/O helpers.

The numerical model intentionally does not own persistence.  This module is
the single place where a run directory is created and where JSON, CSV, NPZ,
source hashes, and artifact checksums are written.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import math
import platform
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib
import numpy as np
import scipy


CHECKSUM_FILENAME = "checksums.sha256"
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def json_safe(value: Any) -> Any:
    """Return a strictly JSON-native value and reject NaN/Infinity.

    NumPy values occur throughout the spectral pipeline.  Converting them in
    one audited function prevents accidental ``NaN`` tokens (which are not
    valid JSON) and keeps all metadata portable.
    """

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return json_safe(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Non-finite floating-point value is not JSON-safe: {value!r}")
        return value
    raise TypeError(f"Unsupported JSON value type: {type(value).__name__}")


def write_json(path: str | Path, value: Any) -> Path:
    """Write deterministic strict JSON, creating only the parent directory."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            json_safe(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def write_csv(
    path: str | Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> Path:
    """Write a headered UTF-8 CSV with a stable column order.

    CSV is used for point data and therefore permits blank cells and IEEE NaN
    for intentionally undefined C-region kinetics.  Strict-JSON rules apply
    only to metadata files.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="raise")
        writer.writeheader()
        for row in rows:
            converted: dict[str, Any] = {}
            for name in fieldnames:
                item = row.get(name, "")
                if isinstance(item, np.generic):
                    item = item.item()
                converted[name] = item
            writer.writerow(converted)
    return destination


def write_npz(path: str | Path, **arrays: Any) -> Path:
    """Write a compressed NumPy archive with deterministic array names."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        **{str(name): np.asarray(value) for name, value in sorted(arrays.items())},
    )
    return destination


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of one regular file."""

    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes(source_root: str | Path) -> dict[str, Any]:
    """Hash every Python source file and an aggregate ordered manifest."""

    root = Path(source_root).resolve()
    files = sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)
    per_file = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in files
    }
    aggregate = hashlib.sha256()
    for relative, digest in per_file.items():
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
    return {
        "root": str(root),
        "aggregate_sha256": aggregate.hexdigest(),
        "files": per_file,
    }


def environment_snapshot() -> dict[str, Any]:
    """Return the minimal environment required to reproduce a run."""

    return {
        "captured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "libraries": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }


def make_run_directory(
    output_root: str | Path,
    run_id: str | None = None,
) -> tuple[Path, str]:
    """Create and return a new run directory, refusing any overwrite."""

    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    identifier = run_id or datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    if not _RUN_ID_RE.fullmatch(identifier) or identifier in {".", ".."}:
        raise ValueError(
            "run_id must start with an alphanumeric character and contain only "
            "letters, digits, '.', '-', or '_'"
        )
    destination = root / identifier
    destination.mkdir(parents=False, exist_ok=False)
    return destination, identifier


def artifact_hashes(
    run_directory: str | Path,
    *,
    exclude: Iterable[str] = (CHECKSUM_FILENAME,),
) -> dict[str, str]:
    """Return ordered SHA-256 hashes for all regular run artifacts."""

    root = Path(run_directory).resolve()
    excluded = set(exclude)
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and path.relative_to(root).as_posix() not in excluded
    )
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in paths
    }


def write_checksums(run_directory: str | Path) -> tuple[Path, dict[str, str]]:
    """Write and immediately verify ``checksums.sha256``."""

    root = Path(run_directory).resolve()
    checksums = artifact_hashes(root)
    destination = root / CHECKSUM_FILENAME
    destination.write_text(
        "".join(f"{digest}  {relative}\n" for relative, digest in checksums.items()),
        encoding="utf-8",
    )
    for relative, expected in checksums.items():
        actual = sha256_file(root / relative)
        if actual != expected:
            raise RuntimeError(f"Checksum verification failed for {relative}")
    return destination, checksums


def relative_paths(paths: Iterable[str | Path], root: str | Path) -> list[str]:
    """Return POSIX paths relative to ``root``, rejecting escaped paths."""

    base = Path(root).resolve()
    relatives: list[str] = []
    for value in paths:
        path = Path(value).resolve()
        try:
            relative = path.relative_to(base)
        except ValueError as exc:
            raise ValueError(f"Artifact lies outside run directory: {path}") from exc
        relatives.append(relative.as_posix())
    return sorted(relatives)


__all__ = [
    "CHECKSUM_FILENAME",
    "artifact_hashes",
    "environment_snapshot",
    "json_safe",
    "make_run_directory",
    "relative_paths",
    "sha256_file",
    "source_hashes",
    "write_checksums",
    "write_csv",
    "write_json",
    "write_npz",
]
