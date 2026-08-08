"""Small, deterministic I/O helpers for the Hutchings figure collection."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np


CHECKSUM_FILENAME = "checksums.sha256"


def jsonable(value: Any) -> Any:
    """Convert common scientific-Python values to strict JSON values."""

    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Value of type {type(value).__name__} is not JSON serializable")


def write_json(path: str | Path, value: Any) -> Path:
    """Write one UTF-8, strictly finite JSON document."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            jsonable(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return target


def write_csv_rows(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> Path:
    """Write records with a deterministic union of fields."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        ordered: list[str] = []
        for row in rows:
            for key in row:
                name = str(key)
                if name not in ordered:
                    ordered.append(name)
        fieldnames = ordered
    names = list(fieldnames)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    name: jsonable(row.get(name))
                    if row.get(name) is not None
                    else ""
                    for name in names
                }
            )
    return target


def write_npz(path: str | Path, arrays: Mapping[str, Any]) -> Path:
    """Write a compressed NPZ without object arrays or implicit pickle data."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    clean: dict[str, np.ndarray] = {}
    for raw_name, raw_value in arrays.items():
        name = str(raw_name)
        value = np.asarray(raw_value)
        if value.dtype == object:
            value = value.astype(str)
        clean[name] = value
    if not clean:
        clean["empty"] = np.empty(0, dtype=float)
    np.savez_compressed(target, **clean)
    return target


def sha256_file(path: str | Path) -> str:
    target = Path(path)
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_files(root: str | Path, *, exclude: Iterable[str] = ()) -> list[Path]:
    """Return sorted, non-symlink files below ``root``."""

    base = Path(root)
    excluded = {str(item) for item in exclude}
    files: list[Path] = []
    for path in base.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"Generated artifacts must not be symlinks: {path}")
        if path.is_file():
            relative = path.relative_to(base).as_posix()
            if relative not in excluded and path.name not in excluded:
                files.append(path)
    return sorted(files, key=lambda item: item.relative_to(base).as_posix())


def artifact_hashes(
    root: str | Path,
    *,
    exclude: Iterable[str] = (),
) -> dict[str, str]:
    base = Path(root)
    return {
        path.relative_to(base).as_posix(): sha256_file(path)
        for path in regular_files(base, exclude=exclude)
    }


def write_checksums(
    root: str | Path,
    *,
    filename: str = CHECKSUM_FILENAME,
) -> tuple[Path, dict[str, str]]:
    """Hash every generated file except the checksum listing itself."""

    base = Path(root)
    checksums = artifact_hashes(base, exclude=(filename,))
    target = base / filename
    target.write_text(
        "".join(f"{digest}  {relative}\n" for relative, digest in checksums.items()),
        encoding="utf-8",
    )
    return target, checksums


def verify_checksums(
    root: str | Path,
    *,
    filename: str = CHECKSUM_FILENAME,
    require_complete: bool = True,
) -> dict[str, str]:
    """Verify a SHA-256 listing and optionally its complete file coverage."""

    base = Path(root)
    checksum_path = base / filename
    if not checksum_path.is_file():
        raise FileNotFoundError(f"Missing checksum file: {checksum_path}")
    expected: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        checksum_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as exc:
            raise ValueError(
                f"Malformed checksum line {line_number} in {checksum_path}"
            ) from exc
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(
                f"Invalid SHA-256 digest on line {line_number} in {checksum_path}"
            )
        if relative in expected:
            raise ValueError(f"Duplicate checksum entry: {relative}")
        expected[relative] = digest
    for relative, digest in expected.items():
        candidate = base / relative
        try:
            candidate.resolve().relative_to(base.resolve())
        except ValueError as exc:
            raise ValueError(f"Checksum entry escapes run root: {relative}") from exc
        if not candidate.is_file() or candidate.is_symlink():
            raise FileNotFoundError(f"Missing or invalid checksummed artifact: {candidate}")
        actual = sha256_file(candidate)
        if actual != digest:
            raise RuntimeError(
                f"Checksum mismatch for {relative}: expected {digest}, got {actual}"
            )
    if require_complete:
        actual_paths = {
            path.relative_to(base).as_posix()
            for path in regular_files(base, exclude=(filename,))
        }
        if actual_paths != set(expected):
            missing = sorted(actual_paths - set(expected))
            stale = sorted(set(expected) - actual_paths)
            raise RuntimeError(
                "Checksum coverage mismatch: "
                f"unlisted={missing!r}, stale={stale!r}"
            )
    return expected


def assert_targets_absent(output_root: str | Path, names: Iterable[str]) -> None:
    """Refuse a publication if any generated target already exists."""

    root = Path(output_root)
    conflicts = [root / str(name) for name in names if (root / str(name)).exists()]
    if conflicts:
        joined = ", ".join(str(path) for path in conflicts)
        raise FileExistsError(f"Refusing to overwrite existing collection target(s): {joined}")


@contextmanager
def staging_directory(output_root: str | Path) -> Iterator[Path]:
    """Create a sibling staging directory and remove it on failure."""

    destination = Path(output_root).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-",
            dir=destination.parent,
        )
    ).resolve()
    if stage.parent != destination.parent:
        raise RuntimeError(f"Unexpected staging location: {stage}")
    try:
        yield stage
    finally:
        if stage.exists():
            if stage.is_symlink() or stage.parent != destination.parent:
                raise RuntimeError(f"Refusing unsafe staging cleanup: {stage}")
            shutil.rmtree(stage)


def publish_staged_children(
    staging_root: str | Path,
    output_root: str | Path,
    names: Sequence[str],
) -> list[Path]:
    """Publish validated children with rollback if a rename fails."""

    stage = Path(staging_root).resolve()
    destination = Path(output_root).expanduser().resolve()
    if stage.parent != destination.parent:
        raise ValueError("Staging and output roots must share a parent filesystem")
    destination.mkdir(parents=True, exist_ok=True)
    assert_targets_absent(destination, names)
    published: list[Path] = []
    try:
        for name in names:
            source = stage / name
            target = destination / name
            if not source.exists() or source.is_symlink():
                raise FileNotFoundError(f"Missing staged publication target: {source}")
            os.replace(source, target)
            published.append(target)
    except Exception:
        for target in reversed(published):
            source = stage / target.name
            if target.exists() and not source.exists():
                os.replace(target, source)
        raise
    return published


__all__ = [
    "CHECKSUM_FILENAME",
    "artifact_hashes",
    "assert_targets_absent",
    "jsonable",
    "publish_staged_children",
    "regular_files",
    "sha256_file",
    "staging_directory",
    "verify_checksums",
    "write_checksums",
    "write_csv_rows",
    "write_json",
    "write_npz",
]
