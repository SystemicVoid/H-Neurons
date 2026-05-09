"""Shared artifact IO helpers for claim-bearing JSON, JSONL, and alpha panels."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib
import json
from pathlib import Path
from typing import Any

from utils import format_alpha_label

JsonDict = dict[str, Any]
PathLike = str | Path


class JsonlFormatError(ValueError):
    """Raised when a JSONL artifact violates the selected row policy."""


def alpha_jsonl_filename(alpha: float | str) -> str:
    return f"alpha_{format_alpha_label(float(alpha))}.jsonl"


def alpha_jsonl_path(output_dir: PathLike, alpha: float | str) -> Path:
    return Path(output_dir) / alpha_jsonl_filename(alpha)


def file_sha256(path: PathLike) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json_dumps(
    payload: Any,
    *,
    pretty: bool = False,
    ensure_ascii: bool = False,
    sort_keys: bool = True,
    trailing_newline: bool = True,
) -> str:
    if pretty:
        text = json.dumps(
            payload,
            ensure_ascii=ensure_ascii,
            indent=2,
            sort_keys=sort_keys,
        )
    else:
        text = json.dumps(
            payload,
            ensure_ascii=ensure_ascii,
            sort_keys=sort_keys,
        )
    return f"{text}\n" if trailing_newline else text


def write_canonical_json(
    path: PathLike,
    payload: Any,
    *,
    pretty: bool = True,
    ensure_ascii: bool = False,
    sort_keys: bool = True,
) -> None:
    Path(path).write_text(
        canonical_json_dumps(
            payload,
            pretty=pretty,
            ensure_ascii=ensure_ascii,
            sort_keys=sort_keys,
        ),
        encoding="utf-8",
    )


def _require_json_object(payload: Any, *, path: Path, line_number: int) -> JsonDict:
    if not isinstance(payload, dict):
        raise JsonlFormatError(f"{path}:{line_number}: JSONL row must be an object")
    return dict(payload)


def read_jsonl_strict(path: PathLike, *, missing_ok: bool = False) -> list[JsonDict]:
    jsonl_path = Path(path)
    if missing_ok and not jsonl_path.exists():
        return []

    rows: list[JsonDict] = []
    with jsonl_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise JsonlFormatError(
                    f"{jsonl_path}:{line_number}: invalid JSONL row: {exc.msg}"
                ) from exc
            rows.append(
                _require_json_object(
                    payload,
                    path=jsonl_path,
                    line_number=line_number,
                )
            )
    return rows


def read_jsonl_recovery(path: PathLike, *, missing_ok: bool = False) -> list[JsonDict]:
    jsonl_path = Path(path)
    if missing_ok and not jsonl_path.exists():
        return []

    rows: list[JsonDict] = []
    buffer = ""
    start_line = 1
    for line_number, line in enumerate(
        jsonl_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip() and not buffer:
            continue
        if not buffer:
            start_line = line_number
            buffer = line
        else:
            buffer = f"{buffer}\n{line}"
        try:
            payload = json.loads(buffer, strict=False)
        except json.JSONDecodeError:
            continue
        rows.append(
            _require_json_object(payload, path=jsonl_path, line_number=start_line)
        )
        buffer = ""

    if buffer.strip():
        try:
            payload = json.loads(buffer, strict=False)
        except json.JSONDecodeError as exc:
            raise JsonlFormatError(
                f"{jsonl_path}:{start_line}: unrecoverable JSONL row: {exc.msg}"
            ) from exc
        rows.append(
            _require_json_object(payload, path=jsonl_path, line_number=start_line)
        )
    return rows


def jsonl_row_dumps(
    row: Mapping[str, Any],
    *,
    ensure_ascii: bool = False,
    sort_keys: bool = True,
) -> str:
    return json.dumps(row, ensure_ascii=ensure_ascii, sort_keys=sort_keys)


def write_jsonl_rows(
    path: PathLike,
    rows: Iterable[Mapping[str, Any]],
    *,
    ensure_ascii: bool = False,
    sort_keys: bool = True,
) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                jsonl_row_dumps(row, ensure_ascii=ensure_ascii, sort_keys=sort_keys)
                + "\n"
            )


def append_jsonl_row(
    path: PathLike,
    row: Mapping[str, Any],
    *,
    ensure_ascii: bool = False,
    sort_keys: bool = True,
) -> None:
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(
            jsonl_row_dumps(row, ensure_ascii=ensure_ascii, sort_keys=sort_keys) + "\n"
        )


def row_sha256(row: Mapping[str, Any]) -> str:
    payload = json.dumps(
        row,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return text_sha256(payload)


def jsonl_rows_sha256(rows: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        payload = json.dumps(
            row,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest.update(payload.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def row_id(row: Mapping[str, Any], *, field: str = "id") -> str:
    if field not in row:
        raise KeyError(f"row is missing identity field {field!r}")
    return str(row[field])
