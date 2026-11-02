"""Shared input, hashing, and manifest helpers."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

PREDICTION_KEYS = (
    "response", "output", "prediction", "text", "content", "completion",
    "answer", "generated", "model_output",
)
def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_value(value) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_jsonl(path: str | Path) -> list[dict]:
    records = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc
        if not isinstance(record, dict):
            raise TypeError(f"JSONL line {line_number} is not an object")
        records.append(record)
    return records


def sample_id(record: dict) -> str:
    sid = record.get("sample_id") or (record.get("rubric") or {}).get("sample_id")
    if not sid:
        raise ValueError("Benchmark record has no sample_id")
    return str(sid)


def source_html(record: dict, webvr_root: str | Path | None = None) -> str:
    resolved = record.get("_resolved_source_html")
    if isinstance(resolved, str):
        return resolved
    source = record.get("source_page") or {}
    if isinstance(source.get("html"), str):
        return source["html"]
    relative = source.get("html_path")
    if relative:
        if webvr_root is None:
            raise ValueError(
                f"{sample_id(record)} references WebVR HTML at {relative}; "
                "provide --webvr-root pointing to the downloaded BroAlanTaps/WebVR dataset"
            )
        root = Path(webvr_root).expanduser().resolve()
        path = (root / str(relative)).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Unsafe WebVR HTML path for {sample_id(record)}: {relative}") from exc
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing WebVR HTML for {sample_id(record)}: {path}"
            )
        return path.read_text(encoding="utf-8")
    raise ValueError(f"Could not resolve source HTML for {sample_id(record)}")


def extract_html(content: str) -> str:
    match = re.search(r"```html\s*(.*?)\s*```", content or "",
                      flags=re.DOTALL | re.IGNORECASE)
    if match:
        html = match.group(1).strip()
    else:
        lowered = (content or "").lower()
        start = lowered.find("<!doctype")
        if start < 0:
            start = lowered.find("<html")
        if start < 0:
            return ""
        html = content[start:].strip()
    end = html.lower().find("</html>")
    if end >= 0:
        return html[:end + len("</html>")]
    return html + "\n</html>" if "<html" in html.lower() else html


def _prediction_text(record: dict) -> str:
    for key in PREDICTION_KEYS:
        if record.get(key) is not None:
            return str(record[key])
    return ""


def load_predictions(path: str | Path) -> dict[str, str]:
    path = Path(path)
    output: dict[str, str] = {}
    if path.is_dir():
        for item in sorted(path.iterdir()):
            if item.suffix.lower() not in {".html", ".htm", ".txt", ".json"}:
                continue
            sid = item.stem
            if item.suffix.lower() == ".json":
                record = json.loads(item.read_text(encoding="utf-8"))
                if not isinstance(record, dict):
                    continue
                sid = str(record.get("sample_id", sid))
                text = _prediction_text(record)
            else:
                text = item.read_text(encoding="utf-8")
            if text.strip() or sid not in output:
                output[sid] = text
        return output

    raw = path.read_text(encoding="utf-8")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        sid = parsed.get("sample_id") or parsed.get("id") or parsed.get("sid")
        if sid is not None:
            return {str(sid): _prediction_text(parsed)}
        for sid, value in parsed.items():
            output[str(sid)] = value if isinstance(value, str) else _prediction_text(value)
        return output
    records = parsed if isinstance(parsed, list) else [
        json.loads(line) for line in raw.splitlines() if line.strip()
    ]
    for record in records:
        if not isinstance(record, dict):
            continue
        sid = record.get("sample_id") or record.get("id") or record.get("sid")
        if sid is None:
            continue
        sid = str(sid)
        text = _prediction_text(record)
        if text.strip() or not output.get(sid, "").strip():
            output[sid] = text
    return output


def path_hash(path: str | Path) -> str:
    path = Path(path)
    if path.is_file():
        return sha256_file(path)
    return sha256_value({
        item.name: sha256_file(item)
        for item in sorted(path.iterdir()) if item.is_file()
    })


def write_manifest(path: str | Path, *, track: str, benchmark: str | Path,
                   predictions: str | Path | None, config: dict) -> dict:
    source_root = Path(__file__).resolve().parent.parent
    implementation = {
        str(item.relative_to(source_root)): sha256_file(item)
        for item in sorted([*source_root.glob("*.py"), *source_root.glob("scorer/*.py")])
    }
    manifest = {
        "schema_version": "1.0",
        "track": track,
        "benchmark": {"name": Path(benchmark).name, "sha256": sha256_file(benchmark)},
        "predictions": None if predictions is None else {
            "name": Path(predictions).name,
            "sha256": path_hash(predictions),
        },
        "implementation_sha256": sha256_value(implementation),
        "config": config,
    }
    destination = Path(path)
    if destination.exists():
        previous = json.loads(destination.read_text(encoding="utf-8"))
        if previous != manifest:
            raise ValueError("Run inputs changed; use a new output directory")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
    return manifest
