#!/usr/bin/env python3
"""Minimal Track A/B prediction example for video, frames, and audiovisual input."""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import subprocess
import tempfile
import time
from contextlib import suppress
from pathlib import Path

import requests
from dotenv import load_dotenv

from prediction_prompts import INPUT_MODES, build_track_a_prompt, build_track_b_prompt
from scorer.io_utils import load_jsonl, sample_id, source_html

OFFICIAL_API_HOST = "https://generativelanguage.googleapis.com"
DEFAULT_API_ROOT = f"{OFFICIAL_API_HOST}/v1beta"
DEFAULT_MODEL = "gemini-3.1-pro-preview"


class GeminiPredictor:
    def __init__(self, model: str, api_key: str | None = None, timeout: int = 600):
        self.model = model
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "").strip()
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required")
        self.api_root = os.getenv("GEMINI_API_ROOT", DEFAULT_API_ROOT).rstrip("/")
        self.stream = os.getenv("GEMINI_API_STREAM", "").strip().lower() in {
            "1", "true", "yes", "on"
        }
        self.timeout = timeout

    def _request(self, method: str, url: str, **kwargs) -> requests.Response:
        last_error = "request failed"
        for attempt in range(4):
            try:
                body = kwargs.get("data")
                if hasattr(body, "seek"):
                    body.seek(0)
                response = requests.request(method, url, timeout=self.timeout, **kwargs)
                if response.status_code < 400:
                    return response
                last_error = f"HTTP {response.status_code}: {response.text[:500]}"
                if response.status_code not in {408, 409, 425, 429, 500, 502, 503, 504}:
                    break
            except requests.RequestException as exc:
                last_error = f"{type(exc).__name__}: request failed"
            if attempt < 3:
                time.sleep(2 ** attempt)
        raise RuntimeError(last_error)

    def upload_video(self, path: Path) -> dict:
        size = path.stat().st_size
        start = self._request(
            "POST",
            f"{OFFICIAL_API_HOST}/upload/v1beta/files",
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
                "X-Goog-Upload-Protocol": "resumable",
                "X-Goog-Upload-Command": "start",
                "X-Goog-Upload-Header-Content-Length": str(size),
                "X-Goog-Upload-Header-Content-Type": "video/mp4",
            },
            json={"file": {"display_name": path.name}},
        )
        upload_url = start.headers.get("X-Goog-Upload-URL")
        if not upload_url:
            raise RuntimeError("Gemini File API did not return an upload URL")
        with path.open("rb") as handle:
            uploaded = self._request(
                "POST",
                upload_url,
                headers={
                    "Content-Length": str(size),
                    "X-Goog-Upload-Offset": "0",
                    "X-Goog-Upload-Command": "upload, finalize",
                },
                data=handle,
            ).json()["file"]
        name = uploaded["name"]
        while uploaded.get("state") == "PROCESSING":
            time.sleep(2)
            uploaded = self._request(
                "GET", f"{OFFICIAL_API_HOST}/v1beta/{name}",
                headers={"x-goog-api-key": self.api_key},
            ).json()
        if uploaded.get("state") == "FAILED":
            raise RuntimeError(f"Gemini could not process {path.name}")
        return uploaded

    def delete_file(self, name: str) -> None:
        self._request(
            "DELETE", f"{OFFICIAL_API_HOST}/v1beta/{name}",
            headers={"x-goog-api-key": self.api_key},
        )

    def generate(self, prompt: str, media_parts: list[dict],
                 max_output_tokens: int = 0) -> dict:
        config = {}
        if max_output_tokens > 0:
            config["maxOutputTokens"] = max_output_tokens
        method = "streamGenerateContent?alt=sse" if self.stream else "generateContent"
        response = self._request(
            "POST",
            f"{self.api_root}/models/{self.model}:{method}",
            headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
            json={
                "contents": [{"role": "user", "parts": [*media_parts, {"text": prompt}]}],
                "generationConfig": config,
            },
            stream=self.stream,
        )
        if self.stream:
            events = []
            for raw_line in response.iter_lines(decode_unicode=True):
                line = (raw_line or "").strip()
                if line.startswith("data:") and line[5:].strip() != "[DONE]":
                    events.append(json.loads(line[5:].strip()))
        else:
            events = [response.json()]
        text = "".join(
            part.get("text", "")
            for event in events
            for candidate in event.get("candidates", [])
            for part in candidate.get("content", {}).get("parts", [])
            if not part.get("thought")
        )
        if not text.strip():
            raise RuntimeError(f"Gemini returned no text: {json.dumps(events[-1])[:500]}")
        usage = next(
            (event["usageMetadata"] for event in reversed(events) if event.get("usageMetadata")),
            None,
        )
        return {"response": text, "usage": usage}


def run_ffmpeg(arguments: list[str]) -> None:
    try:
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *arguments],
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required for VL input preparation") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("ffmpeg failed while preparing VL input") from exc


def make_silent_video(source: Path, destination: Path) -> None:
    run_ffmpeg(["-i", str(source), "-map", "0:v:0", "-c:v", "copy", "-an", str(destination)])


def make_frames(source: Path, destination: Path, fps: float, width: int) -> list[Path]:
    from PIL import Image, ImageDraw, ImageFont

    destination.mkdir(parents=True, exist_ok=True)
    source_frames = destination / "source"
    source_frames.mkdir()
    run_ffmpeg([
        "-i", str(source), "-vf", f"fps={fps},scale=1280:-2:flags=lanczos",
        str(source_frames / "frame_%04d.png"),
    ])
    originals = sorted(source_frames.glob("frame_*.png"))
    if not originals:
        raise RuntimeError("ffmpeg produced no frames")
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            max(14, int(width * 0.42) // 18),
        )
    except OSError:
        font = ImageFont.load_default()
    frames = []
    for index, original in enumerate(originals, 1):
        with Image.open(original) as raw:
            image = raw.convert("RGB")
            height = round(width * image.height / image.width)
            canvas = image.resize((width, height), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(canvas)
        draw.rectangle(
            [0, 0, int(width * 0.42 * 0.42), int(width * 0.10 * 0.42)],
            fill=(0, 0, 0),
        )
        draw.text((4, 2), f"t={index / fps:g}s", fill=(255, 255, 0), font=font)
        output = destination / f"frame_{index:04d}.webp"
        canvas.save(output, format="WEBP", lossless=True)
        canvas.close()
        frames.append(output)
    return frames


def inline_images(paths: list[Path]) -> list[dict]:
    return [
        {
            "inlineData": {
                "mimeType": "image/webp",
                "data": base64.b64encode(path.read_bytes()).decode("ascii"),
            }
        }
        for path in paths
    ]


def inline_video(path: Path) -> dict:
    return {
        "inlineData": {
            "mimeType": "video/mp4",
            "data": base64.b64encode(path.read_bytes()).decode("ascii"),
        }
    }


def resolve_video(record: dict, benchmark_path: Path) -> Path:
    root = benchmark_path.resolve().parent
    path = (root / record["video"]).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Unsafe video path for {sample_id(record)}") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def build_prompt(record: dict, track: str, input_mode: str, html: str,
                 n_frames: int = 0) -> str:
    values = {
        "transcript": record["transcript"],
        "input_mode": input_mode,
        "n_frames": n_frames,
    }
    if track == "a":
        return build_track_a_prompt(html=html, **values)
    return build_track_b_prompt(**values)


def selected_records(records: list[dict], requested: list[str], limit: int) -> list[dict]:
    if requested:
        wanted = set(requested)
        selected = [record for record in records if sample_id(record) in wanted]
        missing = wanted - {sample_id(record) for record in selected}
        if missing:
            raise ValueError(f"Unknown sample IDs: {sorted(missing)}")
        return selected
    return records if limit == 0 else records[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--webvr-root",
                        help="Downloaded WebVR directory; required for Track A")
    parser.add_argument("--track", choices=["a", "b"], required=True)
    parser.add_argument("--input-mode", choices=INPUT_MODES, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", default="predictions.jsonl")
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=1,
                        help="Number of records to run; 0 runs all records")
    parser.add_argument("--frame-fps", type=float, default=1.0)
    parser.add_argument("--frame-width", type=int, default=644)
    parser.add_argument("--max-output-tokens", type=int, default=0)
    parser.add_argument("--inline-video", action="store_true",
                        help="Inline video bytes instead of using the Gemini File API")
    parser.add_argument("--dry-run", action="store_true",
                        help="Write prompts without calling the model")
    args = parser.parse_args()
    if args.limit < 0 or args.frame_fps <= 0 or args.frame_width <= 0:
        parser.error("limit, frame-fps, and frame-width must be non-negative/positive")
    if args.track == "a" and not args.webvr_root:
        parser.error("--webvr-root is required for Track A")

    load_dotenv()
    benchmark_path = Path(args.benchmark)
    records = selected_records(load_jsonl(benchmark_path), args.sample_id, args.limit)
    client = None if args.dry_run else GeminiPredictor(args.model)
    output = Path(args.output)
    with output.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records, 1):
            sid = sample_id(record)
            html = source_html(record, args.webvr_root) if args.track == "a" else ""
            video = resolve_video(record, benchmark_path)
            with tempfile.TemporaryDirectory(prefix=f"deicticedit-{sid}-") as tmp:
                temporary = Path(tmp)
                remote_file = None
                try:
                    if args.input_mode == "vl-frames":
                        if args.dry_run:
                            duration = float(record["metadata"]["video_duration_sec"])
                            n_frames = max(1, math.ceil(duration * args.frame_fps))
                            media_parts = []
                        else:
                            frames = make_frames(
                                video, temporary / "frames", args.frame_fps, args.frame_width
                            )
                            n_frames = len(frames)
                            media_parts = inline_images(frames)
                    else:
                        n_frames = 0
                        media_path = video
                        if args.input_mode == "vl-video" and not args.dry_run:
                            media_path = temporary / "silent.mp4"
                            make_silent_video(video, media_path)
                        if args.dry_run:
                            media_parts = []
                        elif args.inline_video:
                            media_parts = [inline_video(media_path)]
                        else:
                            remote_file = client.upload_video(media_path)
                            media_parts = [{"fileData": {
                                "mimeType": remote_file.get("mimeType", "video/mp4"),
                                "fileUri": remote_file["uri"],
                            }}]
                    prompt = build_prompt(record, args.track, args.input_mode, html, n_frames)
                    if args.dry_run:
                        result = {"prompt": prompt}
                    else:
                        result = client.generate(prompt, media_parts, args.max_output_tokens)
                    row = {
                        "sample_id": sid,
                        "track": args.track.upper(),
                        "model": args.model,
                        "input_mode": args.input_mode,
                        "ok": True,
                        **result,
                    }
                except Exception as exc:  # noqa: BLE001
                    row = {
                        "sample_id": sid,
                        "track": args.track.upper(),
                        "model": args.model,
                        "input_mode": args.input_mode,
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                finally:
                    if remote_file:
                        with suppress(Exception):
                            client.delete_file(remote_file["name"])
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                print(f"[{index}/{len(records)}] {sid}: {'ok' if row['ok'] else 'failed'}")


if __name__ == "__main__":
    main()
