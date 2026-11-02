#!/usr/bin/env python3
"""Score Track A HTML predictions with the official EFS pipeline."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from scorer import final_state
from scorer.io_utils import (
    extract_html,
    load_jsonl,
    load_predictions,
    sample_id,
    source_html,
    write_manifest,
)
from scorer.llm_client import DEFAULT_JUDGE_MODEL

ROOT = Path(__file__).resolve().parent
CATEGORIES = ("target_grounding", "edit_fulfillment", "no_damage", "revision")


def safe_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]", "_", value).strip("._")
    return name or "model"


def load_benchmark(path: str | Path, webvr_root: str | Path | None = None) -> list[dict]:
    records = load_jsonl(path)
    seen = set()
    html_cache: dict[str, str] = {}
    for record in records:
        sid = sample_id(record)
        if sid in seen:
            raise ValueError(f"Duplicate benchmark sample_id: {sid}")
        seen.add(sid)
        rubric = record.get("rubric")
        if not isinstance(rubric, dict):
            raise TypeError(f"{sid}: missing rubric")
        final_state.validate_view(rubric)
        webvr_path = str((record.get("source_page") or {}).get("html_path", ""))
        if webvr_path and webvr_path in html_cache:
            record["_resolved_source_html"] = html_cache[webvr_path]
        else:
            record["_resolved_source_html"] = source_html(record, webvr_root)
            if webvr_path:
                html_cache[webvr_path] = record["_resolved_source_html"]
    return records


def build_organized(benchmark: list[dict], predictions: dict[str, str], output_dir: Path,
                    model_name: str) -> dict:
    organized = output_dir / "organized"
    organized.mkdir(parents=True, exist_ok=True)
    ids = []
    no_prediction = []
    parse_failed = []
    for record in benchmark:
        sid = sample_id(record)
        ids.append(sid)
        sample_dir = organized / sid
        (sample_dir / "source_page").mkdir(parents=True, exist_ok=True)
        (sample_dir / "source_page" / "index.html").write_text(
            source_html(record), encoding="utf-8"
        )
        (sample_dir / "judge_rubric.json").write_text(
            json.dumps(record["rubric"], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        raw = predictions.get(sid, "")
        if not raw.strip():
            no_prediction.append(sid)
            continue
        html = extract_html(raw)
        if not html:
            parse_failed.append(sid)
            continue
        prediction_dir = organized / "predictions" / model_name / sid / "modified_page"
        prediction_dir.mkdir(parents=True, exist_ok=True)
        (prediction_dir.parent / "model_response.txt").write_text(raw, encoding="utf-8")
        (prediction_dir / "index.html").write_text(html, encoding="utf-8")

    stats = {
        "n_samples": len(ids),
        "sample_ids": ids,
        "n_predictions_written": len(ids) - len(no_prediction) - len(parse_failed),
        "no_prediction": no_prediction,
        "html_parse_failed": parse_failed,
    }
    (output_dir / "build_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return stats


def _run_stage(arguments: list[str], log_path: Path, label: str) -> None:
    print(f"[{label}] running", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.run(
            [sys.executable, *arguments], cwd=ROOT, stdout=log,
            stderr=subprocess.STDOUT, check=False,
        )
    if process.returncode:
        raise RuntimeError(f"{label} failed; see {log_path.name}")


def run_judges(organized: Path, model_name: str, judge_model: str,
               workers: int, resume: bool, log_path: Path) -> None:
    common = ["--organized-root", str(organized), "--scoring-policy", "final_state"]
    resume_args = ["--resume"] if resume else []
    _run_stage([
        "-m", "scorer.step1_rubric_judge", *common,
        "--contestant-models", model_name, "--judge-model", judge_model,
        "--num-workers", str(workers), "--prompt-lang", "en",
        "--output", str(organized / "judge_results_rubric.json"), *resume_args,
    ], log_path, "rubric judge")
    _run_stage([
        "-m", "scorer.step2a_replace_image", *common,
        "--contestant-models", model_name, "--prompt-lang", "en",
    ], log_path, "replace-image verification")
    _run_stage([
        "-m", "scorer.step2b_image_content", *common,
        "--judge-model", judge_model, "--num-workers", str(workers),
        "--prompt-lang", "en",
        "--output", str(organized / "image_content_verification.json"),
    ], log_path, "image-content verification")
    _run_stage([
        "-m", "scorer.step3_no_damage", *common,
        "--contestant-models", model_name, "--judge-model", judge_model,
        "--num-workers", str(workers), "--prompt-lang", "en",
        "--output", str(organized / "no_damage_results_v3.json"),
    ], log_path, "No-damage judge")
    _run_stage([
        "-m", "scorer.step4_efs", *common,
        "--contestant-models", model_name,
        "--judge-results", "judge_results_rubric.json",
        "--image-content", "image_content_verification.json",
        "--no-damage", "no_damage_results_v3.json",
        "--output", "judge_results_v3.json",
    ], log_path, "EFS aggregation")


def summarize(output_dir: Path, stats: dict, model_name: str) -> dict:
    judged_path = output_dir / "organized" / "judge_results_v3.json"
    judged = {
        row["sample_id"]: row
        for row in json.loads(judged_path.read_text(encoding="utf-8"))
        if row.get("contestant_model") == model_name
    }
    missing = set(stats["no_prediction"]) | set(stats["html_parse_failed"])
    results = []
    efs_values = []
    category_values: dict[str, list[float]] = defaultdict(list)
    judge_errors = []

    for sid in stats["sample_ids"]:
        if sid in missing:
            reason = "no_prediction" if sid in stats["no_prediction"] else "html_parse_failed"
            row = {
                "sample_id": sid,
                "contestant_model": model_name,
                "evaluation_status": "zero_filled",
                "zero_fill_reason": reason,
                "efs_weighted_v3": 0.0,
                "category_scores_v3": {category: 0.0 for category in CATEGORIES},
            }
        else:
            row = judged.get(sid)
            if not row or row.get("evaluation_status") != "scored" or row.get("efs_weighted_v3") is None:
                judge_errors.append(sid)
                results.append({
                    "sample_id": sid,
                    "contestant_model": model_name,
                    "evaluation_status": "judge_error",
                })
                continue
        results.append(row)
        efs_values.append(float(row["efs_weighted_v3"]))
        for category, value in (row.get("category_scores_v3") or {}).items():
            if category in CATEGORIES and value is not None:
                category_values[category].append(float(value))

    summary = {
        "track": "A",
        "metric": "EFS",
        "scoring_policy": "final_state",
        "contestant_model": model_name,
        "n_samples_in_benchmark": stats["n_samples"],
        "n_scored": len(efs_values),
        "n_judge_failed": len(judge_errors),
        "n_scored_zero_for_missing": len(missing),
        "efs_weighted_v3": sum(efs_values) / len(efs_values) if efs_values else None,
        "category_scores": {
            category: (sum(category_values.get(category, [])) /
                       len(category_values[category]) if category_values.get(category) else None)
            for category in CATEGORIES
        },
        "judge_failed_samples": judge_errors,
    }
    (output_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def score_track_a(*, benchmark_path: str | Path, predictions_path: str | Path,
                  model_name: str, output_dir: str | Path,
                  webvr_root: str | Path | None = None,
                  judge_model: str = DEFAULT_JUDGE_MODEL, workers: int = 8,
                  resume: bool = False, skip_judge: bool = False) -> dict | None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = safe_name(model_name)
    benchmark = load_benchmark(benchmark_path, webvr_root)
    predictions = load_predictions(predictions_path)
    write_manifest(
        output_dir / "run_manifest.json", track="A", benchmark=benchmark_path,
        predictions=predictions_path,
        config={
            "scoring_policy": "final_state",
            "metric": "EFS",
            "judge_provider": "Google Gemini API",
            "judge_model": judge_model,
            "judge_prompt_language": "en",
            "missing_prediction_policy": "zero",
            "judge_failure_policy": "exclude",
            "model_name": model_name,
            "n_samples": len(benchmark),
            "sample_ids_sha256": final_state.digest([sample_id(row) for row in benchmark]),
        },
    )
    stats = build_organized(benchmark, predictions, output_dir, model_name)
    if not skip_judge:
        run_judges(output_dir / "organized", model_name, judge_model, workers, resume,
                   output_dir / "scoring.log")
    if not (output_dir / "organized" / "judge_results_v3.json").exists():
        return None
    return summarize(output_dir, stats, model_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score DeicticEdit-Bench Track A HTML outputs.")
    parser.add_argument("--benchmark", "--jsonl", dest="benchmark", required=True)
    parser.add_argument("--predictions", required=True,
                        help="JSONL/JSON predictions or a directory of HTML files")
    parser.add_argument(
        "--webvr-root",
        help="Downloaded BroAlanTaps/WebVR directory; required when HTML is referenced by path",
    )
    parser.add_argument("--model-name", default="model")
    parser.add_argument("--output-dir", "--work-dir", dest="output_dir", default="track_a_results")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--workers", "--num-workers", dest="workers", type=int, default=8)
    parser.add_argument("--resume", "--resume-judge", dest="resume", action="store_true")
    parser.add_argument("--skip-judge", action="store_true",
                        help="Aggregate existing judge files without making API calls")
    args = parser.parse_args()
    summary = score_track_a(
        benchmark_path=args.benchmark,
        predictions_path=args.predictions,
        model_name=args.model_name,
        output_dir=args.output_dir,
        webvr_root=args.webvr_root,
        judge_model=args.judge_model,
        workers=args.workers,
        resume=args.resume,
        skip_judge=args.skip_judge,
    )
    if summary is None:
        print("Inputs prepared. Run again without --skip-judge to produce scores.")
    else:
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
