#!/usr/bin/env python3
"""Score Track B instruction rewrites with Component-F1 IRS."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from run_eval import load_benchmark, safe_name
from scorer.final_state import digest
from scorer.io_utils import load_predictions, sample_id, source_html, write_manifest
from scorer.llm_client import DEFAULT_JUDGE_MODEL, make_judge_client
from scorer.step_rewrite_judge import (
    judge_rewrite,
    reference_steps_from_rubric,
    zero_scores,
)


def summarize(rows: list[dict], model_name: str, n_benchmark: int,
              output_dir: Path) -> dict:
    scored = [row for row in rows if row.get("evaluation_status") != "judge_error"]
    judge_errors = [row["sample_id"] for row in rows
                    if row.get("evaluation_status") == "judge_error"]

    def mean(key: str):
        values = [float(row[key]) for row in scored]
        return sum(values) / len(values) if values else None

    summary = {
        "track": "B",
        "metric": "Component-F1 IRS",
        "contestant_model": model_name,
        "n_samples_in_benchmark": n_benchmark,
        "n_scored": len(scored),
        "n_judge_failed": len(judge_errors),
        "n_scored_zero_for_missing": sum(
            row.get("zero_fill_reason") == "no_prediction" for row in scored
        ),
        "n_rewrite_parse_failed": sum(
            row.get("zero_fill_reason") == "rewrite_parse_failed" for row in scored
        ),
        "target_f1": mean("target_f1"),
        "action_f1": mean("action_f1"),
        "irs": mean("irs"),
        "joint_f1": mean("joint_f1"),
        "n_reference_steps": sum(row["n_reference_steps"] for row in scored),
        "n_predicted_steps": sum(row["n_predicted_steps"] for row in scored),
        "judge_failed_samples": judge_errors,
    }
    (output_dir / "results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def score_track_b(*, benchmark_path: str | Path, predictions_path: str | Path,
                  model_name: str, output_dir: str | Path,
                  webvr_root: str | Path | None = None,
                  judge_model: str = DEFAULT_JUDGE_MODEL, workers: int = 8,
                  resume: bool = False) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = safe_name(model_name)
    benchmark = load_benchmark(benchmark_path, webvr_root)
    predictions = load_predictions(predictions_path)
    write_manifest(
        output_dir / "run_manifest.json", track="B", benchmark=benchmark_path,
        predictions=predictions_path,
        config={
            "metric": "Component-F1 IRS",
            "target_formula": "2*TP_target/(n_reference+n_predicted)",
            "action_formula": "2*TP_action/(n_reference+n_predicted)",
            "irs_formula": "0.5*target_f1+0.5*action_f1",
            "averaging": "instance_macro",
            "judge_provider": "Google Gemini API",
            "judge_model": judge_model,
            "judge_prompt_language": "en",
            "missing_prediction_policy": "zero",
            "judge_failure_policy": "exclude",
            "model_name": model_name,
            "n_samples": len(benchmark),
            "sample_ids_sha256": digest([sample_id(row) for row in benchmark]),
        },
    )
    result_path = output_dir / "results.json"
    existing = {}
    if resume and result_path.exists():
        for row in json.loads(result_path.read_text(encoding="utf-8")):
            if row.get("evaluation_status") != "judge_error":
                existing[row["sample_id"]] = row

    client = None
    rows = dict(existing)
    tasks = []
    for record in benchmark:
        sid = sample_id(record)
        if sid in rows:
            continue
        n_reference = len(reference_steps_from_rubric(record["rubric"]))
        prediction = predictions.get(sid, "")
        if not prediction.strip():
            rows[sid] = {
                "sample_id": sid,
                "contestant_model": model_name,
                "evaluation_status": "zero_filled",
                "zero_fill_reason": "no_prediction",
                **zero_scores(n_reference),
            }
        else:
            tasks.append((record, prediction))

    if tasks:
        client = make_judge_client(judge_model)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    judge_rewrite,
                    client,
                    judge_model,
                    record["rubric"],
                    source_html(record),
                    prediction,
                    sample_id(record),
                    model_name,
                ): sample_id(record)
                for record, prediction in tasks
            }
            for completed, future in enumerate(as_completed(futures), 1):
                row = future.result()
                rows[row["sample_id"]] = row
                if completed % 25 == 0:
                    partial = [rows[sample_id(record)] for record in benchmark
                               if sample_id(record) in rows]
                    result_path.write_text(
                        json.dumps(partial, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                print(f"[{completed}/{len(tasks)}] {row['sample_id']}: "
                      f"{row['evaluation_status']}", flush=True)

    ordered = [rows[sample_id(record)] for record in benchmark]
    return summarize(ordered, model_name, len(benchmark), output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score DeicticEdit-Bench Track B rewrite outputs."
    )
    parser.add_argument("--benchmark", "--jsonl", dest="benchmark", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument(
        "--webvr-root",
        help="Downloaded BroAlanTaps/WebVR directory; required when HTML is referenced by path",
    )
    parser.add_argument("--model-name", default="model")
    parser.add_argument("--output-dir", "--work-dir", dest="output_dir", default="track_b_results")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--workers", "--num-workers", dest="workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    summary = score_track_b(
        benchmark_path=args.benchmark,
        predictions_path=args.predictions,
        model_name=args.model_name,
        output_dir=args.output_dir,
        webvr_root=args.webvr_root,
        judge_model=args.judge_model,
        workers=args.workers,
        resume=args.resume,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
