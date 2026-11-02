#!/usr/bin/env python3
"""Run and score Track C with the fixed DeepSeek executor."""

from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from executor import DEEPSEEK_MODEL, DEEPSEEK_URL, DeepSeekExecutor
from run_eval import load_benchmark, safe_name, score_track_a
from scorer.e2e_report import build_report
from scorer.executor_prompt import (
    TEXT_REQUEST_BLOCK,
    UNIVERSAL_PROMPT,
    build_executor_prompt,
)
from scorer.io_utils import (
    load_predictions,
    sample_id,
    sha256_value,
    source_html,
    write_manifest,
)
from scorer.llm_client import DEFAULT_JUDGE_MODEL
from scorer.step_rewrite_judge import gt_steps_from_rubric, parse_rewrite, render_steps


def _load_completed(path: Path) -> dict[tuple[str, str], dict]:
    completed = {}
    if not path.exists():
        return completed
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("ok"):
            completed[(row["sample_id"], row["kind"])] = row
    return completed


def generate_html(*, benchmark: list[dict], rewrites: dict[str, str], output_dir: Path,
                  workers: int, rpm: int, timeout: int, retries: int,
                  max_tokens: int, stream: bool, resume: bool) -> tuple[Path, Path]:
    log_path = output_dir / "executor_results.jsonl"
    completed = _load_completed(log_path) if resume else {}
    jobs = []
    zero_recovered = set()
    prompt_hashes = {"recovered": {}, "oracle": {}}

    for record in benchmark:
        sid = sample_id(record)
        html = source_html(record)
        oracle_steps = gt_steps_from_rubric(record["rubric"], strict=False)
        oracle_prompt = build_executor_prompt(html, render_steps(oracle_steps, "en"))
        prompt_hashes["oracle"][sid] = sha256_value(oracle_prompt)
        if (sid, "oracle") not in completed:
            jobs.append((sid, "oracle", oracle_prompt))

        predicted_steps, error = parse_rewrite(rewrites.get(sid, ""))
        if error:
            zero_recovered.add(sid)
            continue
        recovered_prompt = build_executor_prompt(html, render_steps(predicted_steps, "en"))
        prompt_hashes["recovered"][sid] = sha256_value(recovered_prompt)
        if (sid, "recovered") not in completed:
            jobs.append((sid, "recovered", recovered_prompt))

    executor = DeepSeekExecutor(
        timeout=timeout, retries=retries, max_tokens=max_tokens, rpm=rpm, stream=stream
    ) if jobs else None
    lock = threading.Lock()
    mode = "a" if resume and log_path.exists() else "w"
    with log_path.open(mode, encoding="utf-8") as log:
        if jobs:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(executor.generate, prompt): (sid, kind, prompt)
                    for sid, kind, prompt in jobs
                }
                for index, future in enumerate(as_completed(futures), 1):
                    sid, kind, prompt = futures[future]
                    result = future.result()
                    row = {
                        "sample_id": sid,
                        "kind": kind,
                        "prompt_sha256": sha256_value(prompt),
                        **result,
                    }
                    with lock:
                        log.write(json.dumps(row, ensure_ascii=False) + "\n")
                        log.flush()
                    if row.get("ok"):
                        completed[(sid, kind)] = row
                    print(f"[{index}/{len(jobs)}] {sid} {kind}: "
                          f"{'ok' if row.get('ok') else 'failed'}", flush=True)

    recovered_path = output_dir / "recovered_predictions.jsonl"
    oracle_path = output_dir / "oracle_predictions.jsonl"
    with recovered_path.open("w", encoding="utf-8") as recovered_file, \
            oracle_path.open("w", encoding="utf-8") as oracle_file:
        executor_failed = {"recovered": [], "oracle": []}
        for record in benchmark:
            sid = sample_id(record)
            recovered = completed.get((sid, "recovered"), {})
            oracle = completed.get((sid, "oracle"), {})
            recovered_file.write(json.dumps({
                "sample_id": sid,
                "response": recovered.get("response", ""),
                "ok": bool(recovered.get("ok")),
                "error": ("rewrite unavailable" if sid in zero_recovered
                          else recovered.get("error")),
            }, ensure_ascii=False) + "\n")
            oracle_file.write(json.dumps({
                "sample_id": sid,
                "response": oracle.get("response", ""),
                "ok": bool(oracle.get("ok")),
                "error": oracle.get("error"),
            }, ensure_ascii=False) + "\n")
            if sid not in zero_recovered and not recovered.get("ok"):
                executor_failed["recovered"].append(sid)
            if not oracle.get("ok"):
                executor_failed["oracle"].append(sid)

    (output_dir / "execution_status.json").write_text(json.dumps({
        "rewrite_unavailable": sorted(zero_recovered),
        "executor_failed": executor_failed,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    (output_dir / "prompt_manifest.json").write_text(json.dumps({
        "executor_model": DEEPSEEK_MODEL,
        "executor_endpoint": DEEPSEEK_URL,
        "prompt_hashes": prompt_hashes,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return recovered_path, oracle_path


def run_track_c(*, benchmark_path: str | Path, rewrite_predictions_path: str | Path,
                model_name: str, output_dir: str | Path,
                webvr_root: str | Path | None = None,
                judge_model: str = DEFAULT_JUDGE_MODEL, judge_workers: int = 8,
                executor_workers: int = 4, executor_rpm: int = 60,
                executor_timeout: int = 600, executor_retries: int = 8,
                executor_max_tokens: int = 65536, stream: bool = False,
                resume: bool = False, skip_execution: bool = False,
                skip_scoring: bool = False) -> dict | None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = safe_name(model_name)
    benchmark = load_benchmark(benchmark_path, webvr_root)
    rewrites = load_predictions(rewrite_predictions_path)
    write_manifest(
        output_dir / "run_manifest.json", track="C", benchmark=benchmark_path,
        predictions=rewrite_predictions_path,
        config={
            "scoring_policy": "final_state",
            "metric": "NIU=mean(recovered_EFS)/mean(oracle_EFS)",
            "averaging": "paired_dataset_instance_macro",
            "executor_provider": "DeepSeek API",
            "executor_endpoint": DEEPSEEK_URL,
            "executor_model": DEEPSEEK_MODEL,
            "executor_max_tokens": executor_max_tokens,
            "executor_stream": stream,
            "executor_prompt_sha256": sha256_value({
                "universal_prompt": UNIVERSAL_PROMPT,
                "request_block": TEXT_REQUEST_BLOCK,
            }),
            "recovered_oracle_config_equal": True,
            "judge_provider": "Google Gemini API",
            "judge_model": judge_model,
            "model_name": model_name,
            "n_samples": len(benchmark),
            "sample_ids_sha256": sha256_value([sample_id(row) for row in benchmark]),
        },
    )
    if skip_execution:
        recovered_path = output_dir / "recovered_predictions.jsonl"
        oracle_path = output_dir / "oracle_predictions.jsonl"
        if not recovered_path.exists() or not oracle_path.exists():
            raise ValueError("--skip-execution requires existing recovered and oracle predictions")
    else:
        recovered_path, oracle_path = generate_html(
            benchmark=benchmark,
            rewrites=rewrites,
            output_dir=output_dir,
            workers=executor_workers,
            rpm=executor_rpm,
            timeout=executor_timeout,
            retries=executor_retries,
            max_tokens=executor_max_tokens,
            stream=stream,
            resume=resume,
        )
    if skip_scoring:
        return None

    recovered_dir = output_dir / "recovered_scoring"
    oracle_dir = output_dir / "oracle_scoring"
    score_track_a(
        benchmark_path=benchmark_path,
        predictions_path=recovered_path,
        model_name=f"{model_name}-recovered",
        output_dir=recovered_dir,
        webvr_root=webvr_root,
        judge_model=judge_model,
        workers=judge_workers,
        resume=resume,
    )
    score_track_a(
        benchmark_path=benchmark_path,
        predictions_path=oracle_path,
        model_name="deepseek-v4-pro-oracle",
        output_dir=oracle_dir,
        webvr_root=webvr_root,
        judge_model=judge_model,
        workers=judge_workers,
        resume=resume,
    )
    execution_status = json.loads(
        (output_dir / "execution_status.json").read_text(encoding="utf-8")
    )
    executor_failed = set(execution_status["executor_failed"]["recovered"])
    executor_failed.update(execution_status["executor_failed"]["oracle"])
    summary, _ = build_report(
        recovered_dir / "results.json", oracle_dir / "results.json", output_dir,
        executor_failed=executor_failed,
    )
    summary["contestant_model"] = model_name
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run and score DeicticEdit-Bench Track C with the fixed DeepSeek executor."
    )
    parser.add_argument("--benchmark", "--jsonl", dest="benchmark", required=True)
    parser.add_argument("--rewrite-predictions", "--predictions", dest="rewrites", required=True)
    parser.add_argument(
        "--webvr-root",
        help="Downloaded BroAlanTaps/WebVR directory; required when HTML is referenced by path",
    )
    parser.add_argument("--model-name", default="model")
    parser.add_argument("--output-dir", default="track_c_results")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--judge-workers", type=int, default=8)
    parser.add_argument("--executor-workers", type=int, default=4)
    parser.add_argument("--executor-rpm", type=int, default=60)
    parser.add_argument("--executor-timeout", type=int, default=600)
    parser.add_argument("--executor-retries", type=int, default=8)
    parser.add_argument("--executor-max-tokens", type=int, default=65536)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-execution", action="store_true")
    parser.add_argument("--skip-scoring", action="store_true")
    args = parser.parse_args()
    summary = run_track_c(
        benchmark_path=args.benchmark,
        rewrite_predictions_path=args.rewrites,
        model_name=args.model_name,
        output_dir=args.output_dir,
        webvr_root=args.webvr_root,
        judge_model=args.judge_model,
        judge_workers=args.judge_workers,
        executor_workers=args.executor_workers,
        executor_rpm=args.executor_rpm,
        executor_timeout=args.executor_timeout,
        executor_retries=args.executor_retries,
        executor_max_tokens=args.executor_max_tokens,
        stream=args.stream,
        resume=args.resume,
        skip_execution=args.skip_execution,
        skip_scoring=args.skip_scoring,
    )
    if summary is None:
        print("Executor outputs are ready; scoring was skipped.")
    else:
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
