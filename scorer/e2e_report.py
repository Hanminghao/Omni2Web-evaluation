"""Build a paired Track C report from recovered and oracle Track A results."""

from __future__ import annotations

import json
from pathlib import Path

CATEGORIES = ("target_grounding", "edit_fulfillment", "no_damage", "revision")


def _load_results(path: str | Path) -> dict[str, dict]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return {row["sample_id"]: row for row in rows}


def build_report(recovered_path: str | Path, oracle_path: str | Path,
                 output_dir: str | Path | None = None,
                 executor_failed: set[str] | None = None) -> tuple[dict, list[dict]]:
    recovered = _load_results(recovered_path)
    oracle = _load_results(oracle_path)
    if set(recovered) != set(oracle):
        raise ValueError("Recovered and oracle result files have different sample-ID sets")

    executor_failed = executor_failed or set()
    rows = []
    recovered_values = []
    oracle_values = []
    judge_failed = 0
    for sid in sorted(recovered):
        left, right = recovered[sid], oracle[sid]
        if sid in executor_failed:
            status = "executor_error"
            recovered_efs = oracle_efs = None
        elif (left.get("evaluation_status") == "judge_error" or
                right.get("evaluation_status") == "judge_error"):
            status = "judge_error"
            recovered_efs = oracle_efs = None
            judge_failed += 1
        else:
            status = "scored"
            recovered_efs = left.get("efs_weighted_v3")
            oracle_efs = right.get("efs_weighted_v3")
            if recovered_efs is None or oracle_efs is None:
                raise ValueError(f"{sid}: missing EFS without a judge_error status")
            recovered_values.append(float(recovered_efs))
            oracle_values.append(float(oracle_efs))
        rows.append({
            "sample_id": sid,
            "evaluation_status": status,
            "recovered_efs": recovered_efs,
            "oracle_efs": oracle_efs,
        })

    mean_recovered = sum(recovered_values) / len(recovered_values) if recovered_values else None
    mean_oracle = sum(oracle_values) / len(oracle_values) if oracle_values else None
    summary = {
        "track": "C",
        "metric": "NIU",
        "n_samples_in_benchmark": len(rows),
        "n_paired_scored": len(recovered_values),
        "n_judge_failed": judge_failed,
        "n_executor_failed": len(executor_failed),
        "recovered_efs": mean_recovered,
        "oracle_efs": mean_oracle,
        "niu": (mean_recovered / mean_oracle
                if mean_recovered is not None and mean_oracle else None),
    }
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "results.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return summary, rows
