#!/usr/bin/env python3
"""Combine rubric judgments, image checks, and No-damage into EFS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scorer import final_state


def load_optional(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate final-state EFS results.")
    parser.add_argument("--organized-root", required=True)
    parser.add_argument("--contestant-models", nargs="+", required=True)
    parser.add_argument("--judge-results", default="judge_results_rubric.json")
    parser.add_argument("--replace-image", default="replace_image_verification.json")
    parser.add_argument("--image-content", default="image_content_verification.json")
    parser.add_argument("--no-damage", default="no_damage_results_v3.json")
    parser.add_argument("--output", default="judge_results_v3.json")
    parser.add_argument("--scoring-policy", choices=["final_state"],
                        default="final_state")
    args = parser.parse_args()

    root = Path(args.organized_root)
    judgments = json.loads((root / args.judge_results).read_text(encoding="utf-8"))
    replace_image = load_optional(root / args.replace_image)
    image_content = load_optional(root / args.image_content)
    no_damage = load_optional(root / args.no_damage)
    corrections = final_state.correction_map(replace_image, image_content)
    damage_by_sample = {
        (row["sample_id"], row["contestant_model"]): row for row in no_damage
    }
    requested_models = set(args.contestant_models)
    output = []
    for judgment in judgments:
        model = judgment.get("contestant_model")
        if model not in requested_models:
            continue
        sid = judgment["sample_id"]
        rubric = json.loads((root / sid / "judge_rubric.json").read_text(encoding="utf-8"))
        final_state.validate_view(rubric)
        damage = damage_by_sample.get((sid, model))
        base = {
            "sample_id": sid,
            "contestant_model": model,
            "scoring_policy": "final_state",
        }
        if (judgment.get("evaluation_status") != "scored" or
                not damage or "overall_score" not in damage):
            output.append({**base, "evaluation_status": "judge_error",
                           "efs_weighted_v3": None})
            continue
        if judgment.get("input_signature") != damage.get("input_signature"):
            raise ValueError(f"{sid}: rubric and No-damage inputs do not match")
        result = final_state.reaggregate(
            judgment,
            rubric,
            corrections=corrections,
            no_damage=damage["overall_score"],
            no_damage_source="judge",
        )
        result["evaluation_status"] = "scored"
        result["input_signature"] = judgment["input_signature"]
        output.append(result)

    (root / args.output).write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
