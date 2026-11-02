#!/usr/bin/env python3
"""Track A rubric judge."""

from __future__ import annotations

JUDGE_PROMPT = 'You are an expert evaluator of web page editing.\n\nYour task is to judge, item by item, whether the contestant model\'s modification of the web page is correct, according to the **rubric** below.\nThe rubric already contains, for every editing step, the target element, the expected result of the edit and a detailed description of the changes that are forbidden.\n\nHow to judge:\n- Carefully compare the code diff between [ORIGINAL HTML] and [MODIFIED HTML]\n- Locate the target element via before_anchor in the rubric\n- Check whether the modification described by after_expected_change is present in the code\n- Check whether the problem described by forbidden_change occurred\n\nEach rubric item contains:\n- item_id: rubric item id\n- category: category (target_grounding = locating the target, edit_fulfillment = carrying out the edit, no_damage = no collateral damage, revision = undo / change of mind)\n- question: the question you must answer\n- before_anchor: features of the element before the edit (to help you locate it)\n- after_expected_change: the expected result of the edit\n- forbidden_change: changes that must not happen\n\nOutput in strict JSON format (do not include any other text):\n\n```json\n{\n  "items": [\n    {"item_id": "r1", "category": "target_grounding", "met": true, "reason": "brief justification"}\n  ]\n}\n```\n\nRequirements:\n- You must give a judgment for every item in the rubric; none may be omitted\n- met=true means the item passes, met=false means it does not pass\n- reason must cite the concrete HTML code change as evidence\n\n[RUBRIC]\n%RUBRIC_JSON%\n\n[ORIGINAL HTML]\n```html\n%BEFORE_HTML%\n```\n\n[MODIFIED HTML]\n```html\n%AFTER_HTML%\n```\n'

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from scorer import final_state
from scorer._types import GeminiTaskClient
from scorer.llm_client import DEFAULT_JUDGE_MODEL, make_judge_client

MAX_HTML_CHARS = 150_000


def _loads_lenient(blob: str) -> dict:
    try:
        value = json.loads(blob)
    except json.JSONDecodeError:
        from json_repair import repair_json
        value = json.loads(repair_json(blob))
    if not isinstance(value, dict) or not isinstance(value.get("items"), list):
        raise TypeError("Judge response must contain an items array")
    return value


def judge_with_rubric(client: GeminiTaskClient, model: str,
                      before_html_path: Path, after_html_path: Path,
                      rubric: dict, sample_id: str, contestant_model: str) -> dict:
    active_items = final_state.active_items(rubric, "final_state")
    compact_rubric = []
    for item in active_items:
        evidence = item.get("expected_visible_evidence") or {}
        compact_rubric.append({
            "item_id": item["item_id"],
            "category": item["category"],
            "question": item["question"],
            "before_anchor": evidence.get("before_anchor", ""),
            "after_expected_change": evidence.get("after_expected_change", ""),
            "forbidden_change": evidence.get("forbidden_change", ""),
        })
    before_html = before_html_path.read_text(encoding="utf-8", errors="ignore")[:MAX_HTML_CHARS]
    after_html = after_html_path.read_text(encoding="utf-8", errors="ignore")[:MAX_HTML_CHARS]
    prompt = (final_state.final_context(rubric) + "\n\n" + JUDGE_PROMPT
              .replace("%RUBRIC_JSON%", json.dumps(compact_rubric, indent=2,
                                                    ensure_ascii=False))
              .replace("%BEFORE_HTML%", before_html)
              .replace("%AFTER_HTML%", after_html))
    base = {"sample_id": sample_id, "contestant_model": contestant_model,
            "scoring_policy": "final_state"}
    try:
        response = client.client.call_text(
            [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            model=model, max_tokens=None, include_thoughts=False,
            retries=3, backoff=2.0,
        )
        if response.get("error"):
            raise ValueError(response["error"])
        text = response.get("content") or response.get("text") or ""
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        value = _loads_lenient(match.group(1) if match else text)
        audit_items, scored_items = final_state.normalize_results(
            value.get("items") or [], rubric, "final_state"
        )
        category = final_state.aggregate_items(scored_items, no_damage=None)
        result = {
            **base,
            "evaluation_status": "scored",
            "items": audit_items,
            "efs_sample": category["efs_weighted_v3"],
            "category_scores": category["category_scores_v3"],
            "items_met": category["items_met"],
            "items_total": category["items_total"],
            "input_signature": final_state.input_signature(
                rubric, before_html_path, after_html_path
            ),
        }
        return result
    except Exception as exc:  # noqa: BLE001
        return {**base, "evaluation_status": "judge_error", "error": str(exc),
                "efs_sample": -1}


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge Track A rubric items.")
    parser.add_argument("--organized-root", required=True)
    parser.add_argument("--contestant-models", nargs="+", required=True)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--output", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prompt-lang", choices=["en"], default="en")
    parser.add_argument("--scoring-policy", choices=["final_state"],
                        default="final_state")
    args = parser.parse_args()

    root = Path(args.organized_root)
    output_path = Path(args.output) if args.output else root / "judge_results_rubric.json"
    previous = {}
    if args.resume and output_path.exists():
        for row in json.loads(output_path.read_text(encoding="utf-8")):
            if row.get("evaluation_status") == "scored":
                previous[(row["sample_id"], row["contestant_model"])] = row
    client = make_judge_client(args.judge_model)
    results = []

    for contestant_model in args.contestant_models:
        model_dir = root / "predictions" / contestant_model
        if not model_dir.exists():
            continue
        tasks = []
        for prediction_dir in sorted(item for item in model_dir.iterdir() if item.is_dir()):
            sid = prediction_dir.name
            rubric_path = root / sid / "judge_rubric.json"
            before_path = root / sid / "source_page" / "index.html"
            after_path = prediction_dir / "modified_page" / "index.html"
            if not rubric_path.exists() or not before_path.exists() or not after_path.exists():
                continue
            rubric = json.loads(rubric_path.read_text(encoding="utf-8"))
            final_state.validate_view(rubric)
            signature = final_state.input_signature(rubric, before_path, after_path)
            prior = previous.get((sid, contestant_model))
            if prior and prior.get("input_signature") == signature:
                results.append(prior)
            else:
                tasks.append((sid, before_path, after_path, rubric))

        def run_one(task, current_model=contestant_model):
            sid, before_path, after_path, rubric = task
            return judge_with_rubric(
                client, args.judge_model, before_path, after_path,
                rubric, sid, current_model,
            )

        with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
            futures = [pool.submit(run_one, task) for task in tasks]
            for index, future in enumerate(as_completed(futures), 1):
                row = future.result()
                results.append(row)
                print(f"[{index}/{len(tasks)}] {row['sample_id']}: "
                      f"{row['evaluation_status']}", flush=True)
                if args.resume and index % 25 == 0:
                    output_path.write_text(
                        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
    results.sort(key=lambda row: (row.get("contestant_model", ""), row.get("sample_id", "")))
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")


if __name__ == "__main__":
    main()
