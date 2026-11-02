#!/usr/bin/env python3
"""Track B semantic matching and Component-F1 scoring."""

from __future__ import annotations

JUDGE_PROMPT = 'You are an expert evaluator of *instruction rewriting* for web page editing.\n\nA user recorded a video in which they point at parts of a web page and speak editing requests\nusing deictic references ("this", "here", "these two"). A contestant model watched that video\nand rewrote the request into explicit, self-contained editing steps. Your job is to score that\nrewrite against the reference answer.\n\nYou are given:\n- [REFERENCE STEPS]: the ground-truth steps, written by a human annotator from the same video.\n  Each has a `target` (which part of the page) and an `instruction` (what to change).\n- [CONTESTANT STEPS]: what the contestant model produced. The step numbering may differ, steps\n  may be missing, merged, split, extra, or out of order.\n- [SOURCE HTML]: the page before any edit, so you can check whether a target description really\n  points at the element the reference means.\n\nDo this, in order:\n\n1. **Match** each reference step to at most one contestant step. Match by *what is being done to\n   what*, not by step number and not by wording. A contestant step may be used for at most one\n   reference step. If no contestant step plausibly corresponds to a reference step, leave it\n   unmatched (`matched_contestant_step: null`) and score both criteria 0.\n\n2. For each matched pair, judge two criteria independently, each 0 or 1:\n\n   - `grounding_met`: does the contestant\'s `target` identify the **same element or region** of\n     the page as the reference\'s `target`? Use [SOURCE HTML] to resolve both descriptions to\n     actual elements. Score 1 when they resolve to the same element/region even if worded\n     completely differently (a quoted visible string, a heading name, a position, a CSS-ish\n     description are all acceptable ways to say it). Score 0 when it resolves to a different\n     element, when it is so vague that it could be many elements, or when it still contains a\n     deictic reference ("this one", "here") that carries the location.\n   - `instruction_met`: if a competent web developer followed the contestant\'s `instruction`\n     on that element, would the result be the change the reference\'s `instruction` describes?\n     Score 1 when the *outcome* matches. Score 0 when the outcome differs, when the instruction\n     is too vague to act on, or when it still relies on the video to be understood.\n\n**Do not deduct for wording, style, verbosity, language, or ordering.** Only the identified\nelement and the resulting change matter. Being more specific than the reference is fine.\nWhen the reference `instruction` names a value (a color, a count, a position) the contestant\nmust reach the same value; a different value is 0.\n\nOutput ONLY this JSON, no other text:\n\n```json\n{\n  "steps": [\n    {"gt_step": 1, "matched_contestant_step": 2, "grounding_met": 1, "instruction_met": 0,\n     "reason": "brief justification, quote the deciding words from both sides"}\n  ],\n  "extra_contestant_steps": [4],\n  "notes": "optional, one short line"\n}\n```\n\nRequirements:\n- Emit exactly one entry per reference step, in reference order. Never omit one.\n- `extra_contestant_steps`: step numbers of contestant steps you matched to nothing.\n- `reason` must cite the concrete wording that decided the call.\n\n[REFERENCE STEPS]\n%GT_STEPS%\n\n[CONTESTANT STEPS]\n%PRED_STEPS%\n\n[SOURCE HTML]\n```html\n%SOURCE_HTML%\n```\n'

import json
import re

from scorer._types import GeminiTaskClient

MAX_HTML_CHARS = 150_000
EDIT_CATEGORIES = {"edit_fulfillment", "revision"}
_TARGET_ALIASES = ("target", "region", "location", "element", "area", "where", "目标", "区域")
_INSTRUCTION_ALIASES = ("instruction", "edit", "change", "action", "request", "what", "指令", "修改")


def _first_string(value: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        item = value.get(key)
        if isinstance(item, str) and item.strip():
            return item.strip()
    return ""


def extract_json(text: str):
    for match in re.finditer(r"```(?:json|JSON)?\s*(.*?)```", text or "", re.DOTALL):
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    try:
        return json.loads((text or "").strip())
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for index, char in enumerate(text or ""):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
            return value
        except json.JSONDecodeError:
            continue
    return None


def parse_rewrite(text: str) -> tuple[list[dict], str]:
    if not (text or "").strip():
        return [], "empty response"
    value = extract_json(text)
    if isinstance(value, dict):
        for key in ("steps", "rewrite", "instructions", "result", "output", "步骤"):
            if isinstance(value.get(key), list):
                value = value[key]
                break
        else:
            value = [value]
    if not isinstance(value, list):
        return [], "response does not contain a JSON step array"
    steps = []
    for position, item in enumerate(value, 1):
        if not isinstance(item, dict):
            continue
        target = _first_string(item, _TARGET_ALIASES)
        instruction = _first_string(item, _INSTRUCTION_ALIASES)
        if not target and not instruction:
            continue
        steps.append({
            "step": len(steps) + 1,
            "raw_step": item.get("step", item.get("step_num", position)),
            "target": target,
            "instruction": instruction,
        })
    return (steps, "") if steps else ([], "JSON contains no usable steps")


def reference_steps_from_rubric(rubric: dict) -> list[dict]:
    """Return every canonical step, including steps with incomplete anchors."""
    items = rubric.get("items") or []
    steps = []
    index = 0
    while index < len(items):
        grounding = items[index]
        category = grounding.get("category")
        if category == "no_damage":
            index += 1
            continue
        if category != "target_grounding" or index + 1 >= len(items):
            raise ValueError(f"Invalid rubric pair at item index {index}")
        edit = items[index + 1]
        if edit.get("category") not in EDIT_CATEGORIES:
            raise ValueError(f"Invalid edit category at item index {index + 1}")
        target = ((grounding.get("expected_visible_evidence") or {})
                  .get("before_anchor") or "").strip()
        instruction = ((edit.get("expected_visible_evidence") or {})
                       .get("after_expected_change") or "").strip()
        steps.append({
            "canonical_step": len(steps) + 1,
            "target": target,
            "instruction": instruction,
            "step_type": edit.get("step_type", ""),
            "category": edit.get("category"),
            "gt_item_ids": [grounding.get("item_id"), edit.get("item_id")],
            "judgeable": bool(target and instruction),
        })
        index += 2
    if not steps:
        raise ValueError("Rubric has no edit steps")
    return steps


def gt_steps_from_rubric(rubric: dict, strict: bool = True) -> list[dict]:
    references = reference_steps_from_rubric(rubric)
    if strict and any(not step["judgeable"] for step in references):
        missing = [step["canonical_step"] for step in references if not step["judgeable"]]
        raise ValueError(f"Missing reference anchors at steps {missing}")
    result = []
    for reference in references:
        if not reference["judgeable"]:
            continue
        result.append({
            "step": len(result) + 1,
            "canonical_step": reference["canonical_step"],
            **{key: reference[key] for key in (
                "target", "instruction", "step_type", "category", "gt_item_ids"
            )},
        })
    return result


def render_steps(steps: list[dict], lang: str = "en") -> str:
    labels = ("Step", "Target", "Edit") if lang == "en" else ("步骤", "目标", "修改")
    lines = []
    for index, step in enumerate(steps, 1):
        lines.extend([
            f"{labels[0]} {index}",
            f"  {labels[1]}: {step.get('target', '')}",
            f"  {labels[2]}: {step.get('instruction', '')}",
        ])
    return "\n".join(lines)


def component_f1(true_positives: int, n_reference: int, n_predicted: int) -> float:
    denominator = n_reference + n_predicted
    return 2.0 * true_positives / denominator if denominator else 0.0


def derive_scores(target_tp: int, action_tp: int, joint_tp: int,
                  n_reference: int, n_predicted: int) -> dict:
    target_f1 = component_f1(target_tp, n_reference, n_predicted)
    action_f1 = component_f1(action_tp, n_reference, n_predicted)
    joint_f1 = component_f1(joint_tp, n_reference, n_predicted)
    return {
        "target_tp": target_tp,
        "action_tp": action_tp,
        "joint_tp": joint_tp,
        "n_reference_steps": n_reference,
        "n_predicted_steps": n_predicted,
        "target_f1": target_f1,
        "action_f1": action_f1,
        "joint_f1": joint_f1,
        "irs": 0.5 * target_f1 + 0.5 * action_f1,
    }


def zero_scores(n_reference: int, n_predicted: int = 0) -> dict:
    return derive_scores(0, 0, 0, n_reference, n_predicted)


def _integer_index(value):
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("boolean step index")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    raise ValueError(f"invalid step index: {value!r}")


def _binary(value, field: str) -> int:
    if value in (0, False, "0", "false"):
        return 0
    if value in (1, True, "1", "true"):
        return 1
    raise ValueError(f"{field} must be binary, got {value!r}")


def validate_judgment(value: dict, n_judged_reference: int,
                      n_predicted: int) -> tuple[list[dict], list[int]]:
    if not isinstance(value, dict) or not isinstance(value.get("steps"), list):
        raise TypeError("judge output must contain a steps array")
    by_reference = {}
    used_predictions = set()
    cleaned = []
    for item in value["steps"]:
        if not isinstance(item, dict):
            raise TypeError("judge steps must be objects")
        reference = _integer_index(item.get("gt_step"))
        if reference is None or not 1 <= reference <= n_judged_reference:
            raise ValueError(f"reference step out of range: {reference}")
        if reference in by_reference:
            raise ValueError(f"duplicate reference step: {reference}")
        predicted = _integer_index(item.get("matched_contestant_step"))
        if predicted is not None:
            if not 1 <= predicted <= n_predicted:
                raise ValueError(f"predicted step out of range: {predicted}")
            if predicted in used_predictions:
                raise ValueError(f"predicted step matched more than once: {predicted}")
            used_predictions.add(predicted)
        grounding = _binary(item.get("grounding_met"), "grounding_met")
        action = _binary(item.get("instruction_met"), "instruction_met")
        if predicted is None and (grounding or action):
            raise ValueError("unmatched step received a positive judgment")
        normalized = dict(item)
        normalized.update(
            gt_step=reference,
            matched_contestant_step=predicted,
            grounding_met=grounding,
            instruction_met=action,
        )
        by_reference[reference] = normalized
    missing = set(range(1, n_judged_reference + 1)) - set(by_reference)
    if missing:
        raise ValueError(f"judge omitted reference steps: {sorted(missing)}")
    for reference in sorted(by_reference):
        cleaned.append(by_reference[reference])
    extras = sorted(set(range(1, n_predicted + 1)) - used_predictions)
    return cleaned, extras


def _judge_prompt(gt_steps: list[dict], pred_steps: list[dict], source_html: str) -> str:
    keep = lambda steps: [
        {key: step.get(key) for key in ("step", "target", "instruction")}
        for step in steps
    ]
    return (JUDGE_PROMPT
            .replace("%GT_STEPS%", json.dumps(keep(gt_steps), indent=2, ensure_ascii=False))
            .replace("%PRED_STEPS%", json.dumps(keep(pred_steps), indent=2, ensure_ascii=False))
            .replace("%SOURCE_HTML%", (source_html or "")[:MAX_HTML_CHARS]))


def judge_rewrite(judge_client: GeminiTaskClient, model: str, rubric: dict,
                  source_html: str, prediction_text: str, sample_id: str,
                  contestant_model: str, semantic_retries: int = 3) -> dict:
    base = {"sample_id": sample_id, "contestant_model": contestant_model}
    try:
        references = reference_steps_from_rubric(rubric)
        gt_steps = gt_steps_from_rubric(rubric, strict=False)
    except ValueError as exc:
        return {**base, "evaluation_status": "judge_error", "error": str(exc)}
    n_reference = len(references)
    predicted_steps, parse_error = parse_rewrite(prediction_text)
    if parse_error:
        return {
            **base,
            "evaluation_status": "zero_filled",
            "zero_fill_reason": "rewrite_parse_failed",
            "parse_error": parse_error,
            "reference_steps": references,
            "predicted_steps": [],
            "steps": [],
            **zero_scores(n_reference),
        }

    prompt = _judge_prompt(gt_steps, predicted_steps, source_html)
    errors = []
    for _ in range(max(1, semantic_retries)):
        try:
            response = judge_client.client.call_text(
                [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                model=model, max_tokens=None, include_thoughts=False,
                retries=3, backoff=2.0,
            )
            if response.get("error"):
                raise ValueError(response["error"])
            value = extract_json(response.get("content") or response.get("text") or "")
            steps, extras = validate_judgment(value, len(gt_steps), len(predicted_steps))
            by_judge_step = {step["step"]: step for step in gt_steps}
            target_tp = action_tp = joint_tp = 0
            for item in steps:
                item["canonical_gt_step"] = by_judge_step[item["gt_step"]]["canonical_step"]
                target_tp += item["grounding_met"]
                action_tp += item["instruction_met"]
                joint_tp += int(item["grounding_met"] and item["instruction_met"])
            return {
                **base,
                "evaluation_status": "scored",
                "reference_steps": references,
                "predicted_steps": predicted_steps,
                "steps": steps,
                "extra_predicted_steps": extras,
                "judge_notes": value.get("notes", ""),
                **derive_scores(target_tp, action_tp, joint_tp,
                                n_reference, len(predicted_steps)),
            }
        except (ValueError, TypeError, KeyError) as exc:
            errors.append(str(exc))
    return {
        **base,
        "evaluation_status": "judge_error",
        "error": errors[-1] if errors else "judge failed",
        "semantic_attempt_errors": errors,
    }
