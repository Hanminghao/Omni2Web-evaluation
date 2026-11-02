"""Validate final-state rubric metadata and select scoreable items."""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

POLICIES = ("final_state",)
WEIGHTS = {"target_grounding": .1, "edit_fulfillment": .5, "no_damage": .2, "revision": .2}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def steps_from_rubric(rubric):
    """Original question step numbers, including steps with empty anchors."""
    pairs = {}
    ids = set()
    for it in rubric["items"]:
        iid, cat = it["item_id"], it["category"]
        if iid in ids:
            raise ValueError(f"Duplicate rubric item: {iid}")
        ids.add(iid)
        if cat == "no_damage":
            continue
        if cat not in WEIGHTS:
            raise ValueError(f"Unknown category: {cat}")
        m = re.search(r"步骤\s*(\d+)|\bstep\s*(\d+)", it["question"], re.IGNORECASE)
        if not m:
            raise ValueError(f"No original step number: {iid}")
        n = int(next(g for g in m.groups() if g))
        role = "grounding" if cat == "target_grounding" else "edit"
        pair = pairs.setdefault(n, {})
        if role in pair:
            raise ValueError(f"Duplicate {role} at step {n}")
        pair[role] = it
    if sorted(pairs) != list(range(1, len(pairs) + 1)):
        raise ValueError("Non-contiguous original steps")
    result = []
    for n, pair in sorted(pairs.items()):
        if set(pair) != {"grounding", "edit"}:
            raise ValueError(f"Missing rubric pair at step {n}")
        g, e = pair["grounding"], pair["edit"]
        ev = e.get("expected_visible_evidence") or {}
        result.append({"step": n, "grounding_item_id": g["item_id"],
                       "edit_item_id": e["item_id"], "category": e["category"],
                       "operation": e.get("step_type", ""),
                       "target": ev.get("before_anchor", ""),
                       "instruction": ev.get("after_expected_change", "")})
    return result


def validate_view(rubric):
    """Check that the compact revision links and exclusion list agree."""
    steps = steps_from_rubric(rubric)
    by_step = {entry["step"]: entry for entry in steps}
    links = rubric.get("revision_links")
    excluded = rubric.get("excluded_steps")
    if not isinstance(links, list) or not isinstance(excluded, list):
        raise TypeError("Rubric must contain revision_links and excluded_steps")

    expected: dict[int, set[int]] = defaultdict(set)
    seen_revision_steps = set()
    for link in links:
        revision_step = link.get("revision_step")
        reverted_steps = link.get("reverted_steps")
        if (type(revision_step) is not int or revision_step not in by_step
                or revision_step in seen_revision_steps or not isinstance(reverted_steps, list)
                or not reverted_steps or len(reverted_steps) != len(set(reverted_steps))):
            raise ValueError("Invalid or duplicate revision link")
        seen_revision_steps.add(revision_step)
        if by_step[revision_step]["category"] != "revision":
            raise ValueError("Revision link does not point to a revision step")
        for old in reverted_steps:
            if type(old) is not int or old not in by_step or old >= revision_step:
                raise ValueError("Invalid reverted step")
            expected[old].add(revision_step)

    actual: dict[int, set[int]] = {}
    all_item_ids = {item["item_id"] for item in rubric["items"]}
    excluded_item_ids = set()
    for entry in excluded:
        step = entry.get("step")
        item_ids = entry.get("item_ids")
        revision_steps = entry.get("reverted_by_steps")
        if type(step) is not int or step not in by_step or step in actual:
            raise ValueError("Invalid or duplicate excluded step")
        expected_items = {by_step[step]["grounding_item_id"], by_step[step]["edit_item_id"]}
        if not isinstance(item_ids, list) or set(item_ids) != expected_items:
            raise ValueError("Excluded item IDs do not match the rubric step")
        if not isinstance(revision_steps, list) or not revision_steps:
            raise ValueError("Excluded step has no revision step")
        if any(type(n) is not int or n <= step or n not in by_step for n in revision_steps):
            raise ValueError("Invalid reverted_by_steps")
        if excluded_item_ids.intersection(item_ids):
            raise ValueError("Rubric item is excluded more than once")
        excluded_item_ids.update(item_ids)
        actual[step] = set(revision_steps)
    if excluded_item_ids - all_item_ids:
        raise ValueError("Unknown rubric item in excluded_steps")
    if dict(expected) != actual:
        raise ValueError("revision_links and excluded_steps disagree")
    return {"revision_links": links, "excluded_steps": excluded}


def active_items(rubric, policy="final_state"):
    if policy not in POLICIES:
        raise ValueError(f"Unknown policy: {policy}")
    view = validate_view(rubric)
    skip = {iid for step in view["excluded_steps"] for iid in step["item_ids"]}
    return [it for it in rubric["items"] if it["item_id"] not in skip]


def normalize_results(results, rubric, policy="final_state"):
    """Preserve raw evidence; only canonical active IDs count, once each."""
    active = {x["item_id"]: x for x in active_items(rubric, policy)}
    all_ids = {x["item_id"] for x in rubric["items"]}
    seen = set()
    audit, scored = [], []
    for raw in results:
        it = deepcopy(raw)
        iid = it["item_id"]
        if iid in seen:
            raise ValueError(f"Duplicate returned item: {iid}")
        seen.add(iid)
        it["raw_met"], it["raw_reason"] = raw.get("met"), raw.get("reason", "")
        it["evaluation_status"] = ("scored" if iid in active else
                                    "skipped_reverted" if iid in all_ids else "ignored_unknown_item")
        if iid in active:


            value = it.get("met")
            if type(value) is not bool and not (type(value) is int and value in (0, 1)):
                raise ValueError(f"Non-binary met: {iid}")
            it["met"] = bool(value)
            it["category"] = active[iid]["category"]
            scored.append(it)
        audit.append(it)
    for iid, rb in active.items():
        if iid not in seen:
            it = {"item_id": iid, "category": rb["category"], "met": False,
                  "reason": "Judge omitted this active rubric item.",
                  "evaluation_status": "missing_active_item", "raw_met": None, "raw_reason": ""}
            audit.append(it)
            scored.append(it)
    return audit, scored


def aggregate_items(items, no_damage):
    if no_damage is not None and (not isinstance(no_damage, (int, float)) or
                                 not math.isfinite(no_damage) or not 0 <= no_damage <= 1):
        raise ValueError("Invalid No-damage score")
    cat_met = defaultdict(list)
    for it in items:
        if it.get("category") in WEIGHTS:
            cat_met[it["category"]].append(bool(it.get("met", False)))
    scores = {c: sum(ms) / len(ms) for c, ms in cat_met.items()}
    if no_damage is not None:
        scores["no_damage"] = no_damage
    total = sum(WEIGHTS[c] for c in scores)
    weighted = sum(WEIGHTS[c] * s for c, s in scores.items()) / total if total else None
    return {"efs_weighted_v3": round(weighted, 4) if weighted is not None else None,
            "category_scores_v3": {c: round(scores[c], 4) if c in scores else None for c in WEIGHTS},
            "category_counts": {c: {"met": sum(cat_met[c]), "total": len(cat_met[c])} for c in WEIGHTS},
            "items_total": len(items), "items_met": sum(bool(i.get("met")) for i in items)}


def correction_map(replace_image, image_content):
    """Collect failed image-replacement checks by sample, model, and item."""
    result = {}
    for c in replace_image:
        if c["original_met"] and not c["corrected_met"]:
            result[(c["sample_id"], c["contestant_model"], c["item_id"])] = c["correction_reason"]
    for c in image_content:
        key = (c["sample_id"], c["contestant_model"], c["item_id"])
        if key in result or not c.get("pred_downloaded", False):
            continue
        diff, match = c.get("is_different", False), c.get("matches_description", False)
        if (diff and not match) or (not diff and c.get("source_downloaded", False)):
            result[key] = f"[IMG_CONTENT] diff={diff} match={match}: {c.get('image_content', '')}. {c.get('reason', '')}"
    return result


def reaggregate(record, rubric, corrections=None, no_damage=None,
                no_damage_source="judge"):
    view = validate_view(rubric)
    if no_damage is None:
        raise ValueError("A valid No-damage result is required; never silently substitute E1")
    audit, scored = normalize_results(record.get("items", []), rubric)
    flips = 0
    for it in scored:
        key = (record["sample_id"], record["contestant_model"], it["item_id"])
        if key in (corrections or {}):
            it["met"] = False
            it["correction_reason"] = corrections[key]
            flips += 1
    return {"sample_id": record["sample_id"], "contestant_model": record["contestant_model"],
            "scoring_policy": "final_state", "evaluation_status": "scored",
            "judgment_source": "rubric_judge",
            "no_damage_source": no_damage_source, "items": audit, **aggregate_items(scored, no_damage),
            "ri_flips": flips, "excluded_steps": view["excluded_steps"],
            "source_rubric_sha256": digest(rubric["items"]),
            "evaluation_sha256": digest(view),
            "source_judgment_sha256": digest(record)}


def final_context(rubric):
    """Evaluator-only reference context. Never use in contestant prompts."""
    view = validate_view(rubric)
    cancelled = {step["step"] for step in view["excluded_steps"]}
    lines = ["FINAL-STATE EVALUATION: Only the final HTML is observable. Score only the supplied active rubric items.",
             "No intermediate execution trace is required. A correct restored state may have no diff from the original.",
             "Cancelled requests below are historical context, NOT requirements or permissions for the final page.",
             "Restoration references apply only to the affected target/properties, never to the whole page.",
             "HISTORY (reference only):"]
    for s in steps_from_rubric(rubric):
        state = "CANCELLED" if s["step"] in cancelled else "ACTIVE"
        lines.append(f"[step{s['step']}] [{state}] {json.dumps(s['instruction'], ensure_ascii=False)}")
    lines.append("REVISION LINKS: " + json.dumps(view["revision_links"], ensure_ascii=False))
    return "\n".join(lines)


def input_signature(rubric, before, after):
    return {"evaluation_sha256": digest(validate_view(rubric)),
            "source_html_sha256": file_digest(before), "prediction_sha256": file_digest(after)}
