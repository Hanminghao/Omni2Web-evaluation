#!/usr/bin/env python3
"""No-damage detection v3: code-level structured diff + LLM semantic judgment.

Instead of separate code detection and LLM judge:
1. Code layer: parse both HTMLs, generate a structured change list
2. LLM layer: given the change list + edit task descriptions, judge each change
   as "expected edit result" or "collateral damage"

This replaces both check_no_damage_v2.py and the single-item no_damage rubric judge.
"""

import json
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString

from scorer import final_state as fs
from scorer._types import GeminiTaskClient
from scorer.llm_client import DEFAULT_JUDGE_MODEL, make_judge_client

SKIP_TAGS = {"script", "style", "noscript", "svg", "math"}

TRACKED_CSS_PROPS = {
    "display", "visibility", "opacity", "color", "background", "background-color",
    "background-image", "font-size", "font-family", "flex-direction", "grid-template-columns",
    "position", "width", "height", "margin", "padding", "border", "text-decoration",
    "filter", "transform", "order", "flex", "gap",
}


def _el_path(el, max_depth=4) -> str:
    parts = []
    cur = el
    for _ in range(max_depth):
        if not cur or not getattr(cur, "name", None):
            break
        tag = cur.name
        cls = ".".join(cur.get("class", [])[:2])
        eid = cur.get("id", "")
        label = tag
        if eid:
            label += f"#{eid}"
        elif cls:
            label += f".{cls}"
        parts.append(label)
        cur = cur.parent
    return " > ".join(reversed(parts))


def _parse_inline_style(style_str: str) -> dict:
    props = {}
    if not style_str:
        return props
    for decl in style_str.split(";"):
        decl = decl.strip()
        if ":" not in decl:
            continue
        prop, val = decl.split(":", 1)
        prop = prop.strip().lower()
        val = val.strip()
        if prop in TRACKED_CSS_PROPS:
            props[prop] = val
    return props


def _text_fingerprint(el) -> str:
    tag = el.parent.name if el.parent else "?"
    cls = ".".join(el.parent.get("class", [])[:2]) if el.parent else ""
    return f"{tag}.{cls}" if cls else tag


def extract_text_with_path(html_content: str) -> list[dict]:
    soup = BeautifulSoup(html_content, "html.parser")
    nodes = []
    for el in soup.descendants:
        if not isinstance(el, NavigableString):
            continue
        if el.parent and el.parent.name in SKIP_TAGS:
            continue
        text = el.strip()
        if not text:
            continue
        path = _el_path(el.parent) if el.parent else "root"
        nodes.append({"text": text, "path": path, "fp": _text_fingerprint(el)})
    return nodes


def extract_elements(html_content: str) -> dict:
    soup = BeautifulSoup(html_content, "html.parser")
    result = {"imgs": [], "links": [], "styled": [], "identifiable": [], "css_rules": []}

    for i, img in enumerate(soup.find_all("img")):
        result["imgs"].append({
            "index": i, "src": img.get("src", ""), "alt": img.get("alt", ""),
            "path": _el_path(img), "style": _parse_inline_style(img.get("style", "")),
        })

    for i, a in enumerate(soup.find_all("a")):
        result["links"].append({
            "index": i, "href": a.get("href", ""), "text": a.get_text(strip=True)[:80],
            "path": _el_path(a),
        })

    for el in soup.find_all(True):
        style_str = el.get("style", "")
        if not style_str:
            continue
        props = _parse_inline_style(style_str)
        if props:
            result["styled"].append({
                "path": _el_path(el), "tag": el.name,
                "classes": " ".join(el.get("class", [])[:3]),
                "props": props, "text_hint": el.get_text(strip=True)[:40],
            })

    for el in soup.find_all(True):
        eid = el.get("id", "")
        cls = " ".join(el.get("class", []))
        if eid or cls:
            result["identifiable"].append({
                "tag": el.name, "id": eid, "class": cls,
                "path": _el_path(el), "text_hint": el.get_text(strip=True)[:30],
            })

    for style_tag in soup.find_all("style"):
        css_text = style_tag.string or ""
        for m in re.finditer(r'([^{}@]+?)\{([^{}]*)\}', css_text):
            selector = m.group(1).strip()
            body = m.group(2).strip()
            if not selector or selector.startswith("@"):
                continue
            props = {}
            for decl in body.split(";"):
                decl = decl.strip()
                if ":" not in decl:
                    continue
                prop, val = decl.split(":", 1)
                props[prop.strip().lower()] = val.strip()
            if props:
                result["css_rules"].append({"selector": selector, "props": props})

    return result


def html_structured_diff(source_html: str, pred_html: str, max_changes: int = 60) -> list[dict]:
    source_texts = extract_text_with_path(source_html)
    pred_texts = extract_text_with_path(pred_html)
    source_els = extract_elements(source_html)
    pred_els = extract_elements(pred_html)

    changes = []

    source_text_counter = Counter(t["text"] for t in source_texts)
    pred_text_counter = Counter(t["text"] for t in pred_texts)
    missing_texts = source_text_counter - pred_text_counter

    source_text_path = {}
    for t in source_texts:
        if t["text"] not in source_text_path:
            source_text_path[t["text"]] = t["path"]

    for text, count in missing_texts.items():
        changes.append({
            "type": "text_removed",
            "location": source_text_path.get(text, "unknown"),
            "original": text[:100],
            "modified": None,
            "count": count,
        })

    added_texts = pred_text_counter - source_text_counter
    pred_text_path = {}
    for t in pred_texts:
        if t["text"] not in pred_text_path:
            pred_text_path[t["text"]] = t["path"]
    for text, count in added_texts.items():
        if len(text) > 5:
            changes.append({
                "type": "text_added",
                "location": pred_text_path.get(text, "unknown"),
                "original": None,
                "modified": text[:100],
                "count": count,
            })

    s_imgs = source_els["imgs"]
    p_imgs = pred_els["imgs"]
    for si in s_imgs:
        matched = None
        for pi in p_imgs:
            if pi["index"] == si["index"]:
                matched = pi
                break
        if not matched:
            for pi in p_imgs:
                if pi["alt"] and pi["alt"] == si["alt"]:
                    matched = pi
                    break
        if matched:
            if matched["src"] != si["src"] and si["src"]:
                changes.append({
                    "type": "img_src_changed",
                    "location": si["path"],
                    "original": si["src"][:100],
                    "modified": matched["src"][:100],
                    "context": f"alt='{si['alt'][:50]}'"
                })
        elif si["src"]:
            changes.append({
                "type": "img_removed",
                "location": si["path"],
                "original": si["src"][:100],
                "modified": None,
                "context": f"alt='{si['alt'][:50]}'"
            })

    s_links = source_els["links"]
    p_links = pred_els["links"]
    p_link_map = {}
    for pl in p_links:
        if pl["text"] and pl["text"] not in p_link_map:
            p_link_map[pl["text"]] = pl
    for sl in s_links:
        if not sl["text"]:
            continue
        pl = p_link_map.get(sl["text"])
        if pl and pl["href"] != sl["href"]:
            changes.append({
                "type": "link_href_changed",
                "location": sl["path"],
                "original": sl["href"][:100],
                "modified": pl["href"][:100],
                "context": f"text='{sl['text'][:50]}'"
            })
        elif not pl:
            changes.append({
                "type": "link_removed",
                "location": sl["path"],
                "original": sl["href"][:80],
                "modified": None,
                "context": f"text='{sl['text'][:50]}'"
            })

    s_styled_map = {}
    for s in source_els["styled"]:
        key = f"{s['tag']}.{s['classes']}|{s['text_hint']}"
        if key not in s_styled_map:
            s_styled_map[key] = s

    p_styled_map = {}
    for p in pred_els["styled"]:
        key = f"{p['tag']}.{p['classes']}|{p['text_hint']}"
        if key not in p_styled_map:
            p_styled_map[key] = p

    for key, ss in s_styled_map.items():
        ps = p_styled_map.get(key)
        if not ps:
            continue
        for prop, sval in ss["props"].items():
            pval = ps["props"].get(prop, "")
            if pval != sval:
                changes.append({
                    "type": "style_changed",
                    "location": ss["path"],
                    "original": f"{prop}: {sval}",
                    "modified": f"{prop}: {pval}" if pval else f"{prop}: (removed)",
                    "context": f"element='{ss['text_hint'][:40]}'"
                })
        for prop, pval in ps["props"].items():
            if prop not in ss["props"]:
                changes.append({
                    "type": "style_added",
                    "location": ps["path"],
                    "original": None,
                    "modified": f"{prop}: {pval}",
                    "context": f"element='{ps['text_hint'][:40]}'"
                })

    s_ids = {(e["tag"], e["id"], e["class"]): e for e in source_els["identifiable"] if e["id"]}
    p_ids = {(e["tag"], e["id"], e["class"]): e for e in pred_els["identifiable"] if e["id"]}
    for key, se in s_ids.items():
        if key not in p_ids:
            changes.append({
                "type": "element_removed",
                "location": se["path"],
                "original": f"<{se['tag']} id='{se['id']}' class='{se['class'][:50]}'>",
                "modified": None,
                "context": se["text_hint"],
            })

    s_css = {}
    for rule in source_els["css_rules"]:
        sel = rule["selector"]
        if sel not in s_css:
            s_css[sel] = {}
        s_css[sel].update(rule["props"])

    p_css = {}
    for rule in pred_els["css_rules"]:
        sel = rule["selector"]
        if sel not in p_css:
            p_css[sel] = {}
        p_css[sel].update(rule["props"])

    for sel, sprops in s_css.items():
        pprops = p_css.get(sel, {})
        for prop, sval in sprops.items():
            pval = pprops.get(prop)
            if pval is None:
                changes.append({
                    "type": "css_rule_removed",
                    "location": f"<style> {sel}",
                    "original": f"{prop}: {sval}"[:100],
                    "modified": None,
                })
            elif pval != sval:
                changes.append({
                    "type": "css_rule_changed",
                    "location": f"<style> {sel}",
                    "original": f"{prop}: {sval}"[:100],
                    "modified": f"{prop}: {pval}"[:100],
                })
    for sel, pprops in p_css.items():
        if sel not in s_css:
            prop_str = "; ".join(f"{k}: {v}" for k, v in list(pprops.items())[:5])
            changes.append({
                "type": "css_rule_added",
                "location": f"<style> {sel}",
                "original": None,
                "modified": prop_str[:100],
            })

    if len(changes) > max_changes:
        priority = {"text_removed": 3, "img_src_changed": 3, "img_removed": 3,
                     "css_rule_removed": 3, "css_rule_changed": 3,
                     "link_removed": 2, "link_href_changed": 2, "style_changed": 2,
                     "element_removed": 2, "css_rule_added": 1, "style_added": 1, "text_added": 0}
        changes.sort(key=lambda c: -priority.get(c["type"], 0))
        changes = changes[:max_changes]

    return changes


def extract_edit_descriptions(rubric: dict, scoring_policy: str = "final_state") -> list[str]:
    active = fs.active_items(rubric, scoring_policy)
    return [fs.final_context(rubric), "FINAL EFFECTIVE REQUIREMENTS:"] + [
        item.get("expected_visible_evidence", {}).get(
            "after_expected_change", item["question"]
        )
        for item in active if item["category"] in ("edit_fulfillment", "revision")
    ]


DAMAGE_ASSESS_PROMPT_EN = """\
You are an expert at detecting web page integrity problems. I will give you a list of changes a web page underwent between before and after an edit, plus a description of the editing task.
Decide, for each change, whether it is a "reasonable result of the editing task" or "unintended collateral damage".

[EDITING TASK DESCRIPTION]
%EDIT_DESCRIPTIONS%

[CHANGE LIST]
%CHANGE_LIST%

Output in strict JSON format (do not include any other text):

```json
{
  "changes": [
    {"id": 1, "is_damage": false, "severity": null, "reason": "brief reason"},
    {"id": 2, "is_damage": true, "severity": "high", "reason": "brief reason"}
  ],
  "damage_count": 0,
  "overall_assessment": "one-sentence summary"
}
```

severity levels:
- "high": content loss (text disappeared, image deleted) or structural breakage (a section is gone)
- "medium": style / attribute anomaly (color changed outside the target area, a link was modified)
- "low": tiny change (slight style adjustment that does not hurt readability)

Judging principles:
- If the change is explicitly requested by the editing task description, or is a reasonable side effect of that operation -> is_damage=false
- If the change happens in an area completely unrelated to the editing task -> is_damage=true
- If you are not sure -> lean towards is_damage=true (judge strictly)
- Adding/removing text inside the edited area is reasonable, but text disappearing far away from the edited area is damage
"""



def format_changes_for_prompt(changes: list[dict]) -> str:
    lab = {"type": "type", "loc": "location", "orig": "original",
           "mod": "modified", "ctx": "context"}
    lines = []
    for i, c in enumerate(changes, 1):
        parts = [f"[{i}] {lab['type']}: {c['type']}, {lab['loc']}: {c['location']}"]
        if c.get("original"):
            parts.append(f"{lab['orig']}: \"{c['original']}\"")
        if c.get("modified"):
            parts.append(f"{lab['mod']}: \"{c['modified']}\"")
        if c.get("context"):
            parts.append(f"{lab['ctx']}: {c['context']}")
        if c.get("count", 1) > 1:
            parts.append(f"(×{c['count']})")
        lines.append(", ".join(parts))
    return "\n".join(lines)


BATCH_SIZE = 20
SEVERITY_WEIGHTS = {"high": 0.3, "medium": 0.15, "low": 0.05}

LLM_TIMEOUT = 90


def _llm_assess_single_batch(
    changes: list[dict],
    edit_descriptions: list[str],
    gemini_client: GeminiTaskClient,
    model: str,
) -> dict:
    """Assess a single batch of changes. Returns result with _api_failed flag."""
    if not changes:
        return {"changes": [], "damage_count": 0, "_api_failed": False}

    edit_desc_text = "\n".join(f"{i+1}. {d}" for i, d in enumerate(edit_descriptions))
    change_text = format_changes_for_prompt(changes)

    tmpl = DAMAGE_ASSESS_PROMPT_EN
    prompt = tmpl.replace("%EDIT_DESCRIPTIONS%", edit_desc_text)
    prompt = prompt.replace("%CHANGE_LIST%", change_text)

    messages = [{"role": "user", "content": prompt}]

    try:
        response = gemini_client.client.call_text(
            messages, model=model, max_tokens=None,
            include_thoughts=False, retries=20, backoff=5.0, timeout=LLM_TIMEOUT,
        )
        text = response.get("content", "")
        if not text.strip():
            return {"changes": [], "damage_count": 0, "_api_failed": True,
                    "overall_assessment": "API返回空", "error": "empty_response"}

        json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(1))
        else:
            result = json.loads(text)

        result["_api_failed"] = False
        result["raw_response"] = text[:500]
        return result

    except Exception as e:  # noqa: BLE001
        return {"changes": [], "damage_count": 0, "_api_failed": True,
                "overall_assessment": f"Error: {str(e)[:100]}", "error": str(e)[:100]}


def llm_assess_damage(
    changes: list[dict],
    edit_descriptions: list[str],
    gemini_client: GeminiTaskClient,
    model: str,
) -> dict:
    if not changes:
        return {"changes": [], "damage_count": 0, "overall_assessment": "无变化",
                "score": 1.0, "raw_response": "", "api_failures": 0, "batches": 0}

    if len(changes) <= BATCH_SIZE:
        for attempt in range(10):
            result = _llm_assess_single_batch(changes, edit_descriptions, gemini_client, model)
            if not result.get("_api_failed"):
                break
            time.sleep(min(60, 5 * (attempt + 1)))
        if result.get("_api_failed"):
            raise RuntimeError(f"API failed after 10 attempts: {result.get('error', 'unknown')}")

        damage_count = 0
        total_weight = 0.0
        for c in result.get("changes", []):
            if c.get("is_damage"):
                damage_count += 1
                total_weight += SEVERITY_WEIGHTS.get(c.get("severity", "medium"), 0.15)
        score = max(0.0, 1.0 - total_weight / max(len(changes) * 0.15, 1))
        result["score"] = round(score, 4)
        result["damage_count"] = damage_count
        result["total_changes"] = len(changes)
        result["api_failures"] = 0
        result["batches"] = 1
        return result


    batches = [changes[i:i+BATCH_SIZE] for i in range(0, len(changes), BATCH_SIZE)]
    all_change_results = []
    total_weight = 0.0
    total_damage_count = 0
    api_failures = 0
    assessments = []

    for batch_idx, batch in enumerate(batches):
        result = None
        for attempt in range(10):
            result = _llm_assess_single_batch(batch, edit_descriptions, gemini_client, model)
            if not result.get("_api_failed"):
                break
            time.sleep(min(60, 5 * (attempt + 1)))

        if result.get("_api_failed"):
            raise RuntimeError(
                f"API failed after 10 attempts on batch {batch_idx+1}/{len(batches)}: "
                f"{result.get('error', 'unknown')}"
            )

        for c in result.get("changes", []):
            all_change_results.append(c)
            if c.get("is_damage"):
                total_damage_count += 1
                total_weight += SEVERITY_WEIGHTS.get(c.get("severity", "medium"), 0.15)
        if result.get("overall_assessment"):
            assessments.append(result["overall_assessment"][:80])

    total_changes = len(changes)
    score = max(0.0, 1.0 - total_weight / max(total_changes * 0.15, 1))

    return {
        "changes": all_change_results,
        "damage_count": total_damage_count,
        "total_changes": total_changes,
        "overall_assessment": "; ".join(assessments)[:200],
        "score": round(score, 4),
        "api_failures": api_failures,
        "batches": len(batches),
        "raw_response": "",
    }


def check_no_damage_v3(
    source_html: str, pred_html: str, rubric: dict,
    gemini_client: GeminiTaskClient, model: str,
    scoring_policy: str = "final_state",
) -> dict:
    changes = html_structured_diff(source_html, pred_html)
    edit_descriptions = extract_edit_descriptions(rubric, scoring_policy)
    llm_result = llm_assess_damage(changes, edit_descriptions, gemini_client, model)

    return {
        "diff_count": len(changes),
        "changes_summary": [
            {"type": c["type"], "location": c["location"],
             "original": (c.get("original") or "")[:60], "modified": (c.get("modified") or "")[:60]}
            for c in changes[:20]
        ],
        "llm_result": {
            "score": llm_result.get("score", 1.0),
            "damage_count": llm_result.get("damage_count", 0),
            "total_changes": llm_result.get("total_changes", 0),
            "overall_assessment": llm_result.get("overall_assessment", ""),
            "damages": [c for c in llm_result.get("changes", []) if c.get("is_damage")],
            "api_failures": llm_result.get("api_failures", 0),
            "batches": llm_result.get("batches", 0),
        },
        "overall_score": llm_result.get("score", 1.0),
    }



def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Score No-damage for Track A/C outputs.")
    parser.add_argument("--organized-root", required=True)
    parser.add_argument("--contestant-models", nargs="+", required=True)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--output", default="")
    parser.add_argument("--prompt-lang", choices=["en"], default="en")
    parser.add_argument("--scoring-policy", choices=["final_state"],
                        default="final_state")
    args = parser.parse_args()
    global BATCH_SIZE
    BATCH_SIZE = args.batch_size
    root = Path(args.organized_root)
    output_path = Path(args.output) if args.output else root / "no_damage_results_v3.json"
    previous = {}
    if output_path.exists():
        for row in json.loads(output_path.read_text(encoding="utf-8")):
            if "overall_score" in row:
                previous[(row["sample_id"], row["contestant_model"])] = row

    tasks = []
    results = []
    for model in args.contestant_models:
        model_dir = root / "predictions" / model
        if not model_dir.exists():
            continue
        for prediction_dir in sorted(item for item in model_dir.iterdir() if item.is_dir()):
            sid = prediction_dir.name
            source_path = root / sid / "source_page" / "index.html"
            prediction_path = prediction_dir / "modified_page" / "index.html"
            rubric_path = root / sid / "judge_rubric.json"
            if not all(path.exists() for path in (source_path, prediction_path, rubric_path)):
                continue
            rubric = json.loads(rubric_path.read_text(encoding="utf-8"))
            fs.validate_view(rubric)
            signature = fs.input_signature(rubric, source_path, prediction_path)
            prior = previous.get((sid, model))
            if prior and prior.get("input_signature") == signature:
                results.append(prior)
            else:
                tasks.append((sid, model, source_path, prediction_path, rubric_path))

    if tasks:
        client = make_judge_client(args.judge_model)

        def run_one(task):
            sid, model, source_path, prediction_path, rubric_path = task
            rubric = json.loads(rubric_path.read_text(encoding="utf-8"))
            result = check_no_damage_v3(
                source_path.read_text(encoding="utf-8", errors="ignore"),
                prediction_path.read_text(encoding="utf-8", errors="ignore"),
                rubric, client, args.judge_model, "final_state",
            )
            result.update(
                sample_id=sid,
                contestant_model=model,
                scoring_policy="final_state",
                evaluation_status="scored",
                input_signature=fs.input_signature(rubric, source_path, prediction_path),
            )
            return result

        with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
            futures = {pool.submit(run_one, task): task for task in tasks}
            for index, future in enumerate(as_completed(futures), 1):
                sid, model, *_ = futures[future]
                try:
                    row = future.result()
                except Exception as exc:  # noqa: BLE001
                    row = {
                        "sample_id": sid,
                        "contestant_model": model,
                        "scoring_policy": "final_state",
                        "evaluation_status": "judge_error",
                        "error": str(exc),
                    }
                results.append(row)
                output_path.write_text(
                    json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                print(f"[{index}/{len(tasks)}] {sid}: {row['evaluation_status']}", flush=True)
    results.sort(key=lambda row: (row.get("contestant_model", ""), row.get("sample_id", "")))
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")


if __name__ == "__main__":
    main()
