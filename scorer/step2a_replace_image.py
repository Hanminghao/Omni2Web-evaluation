#!/usr/bin/env python3
"""Verify replace_image judge results at the code level.

For each edit_fulfillment item that involves image replacement:
1. Use before_anchor keywords to locate the target image in source HTML
2. Find the corresponding image in predicted HTML by fingerprint (index, class, alt)
3. Check if src actually changed (not just CSS filter/gradient added)
4. Override judge met=true → false when replacement is fake
"""

import json
import re
from pathlib import Path

from bs4 import BeautifulSoup

from scorer import final_state as fs


def _get_context(el, depth=3) -> str:
    """Get text context around an element (self + ancestors up to depth)."""
    parts = []
    cur = el
    for _ in range(depth + 1):
        if not cur or not getattr(cur, "name", None):
            break
        cls = " ".join(cur.get("class", []))
        cid = cur.get("id", "")
        alt = cur.get("alt", "")
        text = cur.get_text(strip=True)[:100] if cur == el else ""
        parts.append(f"{cur.name} {cls} {cid} {alt} {text}")
        cur = cur.parent
    return " ".join(parts).lower()




CRITERION = "en"





_EN_STOPWORDS = {
    "the", "a", "an", "this", "that", "these", "those", "and", "or", "of", "in", "on", "at",
    "to", "for", "with", "by", "from", "is", "are", "was", "were", "be", "been", "its", "it",
    "as", "here", "there", "above", "below", "left", "right", "top", "bottom", "first",
    "second", "third", "last", "one", "two", "page", "section", "block", "element", "area",
    "part", "should", "must", "not", "no", "new", "old", "same", "step", "target", "then",
}


def _extract_anchor_keywords(anchor: str) -> list[str]:
    """Extract meaningful keywords from before_anchor text."""
    parts = re.split(r"[，,、。\s'\"''""()（）]+", anchor)
    kws = [p.strip().lower() for p in parts if len(p.strip()) >= 2]
    kws = [k for k in kws if not (k.isascii() and k in _EN_STOPWORDS)]
    return kws


def _keyword_score(context: str, keywords: list[str]) -> int:
    return sum(1 for kw in keywords if kw in context)


def _get_element_filter(el) -> str:
    """Get CSS filter value from element's inline style."""
    style = el.get("style", "")
    m = re.search(r'filter\s*:\s*([^;]+)', style, re.IGNORECASE)
    return m.group(1).strip() if m else ""


def _get_element_classes(el) -> set[str]:
    return set(el.get("class", []))


def _extract_bg_url(style: str) -> str:
    """Extract first url() from a style string."""
    m = re.search(r'url\(["\']?([^"\')\s]+)["\']?\)', style)
    return m.group(1) if m else ""


def collect_images(html_content: str) -> list[dict]:
    """Collect all image-like elements with their context and fingerprint."""
    soup = BeautifulSoup(html_content, "html.parser")
    images = []

    for i, img in enumerate(soup.find_all("img")):
        src = img.get("src", "")
        alt = img.get("alt", "")
        cls = " ".join(img.get("class", []))
        parent_cls = " ".join(img.parent.get("class", [])) if img.parent else ""
        css_filter = _get_element_filter(img)
        context = _get_context(img)

        images.append({
            "index": i, "type": "img", "src": src, "alt": alt,
            "classes": cls, "parent_classes": parent_cls,
            "filter": css_filter, "context": context,
        })

    bg_idx = 0
    for el in soup.find_all(True):
        style = el.get("style", "")
        bg_url = _extract_bg_url(style)
        if bg_url:
            cls = " ".join(el.get("class", []))
            css_filter = _get_element_filter(el)
            context = _get_context(el)
            images.append({
                "index": bg_idx, "type": "bg", "src": bg_url, "alt": "",
                "classes": cls, "parent_classes": "",
                "filter": css_filter, "context": context,
            })
            bg_idx += 1

    style_tags = soup.find_all("style")
    css_text = "\n".join(st.string or "" for st in style_tags)

    bg_idx_css = 0
    for m in re.finditer(r'([^{]+)\{([^}]*)\}', css_text):
        selector = m.group(1).strip()
        body = m.group(2)
        bg_url = _extract_bg_url(body)
        if bg_url:
            filter_m = re.search(r'filter\s*:\s*([^;]+)', body, re.IGNORECASE)
            css_filter = filter_m.group(1).strip() if filter_m else ""
            images.append({
                "index": bg_idx_css, "type": "css_bg", "src": bg_url, "alt": "",
                "classes": selector, "parent_classes": "",
                "filter": css_filter, "context": selector.lower(),
            })
            bg_idx_css += 1

    return images


def collect_all_filters(html_content: str) -> dict[str, str]:
    """Collect all CSS filter declarations (inline + <style> tags)."""
    soup = BeautifulSoup(html_content, "html.parser")
    filters = {}

    for el in soup.find_all(True):
        f = _get_element_filter(el)
        if f:
            cls = " ".join(el.get("class", []))
            key = f"{el.name}.{cls}" if cls else el.name
            filters[key] = f

    for st in soup.find_all("style"):
        text = st.string or ""
        for m in re.finditer(r'([^{]+)\{[^}]*filter\s*:\s*([^;]+)', text, re.IGNORECASE):
            filters[m.group(1).strip()] = m.group(2).strip()

    return filters


def find_target_image(source_images: list[dict], anchor_text: str) -> dict | None:
    """Locate the target image in source HTML using before_anchor keywords."""
    keywords = _extract_anchor_keywords(anchor_text)
    if not keywords:
        return None

    scored = []
    for img in source_images:
        score = _keyword_score(img["context"], keywords)
        if score > 0:
            scored.append((score, img))

    if not scored:
        return None

    scored.sort(key=lambda x: -x[0])
    return scored[0][1]


def find_matching_image(target: dict, pred_images: list[dict]) -> dict | None:
    """Find the corresponding image in pred HTML by fingerprint matching."""
    same_type = [p for p in pred_images if p["type"] == target["type"]]

    if target["type"] == "img":
        for p in same_type:
            if p["index"] == target["index"]:
                return p
        for p in same_type:
            if p["alt"] and p["alt"] == target["alt"]:
                return p
        for p in same_type:
            if p["classes"] and p["classes"] == target["classes"]:
                return p
    elif target["type"] in ("bg", "css_bg"):
        for p in same_type:
            if p["classes"] == target["classes"]:
                return p
        for p in same_type:
            if p["index"] == target["index"]:
                return p

    return None








_REPLACE_IMAGE_STEP_TYPES = {"replace_image", "swap_image", "change_image"}
_REPLACE_IMAGE_KW_ZH = ["图片", "配图", "背景图", "照片", "背景替换", "替换为一张", "替换为一段",
                        "背景替换为", "图替换", "视觉区域的背景"]
_REPLACE_IMAGE_KW_EN = ["image", "picture", "photo", "photograph", "artwork", "illustration",
                        "thumbnail", "banner", "avatar"]


def is_replace_image_item(item: dict) -> bool:
    if item.get("category") != "edit_fulfillment":
        return False
    q = item.get("question", "")
    ev = item.get("expected_visible_evidence", {})
    after = ev.get("after_expected_change", "")
    combined = q + " " + after
    step_type = str(item.get("step_type") or "").strip().lower()
    if step_type:
        return step_type in _REPLACE_IMAGE_STEP_TYPES
    low = combined.lower()
    return (any(kw in combined for kw in _REPLACE_IMAGE_KW_ZH)
            or any(kw in low for kw in _REPLACE_IMAGE_KW_EN))


FILTER_FUNCTIONS = {"grayscale", "sepia", "brightness", "contrast", "saturate",
                    "hue-rotate", "invert", "blur", "opacity"}


def is_filter_only_change(source_filter: str, pred_filter: str, src_changed: bool) -> bool:
    """Check if the only change is adding/modifying CSS filter without changing src."""
    if src_changed:
        return False
    return bool(pred_filter and pred_filter != source_filter)


def is_gradient_only(source_src: str, pred_src: str, pred_context: str) -> bool:
    """Check if image was replaced with a CSS gradient instead of a real image."""
    if not pred_src and "gradient" in pred_context:
        return True
    return bool(pred_src and ("gradient" in pred_src.lower() or pred_src.startswith(("linear-", "radial-"))))


def verify_replace_image(
    source_html: str, pred_html: str, rubric: dict, judge_items: list[dict],
    scoring_policy: str = "final_state",
) -> list[dict]:
    source_images = collect_images(source_html)
    pred_images = collect_images(pred_html)
    source_filters = collect_all_filters(source_html)
    pred_filters = collect_all_filters(pred_html)

    source_all_srcs = {img["src"] for img in source_images if img["src"]}
    pred_all_srcs = {img["src"] for img in pred_images if img["src"]}
    any_new_srcs = bool(pred_all_srcs - source_all_srcs)

    new_filter_keys = set()
    for key, val in pred_filters.items():
        if key not in source_filters or val != source_filters[key]:
            new_filter_keys.add(key)

    corrections = []
    rubric_map = {it["item_id"]: it for it in fs.active_items(rubric, scoring_policy)}

    for ji in judge_items:
        iid = ji["item_id"]
        rb = rubric_map.get(iid, {})
        if not is_replace_image_item(rb):
            continue

        original_met = ji.get("met", False)
        reason = ji.get("reason", "")
        anchor = rb.get("expected_visible_evidence", {}).get("before_anchor", "")

        target = find_target_image(source_images, anchor)
        src_changed = False
        filter_only = False
        target_found = target is not None
        method = "none"
        source_src = ""
        pred_src = ""
        source_filter_val = ""
        pred_filter_val = ""

        if target:
            pred_match = find_matching_image(target, pred_images)
            source_src = target["src"]
            source_filter_val = target["filter"]

            if pred_match:
                pred_src = pred_match["src"]
                pred_filter_val = pred_match["filter"]
                src_changed = pred_src != source_src and bool(pred_src)
                filter_only = is_filter_only_change(source_filter_val, pred_filter_val, src_changed)
                method = "matched"
            else:
                method = "no_pred_match"
        else:
            method = "no_target_found"

        new_met = original_met
        correction_reason = ""

        if original_met:
            if method == "matched" and not src_changed and filter_only:
                new_met = False
                correction_reason = f"CSS filter变化({pred_filter_val})但src未替换: {source_src[:60]}"
            elif method == "matched" and not src_changed and not filter_only:
                if new_filter_keys and not any_new_srcs:
                    new_met = False
                    correction_reason = "全局无新图片src，仅有CSS filter变化"
                elif not src_changed and source_src == pred_src:
                    new_met = False
                    correction_reason = f"目标图片src完全未变: {source_src[:60]}"
            elif method == "no_pred_match":
                pass
            elif method == "no_target_found":
                if not any_new_srcs and new_filter_keys:
                    new_met = False
                    correction_reason = "无法定位目标图片，全局无新图片src，仅有CSS filter变化"
                else:
                    reason_lower = reason.lower()
                    filter_kws = ["filter", "grayscale", "sepia", "brightness", "contrast",
                                  "saturate", "hue-rotate", "invert"]
                    gradient_kws = ["gradient", "radial-gradient", "linear-gradient"]
                    src_kws = ["src", "url(", "链接", "替换为", "新的图片", "新的照片",
                               "photo-", "unsplash"]
                    has_filter = any(kw in reason_lower for kw in filter_kws)
                    has_gradient = any(kw in reason_lower for kw in gradient_kws)
                    has_real_src = any(kw in reason_lower for kw in src_kws)
                    if has_filter and not has_real_src:
                        new_met = False
                        correction_reason = "[reason fallback] judge提到CSS filter但无src替换证据"
                    elif has_gradient and not has_real_src:
                        new_met = False
                        correction_reason = "[reason fallback] judge提到CSS渐变但无真实图片替换证据"

        corrections.append({
            "item_id": iid,
            "is_replace_image": True,
            "original_met": original_met,
            "corrected_met": new_met,
            "correction_reason": correction_reason,
            "original_reason": ji.get("reason", "")[:150],
            "method": method,
            "target_found": target_found,
            "src_changed": src_changed,
            "filter_only": filter_only,
            "source_src": source_src[:80],
            "pred_src": pred_src[:80],
            "source_filter": source_filter_val[:50],
            "pred_filter": pred_filter_val[:50],
            "any_new_srcs_global": any_new_srcs,
            "new_filter_keys_count": len(new_filter_keys),
        })

    return corrections



def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Verify image-replacement rubric items.")
    parser.add_argument("--organized-root", required=True)
    parser.add_argument("--contestant-models", nargs="+", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--prompt-lang", choices=["en"], default="en")
    parser.add_argument("--scoring-policy", choices=["final_state"],
                        default="final_state")
    args = parser.parse_args()
    root = Path(args.organized_root)
    judgments = json.loads((root / "judge_results_rubric.json").read_text(encoding="utf-8"))
    requested = set(args.contestant_models)
    corrections = []
    for judgment in judgments:
        sid = judgment.get("sample_id")
        model = judgment.get("contestant_model")
        if model not in requested or judgment.get("evaluation_status") != "scored":
            continue
        rubric_path = root / sid / "judge_rubric.json"
        source_path = root / sid / "source_page" / "index.html"
        prediction_path = root / "predictions" / model / sid / "modified_page" / "index.html"
        if not all(path.exists() for path in (rubric_path, source_path, prediction_path)):
            continue
        rubric = json.loads(rubric_path.read_text(encoding="utf-8"))
        fs.validate_view(rubric)
        current = verify_replace_image(
            source_path.read_text(encoding="utf-8", errors="ignore"),
            prediction_path.read_text(encoding="utf-8", errors="ignore"),
            rubric,
            judgment.get("items") or [],
            "final_state",
        )
        for item in current:
            item.update(sample_id=sid, contestant_model=model)
        corrections.extend(current)
    output = Path(args.output) if args.output else root / "replace_image_verification.json"
    output.write_text(json.dumps(corrections, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")


if __name__ == "__main__":
    main()
