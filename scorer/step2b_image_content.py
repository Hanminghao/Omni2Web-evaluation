#!/usr/bin/env python3
"""Verify replace_image content by downloading images and asking Gemini to check.

For items where src actually changed (matched + src_changed=true + corrected_met=true),
download both original and new images, send to Gemini for content verification.

This is Step 3b in the eval pipeline — runs after verify_replace_image.py.
"""

import base64
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from scorer import final_state as fs
from scorer._types import GeminiTaskClient
from scorer.llm_client import DEFAULT_JUDGE_MODEL, make_judge_client

VERIFY_PROMPT_EN = """\
You are an expert at verifying image content. You will see two images and one description.

[TASK] Decide:
1. Is the new image really different from the original image (not the same image)?
2. Does the new image match the content description below?

[CONTENT DESCRIPTION]
%DESCRIPTION%

Output in strict JSON format (do not include any other text):

```json
{
  "is_different": true/false,
  "matches_description": true/false,
  "image_content": "briefly state what the new image actually shows (under 20 words)",
  "reason": "your justification (under 30 words)"
}
```

Notes:
- If the new image is a solid color / gradient / abstract pattern rather than a real photo, matches_description must be false
- If the description asks for a specific subject (e.g. "sculpture", "architecture"), the new image must contain that subject to count as a match
- Judge strictly: when in doubt, answer false
"""



def download_image(url: str, timeout: int = 15, max_width: int = 512) -> bytes | None:
    """Download image from URL, return bytes or None on failure."""
    try:
        if not url or not url.startswith("http"):
            return None
        if "unsplash.com" in url:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}w={max_width}&q=80"
        resp = requests.get(url, timeout=timeout, headers={
            "User-Agent": "Mozilla/5.0 (evaluation-pipeline)"
        })
        if resp.status_code == 200 and len(resp.content) > 1000:
            return resp.content
        return None
    except Exception:  # noqa: BLE001
        return None


def image_to_base64_url(img_bytes: bytes) -> str:
    """Convert image bytes to data URL for API."""
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    if img_bytes[:4] == b'\x89PNG':
        mime = "image/png"
    elif img_bytes[:2] == b'\xff\xd8':
        mime = "image/jpeg"
    elif img_bytes[:4] == b'RIFF':
        mime = "image/webp"
    else:
        mime = "image/jpeg"
    return f"data:{mime};base64,{b64}"


def verify_image_content(
    gemini_client: GeminiTaskClient,
    model: str,
    source_img_bytes: bytes | None,
    pred_img_bytes: bytes | None,
    description: str,
) -> dict:
    """Send images to Gemini for content verification."""
    if pred_img_bytes is None:
        return {"is_different": False, "matches_description": False,
                "reason": "新图片无法下载", "image_content": "N/A"}

    prompt = VERIFY_PROMPT_EN.replace("%DESCRIPTION%", description)
    content_parts = []

    if source_img_bytes:
        content_parts.append({"type": "text", "text": "[ORIGINAL IMAGE]"})
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": image_to_base64_url(source_img_bytes)}
        })

    content_parts.append({"type": "text", "text": "[NEW IMAGE]"})
    content_parts.append({
        "type": "image_url",
        "image_url": {"url": image_to_base64_url(pred_img_bytes)}
    })

    content_parts.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content_parts}]

    try:
        response = gemini_client.client.call_text(
            messages, model=model, max_tokens=1024,
            include_thoughts=False, retries=20, backoff=5.0,
        )
        text = response.get("content", "") or response.get("text", "")
        if not text.strip():
            return {"is_different": False, "matches_description": False,
                    "reason": "API返回空", "image_content": "N/A"}

        json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
        return json.loads(text)
    except Exception as e:  # noqa: BLE001
        return {"is_different": False, "matches_description": False,
                "reason": f"API error: {str(e)[:50]}", "image_content": "N/A"}



def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Verify replacement-image content.")
    parser.add_argument("--organized-root", required=True)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--output", default="")
    parser.add_argument("--prompt-lang", choices=["en"], default="en")
    parser.add_argument("--scoring-policy", choices=["final_state"],
                        default="final_state")
    args = parser.parse_args()
    root = Path(args.organized_root)
    candidates = json.loads(
        (root / "replace_image_verification.json").read_text(encoding="utf-8")
    )
    active = {}
    for item in candidates:
        sid = item["sample_id"]
        if sid not in active:
            rubric = json.loads((root / sid / "judge_rubric.json").read_text(encoding="utf-8"))
            active[sid] = {entry["item_id"] for entry in fs.active_items(
                rubric, "final_state"
            )}
    candidates = [
        item for item in candidates
        if item["item_id"] in active[item["sample_id"]]
        and item["corrected_met"] and item["src_changed"]
        and item["source_src"] and item["pred_src"]
    ]
    output_path = Path(args.output) if args.output else root / "image_content_verification.json"
    if not candidates:
        output_path.write_text("[]\n", encoding="utf-8")
        return
    client = make_judge_client(args.judge_model)
    rubric_cache = {}

    def description(sid, item_id):
        if sid not in rubric_cache:
            rubric_cache[sid] = json.loads(
                (root / sid / "judge_rubric.json").read_text(encoding="utf-8")
            )
        for item in rubric_cache[sid]["items"]:
            if item["item_id"] == item_id:
                return ((item.get("expected_visible_evidence") or {})
                        .get("after_expected_change", ""))
        return ""

    def verify(candidate):
        sid, item_id = candidate["sample_id"], candidate["item_id"]
        source_bytes = download_image(candidate["source_src"])
        prediction_bytes = download_image(candidate["pred_src"])
        result = verify_image_content(
            client,
            args.judge_model,
            source_bytes,
            prediction_bytes,
            description(sid, item_id),
        )
        result.update(
            sample_id=sid,
            contestant_model=candidate["contestant_model"],
            item_id=item_id,
            source_src=candidate["source_src"],
            pred_src=candidate["pred_src"],
            source_downloaded=source_bytes is not None,
            pred_downloaded=prediction_bytes is not None,
        )
        return result

    results = []
    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        for future in as_completed([pool.submit(verify, item) for item in candidates]):
            results.append(future.result())
    results.sort(key=lambda row: (row["contestant_model"], row["sample_id"], row["item_id"]))
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")


if __name__ == "__main__":
    main()
