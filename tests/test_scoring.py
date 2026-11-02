import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import run_eval
from predict_example import GeminiPredictor
from prediction_prompts import build_track_a_prompt, build_track_b_prompt
from scorer import e2e_report, final_state
from scorer.io_utils import extract_html, load_jsonl, load_predictions, source_html
from scorer.step2a_replace_image import verify_replace_image
from scorer.step3_no_damage import check_no_damage_v3, html_structured_diff
from scorer.step_rewrite_judge import (
    component_f1,
    derive_scores,
    judge_rewrite,
    parse_rewrite,
    validate_judgment,
)

ROOT = Path(__file__).parents[1]
BENCHMARK = ROOT / "tests" / "fixtures" / "benchmark.jsonl"


def benchmark():
    return load_jsonl(BENCHMARK)


def test_final_state_excludes_reverted_step():
    rubric = benchmark()[0]["rubric"]
    active = {item["item_id"] for item in final_state.active_items(rubric)}
    assert active == {"r3", "r4", "r5"}
    assert rubric["excluded_steps"][0]["step"] == 1

    returned = [
        {"item_id": item["item_id"], "met": True, "reason": "test"}
        for item in rubric["items"]
    ]
    audit, scored = final_state.normalize_results(returned, rubric)
    assert {item["item_id"] for item in scored} == active
    assert {item["item_id"] for item in audit
            if item["evaluation_status"] == "skipped_reverted"} == {"r1", "r2"}


def test_revision_links_and_exclusions_must_agree():
    rubric = json.loads(json.dumps(benchmark()[0]["rubric"]))
    rubric["revision_links"][0]["reverted_steps"] = []
    with pytest.raises(ValueError, match="Invalid or duplicate revision link"):
        final_state.validate_view(rubric)


def test_benchmark_example_uses_public_schema():
    record = benchmark()[0]
    assert set(record) == {
        "sample_id", "video", "transcript", "source_page", "metadata", "rubric"
    }
    assert set(record["rubric"]) == {"items", "revision_links", "excluded_steps"}


def test_prediction_prompts_cover_omni_video_and_frames():
    omni = build_track_a_prompt(
        html="<html>source</html>", transcript="ignored", input_mode="omni"
    )
    video = build_track_a_prompt(
        html="<html>source</html>", transcript="change this", input_mode="vl-video"
    )
    frames = build_track_b_prompt(
        transcript="change this", input_mode="vl-frames", n_frames=12
    )
    assert "listen to the speech" in omni and "<html>source</html>" in omni
    assert "attached video is SILENT" in video and "change this" in video
    assert "12 consecutive frames" in frames and "change this" in frames


def test_prediction_example_uses_official_gemini_request(monkeypatch):
    captured = {}

    class Response:
        status_code = 200
        text = ""

        def json(self):
            return {
                "candidates": [{"content": {"parts": [{"text": "result"}]}}],
                "usageMetadata": {"totalTokenCount": 3},
            }

    def request(method, url, **kwargs):
        captured.update(method=method, url=url, **kwargs)
        return Response()

    monkeypatch.setattr("predict_example.requests.request", request)
    client = GeminiPredictor("gemini-test", api_key="test-key")
    result = client.generate("prompt", [{"inlineData": {
        "mimeType": "image/webp", "data": "AA=="
    }}])
    assert result["response"] == "result"
    assert captured["url"].endswith("/models/gemini-test:generateContent")
    assert captured["headers"]["x-goog-api-key"] == "test-key"
    assert captured["json"]["contents"][0]["parts"][-1] == {"text": "prompt"}


def test_gemini_scorer_supports_streaming_endpoint(monkeypatch):
    from scorer.llm_client import GeminiClient

    captured = {}

    class Response:
        status_code = 200
        text = ""

        def iter_lines(self, decode_unicode=False):
            assert decode_unicode
            return iter([
                'data: {"candidates":[{"content":{"parts":[{"text":"yes"}]}}]}',
                'data: {"usageMetadata":{"totalTokenCount":3}}',
            ])

    def post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return Response()

    monkeypatch.setattr("scorer.llm_client.requests.post", post)
    client = GeminiClient(
        "test-key", "gemini-test", rpm=0,
        api_root="https://example.test/v1beta", stream=True,
    )
    result = client.call_text(
        [{"role": "user", "content": "prompt"}], model="gemini-test", retries=1
    )
    assert result["content"] == "yes"
    assert captured["url"].endswith(
        "/models/gemini-test:streamGenerateContent?alt=sse"
    )


def test_html_input_helpers():
    record = benchmark()[0]
    assert "Welcome" in source_html(record)
    assert extract_html("prefix```html\n<html><p>x</p></html>\n```suffix") == "<html><p>x</p></html>"


def test_load_predictions_accepts_single_jsonl_record(tmp_path):
    path = tmp_path / "prediction.jsonl"
    path.write_text(json.dumps({
        "sample_id": "001", "ok": True, "response": "<html>result</html>"
    }) + "\n", encoding="utf-8")
    assert load_predictions(path) == {"001": "<html>result</html>"}


def test_source_html_resolves_webvr_path(tmp_path):
    webvr = tmp_path / "WebVR"
    page = webvr / "test" / "101" / "101_1.html"
    page.parent.mkdir(parents=True)
    page.write_text("<html><body>WebVR source</body></html>", encoding="utf-8")
    record = {
        "sample_id": "001",
        "source_page": {"html_path": "test/101/101_1.html"},
    }
    assert "WebVR source" in source_html(record, webvr)
    with pytest.raises(ValueError, match="provide --webvr-root"):
        source_html(record)


def test_efs_renormalizes_available_categories():
    result = final_state.aggregate_items([
        {"category": "target_grounding", "met": True},
        {"category": "revision", "met": False},
    ], no_damage=1.0)
    expected = (0.1 * 1.0 + 0.2 * 0.0 + 0.2 * 1.0) / 0.5
    assert result["efs_weighted_v3"] == round(expected, 4)


def test_track_b_component_f1_penalizes_extra_steps():
    scores = derive_scores(2, 1, 1, n_reference=2, n_predicted=3)
    assert scores["target_f1"] == pytest.approx(0.8)
    assert scores["action_f1"] == pytest.approx(0.4)
    assert scores["irs"] == pytest.approx(0.6)
    assert component_f1(0, 2, 0) == 0.0


def test_track_b_matching_constraints():
    valid = {"steps": [
        {"gt_step": 1, "matched_contestant_step": 1,
         "grounding_met": 1, "instruction_met": 1},
        {"gt_step": 2, "matched_contestant_step": None,
         "grounding_met": 0, "instruction_met": 0},
    ]}
    steps, extras = validate_judgment(valid, 2, 2)
    assert len(steps) == 2 and extras == [2]
    duplicate = {"steps": [valid["steps"][0], {
        "gt_step": 2, "matched_contestant_step": 1,
        "grounding_met": 1, "instruction_met": 1,
    }]}
    with pytest.raises(ValueError, match="more than once"):
        validate_judgment(duplicate, 2, 2)
    with pytest.raises(ValueError, match="omitted"):
        validate_judgment({"steps": valid["steps"][:1]}, 2, 2)
    invalid_binary = {"steps": [
        {"gt_step": 1, "matched_contestant_step": 1,
         "grounding_met": 2, "instruction_met": 1},
        valid["steps"][1],
    ]}
    with pytest.raises(ValueError, match="must be binary"):
        validate_judgment(invalid_binary, 2, 2)


def test_track_b_semantic_retry_and_canonical_denominator():
    rubric = json.loads(json.dumps(benchmark()[1]["rubric"]))
    rubric["items"][0]["expected_visible_evidence"]["before_anchor"] = ""
    responses = iter([
        {"content": json.dumps({"steps": [
            {"gt_step": 1, "matched_contestant_step": 2,
             "grounding_met": 1, "instruction_met": 1}
        ]})},
        {"content": json.dumps({"steps": []})},
    ])
    client = SimpleNamespace(client=SimpleNamespace(call_text=lambda *a, **k: next(responses)))
    row = judge_rewrite(client, "judge", rubric, "<html></html>",
                        '[{"target":"x","instruction":"y"}]', "sample", "model", 2)
    assert row["evaluation_status"] == "scored"
    assert row["n_reference_steps"] == 1
    assert row["target_f1"] == 0.0


def test_rewrite_parser_keeps_extra_steps():
    steps, error = parse_rewrite(json.dumps([
        {"target": "a", "instruction": "x"},
        {"target": "b", "instruction": "y"},
    ]))
    assert not error and len(steps) == 2


def test_no_damage_identical_html_and_detected_change():
    html = "<html><body><p id='x'>text</p></body></html>"
    rubric = benchmark()[1]["rubric"]
    result = check_no_damage_v3(html, html, rubric, None, "judge")
    assert result["overall_score"] == 1.0
    assert html_structured_diff(html, html.replace("text", "changed"))


def test_replace_image_filter_only_is_rejected():
    source = '<html><body><section id="hero"><img src="a.jpg" alt="hero"></section></body></html>'
    predicted = '<html><body><section id="hero"><img src="a.jpg" alt="hero" style="filter:grayscale(1)"></section></body></html>'
    rubric = {"items": [{
        "item_id": "r2", "category": "edit_fulfillment", "step_type": "replace_image",
        "expected_visible_evidence": {"before_anchor": "hero image"},
    }]}
    # This helper only needs the active item list; bypass final-state filtering for this unit.
    original = final_state.active_items
    try:
        final_state.active_items = lambda *_: rubric["items"]
        rows = verify_replace_image(source, predicted, rubric,
                                    [{"item_id": "r2", "met": True}], "final_state")
    finally:
        final_state.active_items = original
    assert rows[0]["corrected_met"] is False


def test_track_a_zero_fill_and_judge_failure(tmp_path):
    output = tmp_path
    organized = output / "organized"
    organized.mkdir()
    (organized / "judge_results_v3.json").write_text(json.dumps([
        {"sample_id": "ok", "contestant_model": "m", "evaluation_status": "scored",
         "efs_weighted_v3": 0.8,
         "category_scores_v3": {key: 0.8 for key in run_eval.CATEGORIES}},
        {"sample_id": "judge", "contestant_model": "m", "evaluation_status": "judge_error",
         "efs_weighted_v3": None},
    ]))
    stats = {"sample_ids": ["ok", "missing", "judge"], "n_samples": 3,
             "no_prediction": ["missing"], "html_parse_failed": []}
    summary = run_eval.summarize(output, stats, "m")
    assert summary["n_scored"] == 2
    assert summary["n_judge_failed"] == 1
    assert summary["efs_weighted_v3"] == pytest.approx(0.4)


def test_track_c_uses_paired_dataset_means(tmp_path):
    recovered = [
        {"sample_id": "a", "evaluation_status": "scored", "efs_weighted_v3": 0.4},
        {"sample_id": "b", "evaluation_status": "scored", "efs_weighted_v3": 0.8},
    ]
    oracle = [
        {"sample_id": "a", "evaluation_status": "scored", "efs_weighted_v3": 0.5},
        {"sample_id": "b", "evaluation_status": "scored", "efs_weighted_v3": 1.0},
    ]
    rp, op = tmp_path / "r.json", tmp_path / "o.json"
    rp.write_text(json.dumps(recovered)); op.write_text(json.dumps(oracle))
    summary, _ = e2e_report.build_report(rp, op)
    assert summary["niu"] == pytest.approx(0.8)


def test_track_a_synthetic_orchestration(monkeypatch, tmp_path):
    def fake_judges(organized, model_name, *args):
        rows = []
        for sample_dir in organized.iterdir():
            if not sample_dir.is_dir() or sample_dir.name == "predictions":
                continue
            rows.append({
                "sample_id": sample_dir.name,
                "contestant_model": model_name,
                "evaluation_status": "scored",
                "efs_weighted_v3": 0.75,
                "category_scores_v3": {key: 0.75 for key in run_eval.CATEGORIES},
            })
        (organized / "judge_results_v3.json").write_text(json.dumps(rows))

    monkeypatch.setattr(run_eval, "run_judges", fake_judges)
    output = tmp_path / "track_a"
    summary = run_eval.score_track_a(
        benchmark_path=BENCHMARK,
        predictions_path=ROOT / "examples" / "track_a_predictions.jsonl",
        model_name="synthetic",
        output_dir=output,
    )
    assert summary["efs_weighted_v3"] == 0.75
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["benchmark"]["name"] == "benchmark.jsonl"
    assert str(ROOT) not in json.dumps(manifest)


def test_deepseek_official_nonstream_request(monkeypatch):
    import executor

    captured = {}

    class Response:
        status_code = 200
        def json(self):
            return {"choices": [{"message": {"content": "<html></html>"},
                                  "finish_reason": "stop"}], "usage": {"total_tokens": 3}}

    def post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return Response()

    monkeypatch.setattr(executor.requests, "post", post)
    client = executor.DeepSeekExecutor(api_key="test-key", retries=1, rpm=0)
    result = client.generate("prompt")
    assert result["ok"] is True
    assert captured["url"] == "https://api.deepseek.com/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["json"]["model"] == "deepseek-v4-pro"
    assert captured["json"]["stream"] is False


def test_deepseek_official_stream_response(monkeypatch):
    import executor

    class Response:
        status_code = 200
        def iter_lines(self, decode_unicode=False):
            assert decode_unicode
            return iter([
                'data: {"choices":[{"delta":{"content":"<html>"},"finish_reason":null}]}',
                'data: {"choices":[{"delta":{"content":"</html>"},"finish_reason":"stop"}]}',
                "data: [DONE]",
            ])

    monkeypatch.setattr(executor.requests, "post", lambda *args, **kwargs: Response())
    client = executor.DeepSeekExecutor(api_key="test-key", retries=1, rpm=0, stream=True)
    result = client.generate("prompt")
    assert result["response"] == "<html></html>"
    assert result["finish_reason"] == "stop"


def test_track_b_synthetic_orchestration(monkeypatch, tmp_path):
    import run_eval_rewrite

    class FakeTextClient:
        def call_text(self, messages, **kwargs):
            prompt = messages[0]["content"][0]["text"]
            reference_block = prompt.rsplit("[REFERENCE STEPS]", 1)[1].split(
                "[CONTESTANT STEPS]", 1
            )[0]
            count = reference_block.count('"step":')
            return {"content": json.dumps({
                "steps": [
                    {"gt_step": index, "matched_contestant_step": index,
                     "grounding_met": 1, "instruction_met": 1}
                    for index in range(1, count + 1)
                ]
            })}

    fake = SimpleNamespace(client=FakeTextClient())
    monkeypatch.setattr(run_eval_rewrite, "make_judge_client", lambda *a, **k: fake)
    output = tmp_path / "track_b"
    summary = run_eval_rewrite.score_track_b(
        benchmark_path=BENCHMARK,
        predictions_path=ROOT / "examples" / "track_b_predictions.jsonl",
        model_name="synthetic",
        output_dir=output,
    )
    assert summary["n_scored"] == 2
    assert summary["target_f1"] == 1.0
    assert summary["action_f1"] == 1.0
    assert summary["irs"] == 1.0


def test_track_c_synthetic_executor_orchestration(monkeypatch, tmp_path):
    import run_eval_e2e

    calls = []

    class FakeExecutor:
        def __init__(self, **kwargs):
            self.config = kwargs
        def generate(self, prompt):
            calls.append(prompt)
            return {"ok": True, "response": "<html><body>synthetic</body></html>"}

    monkeypatch.setattr(run_eval_e2e, "DeepSeekExecutor", FakeExecutor)
    output = tmp_path / "track_c"
    summary = run_eval_e2e.run_track_c(
        benchmark_path=BENCHMARK,
        rewrite_predictions_path=ROOT / "examples" / "track_b_predictions.jsonl",
        model_name="synthetic",
        output_dir=output,
        skip_scoring=True,
    )
    assert summary is None
    assert len(calls) == 4
    assert len((output / "recovered_predictions.jsonl").read_text().splitlines()) == 2
    assert len((output / "oracle_predictions.jsonl").read_text().splitlines()) == 2
