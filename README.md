Omni2Web
========

Omni2Web: Benchmarking Audiovisual Website Development.

[Project Page & Leaderboard](https://omni2web-bench.github.io/) | [Hugging Face Dataset](https://huggingface.co/datasets/minghaofdu/Omni2Web) | [Evaluation Code](https://github.com/Hanminghao/Omni2Web-evaluation) | [Citation](#citation)

<img src="https://omni2web-bench.github.io/assets/omni2web-o.svg" width="112" align="right" alt="Omni2Web logo" />

**Abstract:** Omni2Web benchmarks audiovisual website development: models must turn spoken editing requests and cursor movements in screen recordings into the intended webpage changes. The benchmark contains **918 Chinese and English instances across 129 source webpages**, with multi-step requests that require grounding references in visual context and following revisions to recover the user's final intent. Three complementary tracks evaluate **Direct Editing**, **Instruction Recovery**, and **Instruction Utility** through a fixed coding executor. This repository provides the official scoring code for all three tracks and a minimal prediction example supporting audiovisual recordings, silent video with transcripts, and timestamped frames with transcripts.

---

<img src="https://omni2web-bench.github.io/assets/overview.png" width="100%" alt="Omni2Web overview: audiovisual requests, cursor grounding, webpage editing, and the three evaluation tracks" />

## Installation

Python 3.10 or later is recommended.

```bash
git clone https://github.com/Hanminghao/Omni2Web-evaluation.git
cd Omni2Web-evaluation

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set `GEMINI_API_KEY` for scoring. Track C also requires `DEEPSEEK_API_KEY`. The scorer calls the Google Gemini API directly. The Track C executor calls `https://api.deepseek.com/chat/completions` directly.
For a compatible Gemini endpoint, override `GEMINI_API_ROOT`; set
`GEMINI_API_STREAM=true` when that endpoint returns server-sent events. The prediction
example also provides `--inline-video` for endpoints without the Gemini File API.

## Dataset Preparation

### Download the Benchmark

The bundled `examples/benchmark.jsonl` contains the synthetic scoring fixtures, matching the example prediction files.
Download the complete 918-instance benchmark and original audiovisual videos from
[minghaofdu/Omni2Web](https://huggingface.co/datasets/minghaofdu/Omni2Web). Use that dataset directory's root `benchmark.jsonl` for actual evaluation.

```bash
pip install huggingface_hub
hf download minghaofdu/Omni2Web --repo-type dataset --local-dir /path/to/Omni2Web
```

### Annotation Format

The benchmark is JSONL with one object per sample. Each object must contain:

- `sample_id`, `video`, and `transcript`
- `source_page.html_path`, relative to the WebVR checkout
- language, difficulty, duration, and operation metadata in `metadata`
- `rubric.items`
- `rubric.revision_links` and `rubric.excluded_steps`

### Get the Source Webpages

The public Omni2Web JSONL does not redistribute or embed WebVR HTML. Download
[BroAlanTaps/WebVR](https://huggingface.co/datasets/BroAlanTaps/WebVR) separately:

```bash
git lfs install
git clone https://huggingface.co/datasets/BroAlanTaps/WebVR
git -C WebVR checkout 7917c3a2b44a0838eb0acffb80a38ec6030b49b6
```

Each `source_page.html_path` is relative to that checkout. For example,
`test/101/101_1.html` resolves to `WebVR/test/101/101_1.html`. Pass the checkout root to
the scorer with `--webvr-root /path/to/WebVR`. The synthetic test fixtures in
`tests/fixtures/benchmark.jsonl` embed small HTML documents in `source_page.html`
and therefore do not require this argument.

## Prediction Examples

`predict_example.py` exposes the prompts used for Track A and Track B and demonstrates
three input routes through the official Gemini API:

| Input mode | Model input |
| --- | --- |
| `omni` | Original video with its audio track |
| `vl-video` | Silent video plus the manual transcript |
| `vl-frames` | Timestamped frames at 1 FPS by default, plus the manual transcript |

For local prediction, download the videos and use a benchmark copy whose `video`
paths are relative to that JSONL file.

All task prompts are in English. The transcript itself remains in the language spoken
in the video.

The VL routes require `ffmpeg`. The default run processes one sample; pass `--limit 0`
to process the complete benchmark. Use `--dry-run` to write the assembled prompt without
calling the API.

### Generate Predictions

```bash
# Track A with native audiovisual input
python predict_example.py \
  --benchmark /path/to/Omni2Web/benchmark.jsonl \
  --webvr-root /path/to/WebVR \
  --track a --input-mode omni \
  --output track_a_predictions.jsonl

# Track A with a VL model that accepts silent video
python predict_example.py \
  --benchmark /path/to/Omni2Web/benchmark.jsonl \
  --webvr-root /path/to/WebVR \
  --track a --input-mode vl-video \
  --model YOUR_GEMINI_VIDEO_MODEL \
  --output track_a_vl_predictions.jsonl

# Track B with a VL model that accepts ordered image frames
python predict_example.py \
  --benchmark /path/to/Omni2Web/benchmark.jsonl \
  --track b --input-mode vl-frames \
  --model YOUR_GEMINI_VISION_MODEL \
  --output track_b_predictions.jsonl
```

The prompt text is kept separately in `prediction_prompts.py`, so it can be reused with
another model provider without adopting the Gemini request code.

### Prediction Format

Track A predictions may be JSONL, JSON, a directory of HTML/text files, or any object using one of the common response fields accepted by `scorer/io_utils.py`.

Track B predictions contain a JSON array in each response:

```json
[{"step": 1, "target": "the visible Welcome heading", "instruction": "Make the heading blue"}]
```

The prediction files in `examples/` demonstrate both formats using the synthetic
samples in `tests/fixtures/benchmark.jsonl`. For the public benchmark, use predictions
with matching `sample_id` values.

## Evaluation

| Track | Task | Main metric |
| --- | --- | --- |
| **A — Direct Editing** | Produce a complete edited HTML page from the audiovisual request and source HTML | Edit Fidelity Score (**EFS**) |
| **B — Instruction Recovery** | Recover explicit target/action instructions from the audiovisual request | Instruction Recovery Score (**IRS**) |
| **C — Instruction Utility** | Execute recovered and oracle instructions with a fixed coding executor | Normalized Instruction Utility (**NIU**) |

All reported means are instance macro averages. Missing or unparsable contestant outputs receive zero. Judge-side failures are reported and excluded rather than treated as contestant failures.

### Track A: Direct Editing

The submitted final HTML is scored with Edit Fidelity Score (EFS): `0.10 × target grounding + 0.50 × edit fulfillment + 0.20 × No-damage + 0.20 × revision`. Requirements cancelled by a later whole-step reversion are listed in `rubric.excluded_steps` and omitted from final-state scoring.

```bash
python run_eval.py \
  --benchmark /path/to/Omni2Web/benchmark.jsonl \
  --webvr-root /path/to/WebVR \
  --predictions /path/to/track_a_predictions.jsonl \
  --model-name example-model \
  --output-dir track_a_results
```

The pipeline runs rubric judging, image-replacement verification, No-damage scoring, and EFS aggregation. Add `--resume` after an interrupted run.

### Track B: Instruction Recovery

A Gemini judge performs one-to-one semantic matching between reference and predicted steps. `F1_target = 2 TP_target / (n_ref + n_pred)`, `F1_action = 2 TP_action / (n_ref + n_pred)`, and `IRS = 0.5 F1_target + 0.5 F1_action`.

```bash
python run_eval_rewrite.py \
  --benchmark /path/to/Omni2Web/benchmark.jsonl \
  --webvr-root /path/to/WebVR \
  --predictions /path/to/track_b_predictions.jsonl \
  --model-name example-model \
  --output-dir track_b_results
```

The scorer validates that each reference and predicted step is matched at most once. It retries invalid judge structures and rejects duplicate matches, missing reference judgments, and out-of-range indices. Every predicted step contributes to `n_pred`, including unmatched extras.

### Track C: Instruction Utility

The fixed `deepseek-v4-pro` executor produces HTML from the recovered instructions and from oracle instructions under identical settings. `NIU = mean(EFS_recovered) / mean(EFS_oracle)` over paired samples.

```bash
python run_eval_e2e.py \
  --benchmark /path/to/Omni2Web/benchmark.jsonl \
  --webvr-root /path/to/WebVR \
  --rewrite-predictions /path/to/track_b_predictions.jsonl \
  --model-name example-model \
  --output-dir track_c_results
```

Recovered and oracle execution use the same endpoint, model, prompt template, token limit, and sample set. `--stream` selects the official streaming response format. Use `--resume` to reuse successful executor and judge results.

## Outputs

Each track writes the following files to its output directory:

| File | Contents |
| --- | --- |
| `summary.json` | Aggregate metrics and coverage counts |
| `results.json` | Per-sample scores and status |
| `run_manifest.json` | Input hashes and scoring configuration, with no absolute paths |

Track A/C also retain intermediate judge files inside their output directories for audit and safe resumption.

## Repository Structure

```text
Omni2Web-evaluation/
├── predict_example.py       # Minimal Track A/B prediction example
├── prediction_prompts.py    # Reusable English task prompts
├── run_eval.py              # Track A: Direct Editing
├── run_eval_rewrite.py      # Track B: Instruction Recovery
├── run_eval_e2e.py          # Track C: Instruction Utility
├── executor.py              # Fixed Track C coding executor
├── scorer/                  # Judges, metrics, and input utilities
├── examples/                # Synthetic annotations and example predictions
├── tests/                   # API-free tests and synthetic fixtures
├── .env.example             # API-key and endpoint configuration template
├── requirements.txt         # Runtime dependencies
├── requirements-dev.txt     # Test dependencies
├── LICENSE                  # CC BY-NC 4.0
└── NOTICE.md                # Attribution and upstream-content notices
```

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

The tests use synthetic inputs and do not call external APIs.

## Citation

If you find Omni2Web useful in your research, please cite:

```bibtex
@misc{han2026omni2web,
  title={Omni2Web: Benchmarking Audiovisual Website Development},
  author={Han, Minghao and Xing, Zhenghao and Cheng, Xize and Wang, Yuxuan and Lin, Junming and Wang, Ling and Yan, Yinsong and Chu, Yunfei and Yang, Qize and Xu, Jin},
  year={2026}
}
```

## License and Acknowledgements

The original contributions in this release, including the accompanying evaluation code, are provided under **CC BY-NC 4.0**. See [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md).

We use source webpages from [WebVR](https://huggingface.co/datasets/BroAlanTaps/WebVR). Upstream WebVR content and third-party materials retain their own terms.

## Dataset Disclaimer

All webpage content, interface elements, and related webpage materials depicted in the Omni2Web dataset are fictional, AI-generated materials produced using Qwen models based on publicly available open-source datasets. These materials are intended solely for benchmark evaluation and academic research and do not represent the authors' views. They should not be interpreted as descriptions of, or allusions to, any real organization, technology, law or regulation, event, or individual. Downstream users are responsible for complying with the licensing terms of upstream datasets and model service agreements, as well as applicable laws and regulations.
