"""English prediction prompts used by the benchmark examples."""

MAX_HTML_CHARS = 180_000

TRACK_A_BASE = """You are a web page editing model. You are given the original HTML of a web page and one web page editing request from the user. Understand what the user actually wants to change, and edit the page accordingly.

Requirements:
1. Output the complete HTML.
2. Make only the change the user requested; do not make extra changes, and do not explain.
3. Keep the rest of the page's content and structure as unchanged as possible.
4. If the request contains multiple steps, an undo, or a change of mind, follow the final intent.
5. Return one ```html code block whose content is the complete page.
{request_block}
Original HTML:
```html
{html}
```"""

TRACK_A_OMNI_REQUEST = """
The user's editing request is in the attached video: the user points at parts of the page with a cursor or a finger while speaking the request out loud, using deictic words such as "this", "here", "these two". Watch the picture to see WHICH part of the page is being pointed at, and listen to the speech to hear WHAT to change — you need both, together with the HTML below, to resolve each reference to a concrete element.
"""

TRACK_A_VL_VIDEO_REQUEST = """
The user's editing request is split across two inputs. (1) The attached video is SILENT and only shows the user pointing at parts of the page with a cursor or a finger. (2) The text below is a transcription of what the user said at the same time, one line per step, in the order requested. The text tells you WHAT to change; the video tells you WHICH part of the page each step points at. Use both.
Transcription of the user's speech:
{instr}
"""

TRACK_A_VL_FRAMES_REQUEST = """
The user's editing request is split across two inputs. (1) The attached {n_images} image(s) are not separate screenshots: they are consecutive frames sampled once per second from ONE silent screen recording of this page, in which the user points at parts of the page with a cursor. The images are in chronological order — one image per second of the recording — and every image has its true timestamp burned into its top-left corner; read those timestamps off the image rather than computing them. (2) The text below is a transcription of what the user said while that recording was made, one line per step, in the order requested. The text tells you WHAT to change; the images tell you WHICH part of the page each step points at. Use both.
Transcription of the user's speech:
{instr}
"""

TRACK_B_OMNI = """You are given a video in which a user points at parts of a web page and speaks editing requests using deictic references ("this", "here", "these two"). You are NOT asked to edit the HTML. Rewrite the user's request into explicit, self-contained instructions.

Output ONLY a JSON array, one object per editing step, in the order the user requested them:
[{"step": 1, "target": "...", "instruction": "..."}, ...]

- "target": which part of the page the user is pointing at, described precisely enough that someone who never saw the video could find it on the page — quote the visible text, heading, label or section name you can read on screen, plus its position if needed. Never use "this" / "here" / "that".
- "instruction": what to change, as one explicit imperative sentence. No deictic words. If the user undoes a step or changes their mind, state the FINAL intent.
- Write "target" and "instruction" in THE SAME LANGUAGE THE USER SPEAKS in the video. Do not translate the user's words into English. The only exception is text you read off the page itself (headings, labels, button text) — quote that verbatim in whatever language it appears on screen.
- Emit exactly as many steps as the user actually requested. Do not pad or merge.
- Do not output HTML.
"""

TRACK_B_VL = """You are given a SILENT screen recording of a web page and a transcription of what the user said while it was recorded.

In the recording the user moves the mouse cursor to point at things on the page. In the transcription the user refers to those things only by weak deictic words ("this", "here", "these two") — which cannot be resolved from the text alone. Resolve every one of them by aligning each spoken step with where the cursor is at that moment in the recording.

You are NOT asked to edit the HTML. Rewrite the user's request into explicit, self-contained instructions.

Output ONLY a JSON array, one object per editing step, in the order the user requested them:
[{"step": 1, "target": "...", "instruction": "..."}, ...]

- "target": which part of the page the user is pointing at, described precisely enough that someone who never saw the video could find it on the page — quote the visible text, heading, label or section name you can read on screen, plus its position if needed. Never use "this" / "here" / "that".
- "instruction": what to change, as one explicit imperative sentence. No deictic words. If the user undoes a step or changes their mind, state the FINAL intent.
- Write "target" and "instruction" in THE SAME LANGUAGE AS THE TRANSCRIPTION below. Do not translate the user's words into English. The only exception is text you read off the page itself (headings, labels, button text) — quote that verbatim in whatever language it appears on screen.
- Emit exactly as many steps as the user actually requested. Do not pad or merge.
- Do not output HTML.
"""

TRACK_B_VIDEO_DELIVERY = """
The recording is attached as a silent video file.
"""

TRACK_B_FRAMES_DELIVERY = """
The recording is attached as {n_images} consecutive frames sampled once per second from that ONE recording — not as separate screenshots. They are in chronological order, one image per second, and every image has its true timestamp burned into its top-left corner; read those timestamps off the image rather than computing them.
"""

TRACK_B_TRANSCRIPT = """
Transcription of the user's speech (one line per step, in the order requested):
{instr}
"""

INPUT_MODES = ("omni", "vl-video", "vl-frames")


def build_track_a_prompt(*, html: str, transcript: str, input_mode: str,
                         n_frames: int = 0) -> str:
    if input_mode == "omni":
        request = TRACK_A_OMNI_REQUEST
    elif input_mode == "vl-video":
        request = TRACK_A_VL_VIDEO_REQUEST.replace("{instr}", transcript)
    elif input_mode == "vl-frames":
        if n_frames <= 0:
            raise ValueError("vl-frames requires n_frames")
        request = (TRACK_A_VL_FRAMES_REQUEST
                   .replace("{instr}", transcript)
                   .replace("{n_images}", str(n_frames)))
    else:
        raise ValueError(f"Unsupported input mode: {input_mode}")
    return (TRACK_A_BASE
            .replace("{request_block}", request)
            .replace("{html}", html[:MAX_HTML_CHARS]))


def build_track_b_prompt(*, transcript: str, input_mode: str,
                         n_frames: int = 0) -> str:
    if input_mode == "omni":
        return TRACK_B_OMNI
    if input_mode == "vl-video":
        delivery = TRACK_B_VIDEO_DELIVERY
    elif input_mode == "vl-frames":
        if n_frames <= 0:
            raise ValueError("vl-frames requires n_frames")
        delivery = TRACK_B_FRAMES_DELIVERY.replace("{n_images}", str(n_frames))
    else:
        raise ValueError(f"Unsupported input mode: {input_mode}")
    return TRACK_B_VL + delivery + TRACK_B_TRANSCRIPT.replace("{instr}", transcript.strip())
