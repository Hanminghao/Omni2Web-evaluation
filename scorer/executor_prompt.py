"""Frozen text prompt used by the Track C executor."""

MAX_HTML_CHARS = 180000
UNIVERSAL_PROMPT = "You are a web page editing model. You are given the original HTML of a web page and one web page editing request from the user. Understand what the user actually wants to change, and edit the page accordingly.\n\nRequirements:\n1. Output the complete HTML.\n2. Make only the change the user requested; do not make extra changes, and do not explain.\n3. Keep the rest of the page's content and structure as unchanged as possible.\n4. If the request contains multiple steps, an undo, or a change of mind, follow the final intent.\n5. Return one ```html code block whose content is the complete page.\n{request_block}\nOriginal HTML:\n```html\n{html}\n```"
TEXT_REQUEST_BLOCK = "\nThe user's web page editing request:\n{instr}\n"


def build_executor_prompt(source_html: str, rendered_steps: str) -> str:
    request = TEXT_REQUEST_BLOCK.replace("{instr}", rendered_steps)
    return (UNIVERSAL_PROMPT
            .replace("{request_block}", request)
            .replace("{html}", (source_html or "")[:MAX_HTML_CHARS]))
