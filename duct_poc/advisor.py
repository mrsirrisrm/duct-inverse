"""LLM-powered design advice for a duct-system analysis.

Reads GEMINI_API_KEY from environment (loaded from .env). Same pattern as
the insul_poc advisor so users have one key and one UX across both tools.
"""
from __future__ import annotations

import json
import os

from duct_poc.analysis import DesignFacts


SYSTEM_PROMPT = """\
You are an experienced mechanical-services acoustic consultant advising
an engineer who is laying out a small HVAC supply path from a source (FCU
or similar) to a room diffuser. Given a JSON object describing the
current system — block-by-block design, receiver octave-band SPL, NC
rating, controlling bands, and top sensitivities — write SHORT, actionable
design advice.

Rules:
- Identify the weakest point in the current design using
  "controlling_bands" (the octave bands exceeding the NC target curve)
  and the block that is most responsible (infer from block types in
  "blocks" and the ranking in "top_sensitivities").
- Recommend the single most impactful change. Sensitivities are
  dNC / d(parameter) in linear units, so a large-magnitude negative value
  means increasing that parameter lowers NC.
- Use HVAC acoustics vocabulary (regenerative noise, duct lining,
  insertion loss, splitter attenuator, end reflection, room constant)
  but be concrete: name the block and name the parameter.
- Keep it to 3 to 5 sentences. Plain prose, no bullet points.
- Don't restate the NC number — it's shown separately in the UI.
- Do not invent products, codes, or numbers not present in the JSON.
"""


def _gemini_available() -> tuple[bool, str]:
    if not os.environ.get("GEMINI_API_KEY"):
        return False, "GEMINI_API_KEY not set in environment."
    try:
        from google import genai  # noqa: F401
    except ImportError:
        return False, "google-genai package not installed."
    return True, ""


def advice_for(
    facts: DesignFacts,
    model_name: str = "gemini-3.1-flash-lite-preview",
) -> str:
    ok, reason = _gemini_available()
    if not ok:
        raise RuntimeError(reason)

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    payload = json.dumps(facts.to_json(), indent=2)
    prompt = (
        "Here is the current duct-system analysis:\n\n"
        f"```json\n{payload}\n```\n\n"
        "Write the advice now."
    )
    resp = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.3,
        ),
    )
    return (resp.text or "").strip()
