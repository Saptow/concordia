"""Manual smoke test for the education Gemini language model wrapper.

Run with:
  uv run concordia/education/tests/simple_gemini_inference.py

This script intentionally does not use pytest-style `test_*` functions because
it performs a real API call.
"""

from pathlib import Path
import sys

from pydantic import BaseModel
_CONCORDIA_ROOT = Path(__file__).resolve().parents[2]
if str(_CONCORDIA_ROOT) not in sys.path:
  sys.path.insert(0, str(_CONCORDIA_ROOT))

from education.utils.language_model import GeminiLanguageModel


class LessonSummary(BaseModel):
  topic: str
  difficulty: str
  one_sentence_summary: str


def main() -> None:
  model = GeminiLanguageModel()

  prompt = 'Explain photosynthesis in two short sentences for a 14-year-old.'
  text_response = model.sample_text(prompt, max_tokens=120, temperature=0.3)
  print('TEXT RESPONSE:')
  print(text_response)

  structured_prompt = (
      'Summarize a lesson about algebra for a secondary school student.'
  )
  structured_response = model.sample_structured(
      structured_prompt,
      response_schema=LessonSummary,
      temperature=0.3
  )
  print('STRUCTURED RESPONSE:')
  print(structured_response.model_dump_json(indent=2))


if __name__ == '__main__':
  main()
