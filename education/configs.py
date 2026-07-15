from concordia.language_model import language_model
from google.genai import types


_DEFAULT_HISTORY = (
    types.Content(
        role='user',
        parts=[
            types.Part(
                text=(
                    'You always continue inputs provided by the user, and'
                    ' you never repeat what the user already said.'
                ),
            ),
        ],
    ),
    types.Content(
        role='model',
        parts=[
            types.Part(
                text=(
                    'I always continue user-provided text and never repeat'
                    ' what the user already said.'
                ),
            ),
        ],
    ),
    types.Content(
        role='user',
        parts=[
            types.Part(
                text='Question: Is Jake a turtle?\nAnswer: Jake is ',
            ),
        ],
    ),
    types.Content(
        role='model',
        parts=[
            types.Part(text='not a turtle.'),
        ],
    ),
    types.Content(
        role='user',
        parts=[
            types.Part(
                text=(
                    'Question: What is Priya doing right now?\n'
                    'Answer: Priya is currently '
                ),
            ),
        ],
    ),
    types.Content(
        role='model',
        parts=[
            types.Part(text='sleeping.'),
        ],
    ),
)


_DEFAULT_SAFETY_SETTINGS = (
    types.SafetySetting(
        category='HARM_CATEGORY_HARASSMENT',
        threshold='BLOCK_MEDIUM_AND_ABOVE',
    ),
    types.SafetySetting(
        category='HARM_CATEGORY_HATE_SPEECH',
        threshold='BLOCK_MEDIUM_AND_ABOVE',
    ),
    types.SafetySetting(
        category='HARM_CATEGORY_SEXUALLY_EXPLICIT',
        threshold='BLOCK_MEDIUM_AND_ABOVE',
    ),
    types.SafetySetting(
        category='HARM_CATEGORY_DANGEROUS_CONTENT',
        threshold='BLOCK_MEDIUM_AND_ABOVE',
    ),
)


class Configs:
  """Configs for the education language model wrapper."""

  model_name = 'gemini-3.5-flash'
  api_key: str | None = None
  project: str | None = None
  location: str | None = None
  safety_settings = _DEFAULT_SAFETY_SETTINGS
  history = _DEFAULT_HISTORY
  channel = language_model.DEFAULT_STATS_CHANNEL
  sleep_periodically = False
  calls_between_sleeping = 10
  max_multiple_choice_attempts = 20
