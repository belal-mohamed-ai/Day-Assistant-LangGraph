"""
config.py
---------
Central place for model/client configuration so the rest of the codebase
never hard-codes connection details.

Ollama exposes an OpenAI-compatible endpoint, so we point a normal OpenAI
client at it. Two clients are exposed:

- `client`      -- wrapped with Instructor. Used wherever we need a
                   validated, structured (Pydantic) response: intent
                   classification and task extraction.
- `raw_client`  -- the plain client. Used by the response-generation node,
                   which only needs a short freeform sentence and gains
                   nothing from schema validation.
"""

import instructor
from openai import OpenAI

# --- Connection ---------------------------------------------------------
OLLAMA_BASE_URL = "http://localhost:11434/v1"
CHAT_MODEL = "gemma4:31b-cloud"

# --- Instructor behaviour ------------------------------------------------
# max_retries lets Instructor re-prompt the model automatically if its
# first reply doesn't validate against the target Pydantic schema.
DEFAULT_MAX_RETRIES = 2

_raw_client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

# Structured-output client (classify, extract).
# mode=JSON is the most broadly compatible mode across local Ollama models,
# since not all of them support OpenAI-style tool/function calling.
client = instructor.from_openai(_raw_client, mode=instructor.Mode.JSON)

# Plain chat client (respond).
raw_client = _raw_client
