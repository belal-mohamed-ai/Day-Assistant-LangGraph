"""
schemas.py
----------
Pydantic models used for structured, validated LLM outputs.

These are handed to Instructor as `response_model=...` so a node gets back
a type-safe Python object instead of a raw string it would otherwise have
to `json.loads` and hope parses correctly.
"""

from enum import Enum
from typing import List

from pydantic import BaseModel, Field


class IntentType(str, Enum):
    """Every intent the Day Assistant knows how to route."""

    DESCRIBE_DAY = "describe_day"
    ADD_TODO = "add_todo"
    LIST_TODOS = "list_todos"
    COMPLETE_TODO = "complete_todo"
    DELETE_TODO = "delete_todo"
    UNKNOWN = "unknown"


class IntentClassification(BaseModel):
    """The ONLY thing the classifier node is allowed to produce."""

    intent: IntentType = Field(
        description=(
            "The single best-matching intent for the user's message. "
            "Use 'unknown' if none of the other intents clearly apply."
        )
    )


class ExtractedTasks(BaseModel):
    """Structured output for the Task Extraction node (describe_day only)."""

    tasks: List[str] = Field(
        default_factory=list,
        description=(
            "Short, actionable task titles pulled out of a free-form "
            "description of the user's day. Each item should be a clean "
            "imperative phrase, e.g. 'Buy groceries' -- not a full sentence."
        ),
    )


class TodoItem(BaseModel):
    """Shape of a single record inside TODO_STORE.

    Used for typing/validation only -- per the project spec, the store
    itself stays a plain Python dict, not a database or ORM model.
    """

    title: str
    done: bool = False
