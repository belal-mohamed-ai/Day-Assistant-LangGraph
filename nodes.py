"""
nodes.py
--------
Every LangGraph node for the Day Assistant.

Three nodes are LLM-backed (classify, extract, respond); two are pure,
deterministic Python (select, act). The LLM never mutates TODO_STORE --
only action_execution_node does, via the helpers in state.py.
"""

import logging
import re
from typing import Any, Dict, Optional

from config import CHAT_MODEL, DEFAULT_MAX_RETRIES, client, raw_client
from schemas import ExtractedTasks, IntentClassification, IntentType
from state import (
    DayAssistantState,
    TODO_STORE,
    add_task,
    delete_task,
    find_id_by_title_fragment,
    format_todo_list,
    mark_done,
    snapshot,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Intent Classification (LLM, structured output)
# ---------------------------------------------------------------------------
CLASSIFIER_SYSTEM_PROMPT = """You are the routing brain of a day-planning assistant.
Read the user's message and classify it into exactly one intent:

- describe_day: the user is narrating things they need to do (often several at once)
- add_todo: the user wants ONE new task added to their list
- list_todos: the user wants to see their current tasks
- complete_todo: the user wants to mark a task as done
- delete_todo: the user wants a task removed
- unknown: none of the above clearly apply

Return only the intent. Do not explain your reasoning."""


def intent_classifier_node(state: DayAssistantState) -> Dict[str, Any]:
    user_input = state["user_input"]
    try:
        result: IntentClassification = client.chat.completions.create(
            model=CHAT_MODEL,
            response_model=IntentClassification,
            max_retries=DEFAULT_MAX_RETRIES,
            messages=[
                {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
                {"role": "user", "content": user_input},
            ],
        )
        intent = result.intent.value
    except Exception as exc:  # noqa: BLE001 - never crash the REPL
        logger.warning("Intent classification failed, defaulting to 'unknown': %s", exc)
        intent = IntentType.UNKNOWN.value

    return {"intent": intent}


# ---------------------------------------------------------------------------
# 2. Task Extraction (LLM, structured output) -- describe_day only
# ---------------------------------------------------------------------------
EXTRACTION_SYSTEM_PROMPT = """Extract every actionable task from the user's message.
Return each as a short imperative phrase (e.g. "Buy groceries"), not a full sentence.
If there are no actionable tasks, return an empty list."""


def task_extraction_node(state: DayAssistantState) -> Dict[str, Any]:
    user_input = state["user_input"]
    try:
        result: ExtractedTasks = client.chat.completions.create(
            model=CHAT_MODEL,
            response_model=ExtractedTasks,
            max_retries=DEFAULT_MAX_RETRIES,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_input},
            ],
        )
        tasks = [t.strip() for t in result.tasks if t.strip()]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Task extraction failed, falling back to naive split: %s", exc)
        tasks = _naive_task_split(user_input)

    return {"extracted_tasks": tasks}


_FILLER_PREFIXES = [
    r"^tomorrow\s+",
    r"^today\s+",
    r"^i\s+(also\s+)?(need|have|want)\s+to\s+",
    r"^i'?ll\s+",
]


def _naive_task_split(user_input: str) -> list:
    """Rough, dependency-free substitute for the LLM extractor -- only used
    if the LLM/Ollama is completely unreachable. Splits on commas / " and "
    and strips common filler lead-ins so results stay reasonably clean."""
    rough = re.split(r",| and ", user_input)
    tasks = []
    for chunk in rough:
        chunk = chunk.strip()
        for pattern in _FILLER_PREFIXES:
            chunk = re.sub(pattern, "", chunk, flags=re.IGNORECASE)
        chunk = chunk.strip().rstrip(".").strip()
        if len(chunk) > 2:
            tasks.append(chunk[0].upper() + chunk[1:])
    return tasks


# ---------------------------------------------------------------------------
# 3. Memory Selection (Python, deterministic) -- list/complete/delete only
# ---------------------------------------------------------------------------
def memory_selection_node(state: DayAssistantState) -> Dict[str, Any]:
    # Orchestration decides what context is relevant instead of dumping
    # everything into every prompt: here that means "the current todos".
    return {"selected_memory": snapshot(), "memory_scope": "todos"}


# ---------------------------------------------------------------------------
# 4. Action Execution (Python, deterministic CRUD) -- the ONLY node allowed
#    to mutate TODO_STORE.
# ---------------------------------------------------------------------------
_ADD_PREFIXES = [
    r"^please\s+",
    r"^add\s+(a\s+|another\s+)?(new\s+)?(task|todo|to-do|to do)?\s*[:\-]?\s*",
    r"^create\s+(a\s+)?(new\s+)?(task|todo)?\s*[:\-]?\s*",
    r"^new\s+(task|todo)?\s*[:\-]?\s*",
    r"^i\s+need\s+to\s+",
]
_ADD_SUFFIXES = [
    r"\s+to\s+my\s+(to-?do\s*)?list$",
    r"\s+to\s+the\s+list$",
]


def _extract_task_title(user_input: str) -> str:
    """Deterministic parsing of a task title out of an add_todo message."""
    title = user_input.strip()
    for pattern in _ADD_PREFIXES:
        title = re.sub(pattern, "", title, flags=re.IGNORECASE)
    for pattern in _ADD_SUFFIXES:
        title = re.sub(pattern, "", title, flags=re.IGNORECASE)
    return title.strip().rstrip(".").capitalize()


def _extract_task_id(user_input: str) -> Optional[int]:
    """Pull a task id out of a complete/delete message. Falls back to a
    title-fragment match if the user didn't mention a number."""
    match = re.search(r"\d+", user_input)
    if match:
        return int(match.group())
    return find_id_by_title_fragment(user_input)


def action_execution_node(state: DayAssistantState) -> Dict[str, Any]:
    intent = state["intent"]
    user_input = state["user_input"]
    updates: Dict[str, Any] = {}

    if intent == IntentType.DESCRIBE_DAY.value:
        tasks = state.get("extracted_tasks", [])
        for title in tasks:
            add_task(title)
        if tasks:
            action_result = f"Added {len(tasks)} new task(s) to the list: {', '.join(tasks)}."
        else:
            action_result = "No actionable tasks were found in that description."

    elif intent == IntentType.ADD_TODO.value:
        title = _extract_task_title(user_input)
        if title:
            new_id = add_task(title)
            updates["task_id"] = new_id
            updates["task_title"] = title
            action_result = f"Added the task '{title}' to the list."
        else:
            action_result = "Couldn't tell what task to add from that message."

    elif intent == IntentType.LIST_TODOS.value:
        action_result = "The user asked to see their current tasks."

    elif intent == IntentType.COMPLETE_TODO.value:
        task_id = _extract_task_id(user_input)
        if task_id is not None and mark_done(task_id):
            updates["task_id"] = task_id
            action_result = f"Marked task {task_id} ('{TODO_STORE[task_id]['title']}') as completed."
        else:
            action_result = "Couldn't find a matching task to mark as completed."

    elif intent == IntentType.DELETE_TODO.value:
        task_id = _extract_task_id(user_input)
        title = TODO_STORE.get(task_id, {}).get("title") if task_id is not None else None
        if task_id is not None and delete_task(task_id):
            updates["task_id"] = task_id
            action_result = f"Deleted task {task_id} ('{title}') from the list."
        else:
            action_result = "Couldn't find a matching task to delete."

    else:  # pragma: no cover - unknown never reaches act (routed to respond directly)
        action_result = ""

    updates["action_result"] = action_result
    updates["todos"] = snapshot()
    return updates


# ---------------------------------------------------------------------------
# 5. Response Generation (LLM, friendly tone -- facts come from Python)
# ---------------------------------------------------------------------------
RESPONSE_SYSTEM_PROMPT = """You are a warm, concise day-planning assistant.
You will be given a Fact describing something that has ALREADY happened.
Write ONE short, friendly confirmation sentence (max 2 sentences) based ONLY
on that fact. Never invent tasks, numbers, or details that weren't given to
you. If the fact mentions a list of tasks, do not repeat the list yourself --
it will be shown separately -- just write a short, natural intro or outro
line."""


def _fallback_response(intent: str, action_result: str) -> str:
    """Deterministic template used if the LLM call fails, or for the
    unknown intent -- the app must keep working even with Ollama offline."""
    if intent == IntentType.LIST_TODOS.value:
        return f"Here's your to-do list:\n{format_todo_list()}"
    if action_result:
        return action_result
    return (
        "I'm not sure how to help with that yet. Try describing your day, "
        "or ask me to add, list, complete, or delete a task."
    )


def response_generation_node(state: DayAssistantState) -> Dict[str, Any]:
    intent = state["intent"]
    action_result = state.get("action_result", "")

    if intent == IntentType.UNKNOWN.value or not action_result:
        return {"response": _fallback_response(intent, action_result)}

    try:
        completion = raw_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": RESPONSE_SYSTEM_PROMPT},
                {"role": "user", "content": f"Fact: {action_result}"},
            ],
        )
        wrapper = completion.choices[0].message.content.strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Response generation failed, using deterministic template: %s", exc)
        return {"response": _fallback_response(intent, action_result)}

    # Lists stay exact: the LLM only supplies the wrapper sentence. The
    # verbatim list text (already computed deterministically) is appended
    # after it, so counts/titles can never drift or get hallucinated.
    if intent == IntentType.LIST_TODOS.value:
        return {"response": f"{wrapper}\n\n{format_todo_list()}"}

    return {"response": wrapper}
