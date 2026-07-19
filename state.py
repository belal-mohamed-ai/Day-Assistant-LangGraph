"""
state.py
--------
The "suitcase" (DayAssistantState) that every LangGraph node reads and
writes, plus the in-memory todo store and the deterministic CRUD helpers
that operate on it.

Business logic lives HERE, not in nodes.py or graph.py. The LLM never
touches TODO_STORE directly -- only the helpers below do, and only
action_execution_node (in nodes.py) is allowed to call them.
"""

from typing import Any, Dict, List, Optional, TypedDict


class DayAssistantState(TypedDict):
    user_input: str
    intent: str  # describe_day | add_todo | list_todos | complete_todo | delete_todo | unknown
    memory_scope: str
    todos: Dict[int, Dict[str, Any]]
    selected_memory: Dict[int, Dict[str, Any]]
    extracted_tasks: List[str]
    task_title: str
    task_id: Optional[int]
    action_result: str
    response: str


# ---------------------------------------------------------------------------
# In-memory store. Module-level so it persists across REPL turns within a
# single process run. No SQLite / vector DB -- this project is pure
# orchestration, not RAG.
# ---------------------------------------------------------------------------
TODO_STORE: Dict[int, Dict[str, Any]] = {
    1: {"title": "Finish AI assignment", "done": False},
    2: {"title": "Call Ahmed", "done": False},
    3: {"title": "Buy milk", "done": True},
}


def _next_id() -> int:
    return max(TODO_STORE.keys(), default=0) + 1


def add_task(title: str) -> int:
    """Add a new todo and return its assigned id."""
    new_id = _next_id()
    TODO_STORE[new_id] = {"title": title.strip(), "done": False}
    return new_id


def mark_done(task_id: int) -> bool:
    """Mark a todo as completed. Returns False if the id doesn't exist."""
    if task_id in TODO_STORE:
        TODO_STORE[task_id]["done"] = True
        return True
    return False


def delete_task(task_id: int) -> bool:
    """Remove a todo. Returns False if the id doesn't exist."""
    return TODO_STORE.pop(task_id, None) is not None


def find_id_by_title_fragment(fragment: str) -> Optional[int]:
    """Best-effort fallback for when the user didn't give a task number,
    e.g. 'mark call ahmed as done' instead of 'complete task 2'."""
    fragment = fragment.lower().strip()
    if not fragment:
        return None
    for task_id, task in TODO_STORE.items():
        title = task["title"].lower()
        if title in fragment or fragment in title:
            return task_id
    return None


def snapshot() -> Dict[int, Dict[str, Any]]:
    """A shallow copy of the store, safe to hand into graph state without
    letting a node mutate the module-level dict by accident."""
    return {k: dict(v) for k, v in TODO_STORE.items()}


def format_todo_list() -> str:
    """Deterministic, exact rendering of the current todos.

    This exact string is what gets shown to the user for list_todos -- the
    LLM only ever adds a short friendly wrapper around it, it never
    regenerates the list itself, so counts and titles can't drift or get
    hallucinated.
    """
    if not TODO_STORE:
        return "Your to-do list is empty."
    lines = []
    for task_id in sorted(TODO_STORE.keys()):
        task = TODO_STORE[task_id]
        mark = "x" if task["done"] else " "
        lines.append(f"{task_id}. [{mark}] {task['title']}")
    return "\n".join(lines)
