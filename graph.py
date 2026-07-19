"""
graph.py
--------
Pure orchestration: wires nodes together into a LangGraph StateGraph and
defines the conditional routing table. No business logic lives here --
that belongs in nodes.py / state.py.
"""

from langgraph.graph import END, StateGraph

from nodes import (
    action_execution_node,
    intent_classifier_node,
    memory_selection_node,
    response_generation_node,
    task_extraction_node,
)
from schemas import IntentType
from state import DayAssistantState


def route_by_intent(state: DayAssistantState) -> str:
    """Reads the classified intent and decides which node runs next.

    describe_day               -> extract  (LLM pulls out actionable tasks)
    list/complete/delete_todo  -> select   (Python loads relevant memory)
    add_todo                   -> act      (no prior memory needed)
    unknown / anything else    -> respond  (skip straight to a friendly reply)
    """
    intent = state["intent"]

    if intent == IntentType.DESCRIBE_DAY.value:
        return "extract"
    if intent in (
        IntentType.LIST_TODOS.value,
        IntentType.COMPLETE_TODO.value,
        IntentType.DELETE_TODO.value,
    ):
        return "select"
    if intent == IntentType.ADD_TODO.value:
        return "act"
    return "respond"


def build_graph():
    builder = StateGraph(DayAssistantState)

    builder.add_node("classify", intent_classifier_node)
    builder.add_node("extract", task_extraction_node)
    builder.add_node("select", memory_selection_node)
    builder.add_node("act", action_execution_node)
    builder.add_node("respond", response_generation_node)

    builder.set_entry_point("classify")

    builder.add_conditional_edges(
        "classify",
        route_by_intent,
        {
            "extract": "extract",
            "select": "select",
            "act": "act",
            "respond": "respond",
        },
    )

    builder.add_edge("extract", "act")
    builder.add_edge("select", "act")
    builder.add_edge("act", "respond")
    builder.add_edge("respond", END)

    return builder.compile()


# Compiled once at import time -- reused across every REPL turn.
day_assistant = build_graph()
