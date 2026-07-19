"""
main.py
-------
A small Rich-powered REPL that drives the Day Assistant graph.
"""

from rich.console import Console
from rich.panel import Panel

from graph import day_assistant
from state import DayAssistantState

console = Console()

WELCOME = (
    "[bold cyan]Day Assistant[/bold cyan] -- tell me about your day, or ask me "
    "to list, add, complete, or delete tasks.\nType [bold]exit[/bold] or "
    "[bold]quit[/bold] to leave."
)


def make_initial_state(user_input: str) -> DayAssistantState:
    """A fresh per-turn state. TODO_STORE itself lives in state.py and
    persists across turns independently of this dict."""
    return {
        "user_input": user_input,
        "intent": "",
        "memory_scope": "",
        "todos": {},
        "selected_memory": {},
        "extracted_tasks": [],
        "task_title": "",
        "task_id": None,
        "action_result": "",
        "response": "",
    }


def main() -> None:
    console.print(Panel(WELCOME, title="Welcome", border_style="cyan"))

    while True:
        try:
            user_input = console.input("[bold green]You:[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            console.print("[dim]Goodbye![/dim]")
            break

        state = make_initial_state(user_input)
        result = day_assistant.invoke(state)

        console.print(
            Panel(
                result.get("response", "(no response generated)"),
                title=f"Assistant  ·  intent={result.get('intent', 'unknown')}",
                border_style="magenta",
            )
        )


if __name__ == "__main__":
    main()
