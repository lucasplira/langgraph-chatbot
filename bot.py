import json
import os
from datetime import datetime
from pathlib import Path
from typing import Annotated

from typing_extensions import TypedDict

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


# --- Config & validation ----------------------------------------------------

if not os.getenv("OPENROUTER_API_KEY"):
    raise RuntimeError(
        "OPENROUTER_API_KEY is required. Pass it with `docker run -e OPENROUTER_API_KEY=...`"
    )

if not os.getenv("TAVILY_API_KEY"):
    raise RuntimeError(
        "TAVILY_API_KEY is required. Pass it with `docker run -e TAVILY_API_KEY=...`"
    )

MODEL = os.getenv("MODEL", "openai/gpt-4o-mini")
THREAD_ID = os.getenv("THREAD_ID", "default")

LOG_DIR = Path("/app/logs")
LOG_DIR.mkdir(exist_ok=True)


# --- System prompt ----------------------------------------------------------

SYSTEM = SystemMessage(
    content=(
        f"You are a helpful assistant. Today's date is {datetime.now().strftime('%Y-%m-%d')}. "
        "For questions about current events, weather, news, prices, or recent facts, "
        "you MUST use the search tool. "
        "If a search result conflicts with what you remember from training, "
        "trust the search result - it has fresher information than you. "
        "If the search doesn't return a useful answer, say you don't know "
        "rather than inventing facts, dates, or URLs."
    )
)


# --- State ------------------------------------------------------------------

class State(TypedDict):
    messages: Annotated[list, add_messages]


# --- LLM and tools ----------------------------------------------------------

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    model=MODEL,
)

tools = [TavilySearch(max_results=2)]
llm_with_tools = llm.bind_tools(tools)


# --- Logging helper ---------------------------------------------------------

def log_messages(messages, label: str) -> None:
    """Dump a list of LangChain messages to disk as indented JSON.

    Filename format: YYYY-MM-DD_HH-MM-SS-mmm_<label>.json so files sort
    chronologically and don't collide within the same second.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
    path = LOG_DIR / f"{timestamp}_{label}.json"
    payload = [m.model_dump() for m in messages]
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"[log] {path}")


# --- Graph nodes ------------------------------------------------------------

def chatbot(state: State):
    messages = [SYSTEM] + state["messages"]
    log_messages(messages, "input")
    response = llm_with_tools.invoke(messages)
    log_messages([response], "output")
    return {"messages": [response]}


# --- Graph wiring -----------------------------------------------------------

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=tools))

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")

graph = graph_builder.compile(checkpointer=InMemorySaver())

config = {"configurable": {"thread_id": THREAD_ID}}


# --- Runtime ----------------------------------------------------------------

def stream_graph_updates(user_input: str) -> None:
    for event in graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
        stream_mode="values",
    ):
        event["messages"][-1].pretty_print()


def main() -> None:
    print(f"[chatbot] Model: {MODEL}")
    print(f"[chatbot] Thread: {THREAD_ID}")
    print("[chatbot] Type 'quit', 'exit', or 'q' to leave.\n")

    while True:
        try:
            user_input = input("User: ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)


if __name__ == "__main__":
    main()
