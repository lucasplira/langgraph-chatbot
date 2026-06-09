import os
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.factory import create_agent


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


# --- System prompt ----------------------------------------------------------

SYSTEM_PROMPT = (
    f"You are a helpful assistant. Today's date is {datetime.now().strftime('%Y-%m-%d')}. "
    "For questions about current events, weather, news, prices, or recent facts, "
    "you MUST use the search tool. "
    "If a search result conflicts with what you remember from training, "
    "trust the search result - it has fresher information than you. "
    "If the search doesn't return a useful answer, say you don't know "
    "rather than inventing facts, dates, or URLs."
)


# --- LLM and tools ----------------------------------------------------------

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    model=MODEL,
)

tools = [TavilySearch(max_results=5)]


# --- Graph ------------------------------------------------------------------

graph = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": THREAD_ID}}


# --- Runtime ----------------------------------------------------------------

def stream_graph_updates(user_input: str) -> None:
    for chunk in graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
        stream_mode="updates",
    ):
        for node_name, node_output in chunk.items():
            for msg in node_output.get("messages", []):
                if getattr(msg, "tool_calls", None):
                    print(f"\n[debug] node={node_name}")
                    for tc in msg.tool_calls:
                        print(f"[debug] tool_call: {tc['name']} | args: {tc['args']}")
                elif getattr(msg, "name", None):
                    preview = msg.content[:10000].replace("\n", " ")
                    print(f"[debug] tool_result ({msg.name}): {preview}")
                elif msg.content:
                    print(f"\nAssistant: {msg.content}")


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