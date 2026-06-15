import os
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent


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
PARENT_THREAD_ID = os.getenv("THREAD_ID", "parent-thread")
SUBAGENT_THREAD_ID = "subagent-thread"


# --- LLM --------------------------------------------------------------------

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    model=MODEL,
)


# --- Subagent ---------------------------------------------------------------

SUBAGENT_SYSTEM_PROMPT = (
    f"You are a research specialist. Today's date is {datetime.now().strftime('%Y-%m-%d')}. "
    "Your job is to answer questions using the search tool whenever needed. "
    "Always prefer fresh search results over training memory for facts, events, or prices."
)

subagent_checkpointer = InMemorySaver()

subagent_graph = create_agent(
    model=llm,
    tools=[TavilySearch(max_results=5)],
    system_prompt=SUBAGENT_SYSTEM_PROMPT,
    checkpointer=subagent_checkpointer,
)

subagent_config = {"configurable": {"thread_id": SUBAGENT_THREAD_ID}}


@tool
def research_agent(query: str) -> str:
    """Delegate a research question to a specialized agent that can search the web."""
    response = subagent_graph.invoke(
        {"messages": [{"role": "user", "content": query}]},
        subagent_config,
    )
    return response["messages"][-1].content


# --- Parent agent -----------------------------------------------------------

PARENT_SYSTEM_PROMPT = (
    f"You are a helpful assistant. Today's date is {datetime.now().strftime('%Y-%m-%d')}. "
    "When you need to look up facts, recent events, news, prices, or any external information, "
    "delegate the task to the research_agent tool. "
    "Synthesize the result into a clear, concise answer for the user."
)

parent_checkpointer = InMemorySaver()

parent_graph = create_agent(
    model=llm,
    tools=[research_agent],
    system_prompt=PARENT_SYSTEM_PROMPT,
    checkpointer=parent_checkpointer,
)

parent_config = {"configurable": {"thread_id": PARENT_THREAD_ID}}


# --- Runtime ----------------------------------------------------------------

def stream_graph_updates(user_input: str) -> None:
    for chunk in parent_graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        parent_config,
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
    print(f"[chatbot] Parent thread: {PARENT_THREAD_ID}")
    print(f"[chatbot] Subagent thread: {SUBAGENT_THREAD_ID}")
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
