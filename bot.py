import os
import json
from datetime import datetime
from typing import Annotated
from pathlib import Path

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.messages import ToolMessage
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import InMemorySaver

LOG_DIR = Path("/app/logs")
LOG_DIR.mkdir(exist_ok=True)

SYSTEM = SystemMessage(content=(
    "You are a helpful assistant. For any question about current events, "
    "weather, news, or recent facts, you MUST use the search tool. "
    "If the search doesn't return the answer, say you don't know. "
    "Never invent facts or URLs."
))

if not os.getenv("TAVILY_API_KEY"):
    raise RuntimeError("TAVILY_API_KEY is required. Pass it with `docker run -e TAVILY_API_KEY=...`")

class State(TypedDict):
    messages: Annotated[list, add_messages]

class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}


graph_builder = StateGraph(State)


llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    model=os.getenv("MODEL", "openai/gpt-4o-mini"),
)

tool = TavilySearch(max_results=2)
tools = [tool]

memory = InMemorySaver()

llm_with_tools = llm.bind_tools(tools)

def route_tools(
    state: State,
):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END

def _dump_file(messages, label):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
    path = LOG_DIR / f"{timestamp}_{label}.json"
    payload = [m.model_dump() for m in messages]
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"[log] {path}")

def chatbot(state: State):
    _dump_file(state["messages"], "invoke")
    response = llm_with_tools.invoke([SYSTEM] + state["messages"])
    _dump_file([response], "output")
    return {"messages": [response]}

#tools node
tool_node = BasicToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("chatbot", chatbot)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    {"tools": "tools", END: END},
)
graph_builder.add_edge("tools", "chatbot")
graph = graph_builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}



def stream_graph_updates(user_input: str):
    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
    )
    for event in events:
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

while True:
    try:
        user_input = input("User: ")
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye!")
        break

    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    stream_graph_updates(user_input)