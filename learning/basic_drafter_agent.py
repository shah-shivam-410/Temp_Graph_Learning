from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

load_dotenv()

# DOCUMENT_CONTENT = ""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# @tool
# def updated_document_content(new_content: str) -> str:
#     """Update the document with provided content."""
#     global DOCUMENT_CONTENT
#     DOCUMENT_CONTENT = new_content
#     return f"Document updated successfully. The current content is {len(DOCUMENT_CONTENT)} characters long."

# @tool
# def save_document(filename: str) -> str:
#     """Save the document to a file and finish process."""
#     if not filename.endswith(".txt"):
#         filename += ".txt"
#     global DOCUMENT_CONTENT
#     try:
#         with open(filename, "w") as f:
#             f.write(DOCUMENT_CONTENT)
#         return f"Document saved successfully to {filename}."
#     except Exception as e:
#         return f"Error occurred while saving document: {e}"
    
# tools = [updated_document_content, save_document]

@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

tools = [add, multiply]

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash").bind_tools(tools)

def llm_call(state: AgentState) -> AgentState:
    """Call the LLM with the current state and return the updated state."""
    system_prompt = SystemMessage(
        content="You are a helpful assistant that can perform basic arithmetic operations. You can use the provided tools to add or multiply numbers. Please respond with the result of the operation requested by the user."
    )
    resp = model.invoke([system_prompt] + state["messages"])
    return {"messages": [resp]}

def should_continue(state: AgentState) -> str:
    last_msg = state["messages"][-1]
    if not last_msg.tool_calls:
    # if getattr(last_msg, "tool_calls", None):
            return "end"
    else:
         return "continue_to_tools"
    
graph = StateGraph(AgentState)
graph.add_node("llm", llm_call)
graph.add_node("tools", ToolNode(tools=tools))

graph.add_edge(START, "llm")
graph.add_conditional_edges(
     "llm",
     should_continue,
     {
          "end": END,
          "continue_to_tools": "tools"
     }
)
graph.add_edge("tools", "llm")

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()



inputs = {"messages": [
    HumanMessage("Add 100to 100 and then multiply the result by 10. At the end tell me a joke.")
]}

# result = app.invoke(inputs)

# print(result)


print_stream(app.stream(inputs, stream_mode="values"))
