from typing import TypedDict,Annotated,List,Sequence
from langchain_core.messages import BaseMessage,ToolMessage,SystemMessage,HumanMessage
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END 
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import add_messages

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def add_messages_tool(a: int, b: int) -> int:

    ''' This is additon tool to add 2 given numbers'''
    return a + b

@tool
def multiply_messages_tool(a: int, b: int) -> int:

    ''' This is multiply tool to multiply 2 given numbers'''
    return a * b

tools= [add_messages_tool, multiply_messages_tool]

llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0.5).bind_tools(tools)



def llm_call(state: AgentState) -> AgentState:
    # Process the messages using the LLM
    system_prompt =SystemMessage(content="You are a helpful assistant. You can use your intelligence to answer my questions.")
    response = llm.invoke([system_prompt] + state["messages"])
    #print(f"LLM Response: {response.content}")
    return { "messages": [response] }

def should_continue(state: AgentState): 
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls: 
        return "end"
    else:
        return "continue"

graph = StateGraph(AgentState)
graph.add_node("my_agent", llm_call)

tool_node = ToolNode(tools) # ToolNode is a prebuilt node that can be used to call tools. It takes a list of tools as input.
graph.add_node("tool_node", tool_node) 
graph.set_entry_point("my_agent")

graph.add_conditional_edges(
    "my_agent",
    should_continue,
    {
        "continue": "tool_node",
        "end": END
    }
)

graph.add_edge("tool_node", "my_agent")

app =graph.compile()


def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [("user", "Add 23 + 123 and then multiply the result by 5. Also tell me a new joke please. Fianlly subtract 15 from the result last calculated")]}
print_stream(app.stream(inputs, stream_mode="values"))

