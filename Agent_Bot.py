from langchain_core.messages import HumanMessage
from typing import TypedDict, List

from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages: List[HumanMessage]

llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0.5)

def process_messages(state: AgentState) -> AgentState:
    # Process the messages using the LLM
    response = llm.invoke(state["messages"])
    print(f"LLM Response: {response.content}")
    return state

graph = StateGraph(AgentState)
graph.add_node("process_messages", process_messages)
graph.add_edge(START, "process_messages")
graph.add_edge("process_messages", END)
agent=graph.compile()

user_messages = input("Enter messages: ")

while user_messages.lower() != "exit":
    state = AgentState(messages=[HumanMessage(content=user_messages)])
    agent.invoke({"messages":[HumanMessage(content=user_messages)]})
    user_messages = input("Enter messages: ")

