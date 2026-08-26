from typing import TypedDict,Annotated,List,Sequence
from langchain_core.messages import BaseMessage,ToolMessage,SystemMessage,HumanMessage
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END 
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import add_messages

load_dotenv()

#Global varialble to sotore document content
document_content = ""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def update_document_tool(new_content: str) -> str:
    ''' This tool updates the document content with the new content provided.'''
    global document_content
    document_content = new_content
    return f"Document content updated successfully. Current document content: {document_content}"

@tool
def save_document_tool(filename:str) -> str:
    """ Save research document content to file. Give appropriate name to the research document file when the user wants to save the document content.
    Args:
        filename (str): The name of the file to save the document content to.
    """
    global document_content
    if(not filename.endswith(".txt")):
        filename += ".txt"

    try:
        with open(filename, "w") as f:
            f.write(document_content)
        
        print(f"Document content1 saved to {filename} successfully.")

        return f"Document content saved to {filename} successfully. file content: {document_content}"
    except Exception as e:
        return f"Failed to save document content to {filename}. Error: {e}"

tools = [update_document_tool, save_document_tool]

model = ChatAnthropic(model="claude-haiku-4-5", temperature=0.5).bind_tools(tools)

def my_agent(state: AgentState) -> AgentState:
    # Process the messages using the LLM
    system_prompt = SystemMessage(content=
                                  f"""You are a research assistant. You can help with research tasks and create a research document. 
                                  You can also use the following tools to update and save the research document:
                                  - If user wants to update the document content, use the 'update_document_tool'.
                                  - If user wants to save the document content to a file, use the 'save_document_tool'. 
                                  - Always provide the current document content after any modification.
                                  - Procide relevand file name when saving the document content to a file.
                                  """)
    print(f"\nInitial contenet of state messages: {state['messages']}")
    if not state["messages"]:
        user_input = "I am ready to help you with your research tasks. Please provide me with the subject or input to provide research to a file."
        user_message = HumanMessage(content=user_input)

    else:
        user_input = input("\nWhat would you like to do with document?")
        print(f"User input: {user_input}")
        user_message = HumanMessage(content=user_input)

    all_messages = [system_prompt] + list(state["messages"]) + [user_message]

    response = model.invoke(all_messages)
    print(f"\n AI response: {response.content}")

    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"\nUsing Tool calls: {[tc['name'] for tc in response.tool_calls]}")
        
    return {"messages": list(state["messages"]) + [user_message, response]}

def should_continue(state: AgentState):
    ''' Determine whether to continue or end the conversation based on the last message in the state.'''
    messages = state["messages"]
    print(f"\nDocument content in should_continue: {messages}")
    if not messages:
        return "continue"

    # look for recent tool calls in the last message
    for message in reversed(messages):
        # Check if tool message is for saving the document content
        if isinstance(message, ToolMessage) and "saved" in message.content.lower() and \
           "document" in message.content.lower():
            return "end" # Goes to end if the document is saved successfully

    return "continue" # Continue if no tool calls for saving the document content are found

def print_messages(messages):
    ''' Print the messages in more readable format.'''
    if not messages:
        print("No messages to display.")
        return
    for message in messages:
        if isinstance(message,ToolMessage):
            print(f"\nTool Message: {message.content}")

graph = StateGraph(AgentState)
graph.add_node("my_agent", my_agent)
graph.add_node("tool_node", ToolNode(tools))
graph.add_edge("my_agent", "tool_node")

graph.set_entry_point("my_agent")

graph.add_conditional_edges(
    "tool_node",
    should_continue,
    {
        "continue": "my_agent",
        "end": END
    }
)

app = graph.compile()

def run_research_assistant():
    ''' Run the research assistant application.'''
    state = AgentState(messages=[])

    for step in app.stream(state, stream_mode="values"):
        if 'messages' in step:
            print_messages(step['messages'])

    print("\nResearch assistant session ended. Thank you for using the research assistant!")

if __name__ == "__main__":
    run_research_assistant()