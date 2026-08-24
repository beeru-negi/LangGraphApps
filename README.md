# LangGraph Learning and Practice

This repository is a personal learning workspace for practicing LangGraph, LangChain, and tool-calling agents with Anthropic models.

## What This Project Covers

- Building a simple LangGraph chat flow
- Defining graph state with `TypedDict`
- Running an LLM node inside a graph
- Creating and binding tools to an LLM
- Using `ToolNode` for tool execution
- Looping between an agent node and tools until the task is complete

## Project Files

- `Agent_Bot.py`: a basic LangGraph chatbot that takes terminal input and sends it to an Anthropic model
- `ReactTool.py`: a tool-calling LangGraph example that uses arithmetic tools inside an agent loop
- `langgraph.ipynb`: notebook for LangGraph practice and experiments
- `conditional_agent.ipynb`: notebook for trying conditional agent flows
- `requirements.txt`: Python dependencies for this workspace

## Requirements

- Python 3.12 or compatible
- A virtual environment
- An Anthropic API key

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root:

```env
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

## Run the Examples

Run the basic chatbot:

```bash
python Agent_Bot.py
```

Run the tool-calling example:

```bash
python ReactTool.py
```

## Notes on the Code

### `Agent_Bot.py`

This script shows the smallest useful LangGraph flow:

- define a graph state
- create a single LLM node
- connect `START` to the node and then to `END`
- invoke the graph with user input from the terminal

### `ReactTool.py`

This script demonstrates a more agentic pattern:

- define tools with `@tool`
- bind those tools to `ChatAnthropic`
- let the model decide when tools are needed
- route to `ToolNode` when tool calls are present
- return to the agent node until no further tool call is needed

## Learning Goals

This repository is useful for practicing:

- graph-based agent design
- message-driven state updates
- tool calling with LangChain and LangGraph
- debugging API and model configuration issues
- moving from simple chains to agent loops

## Security Reminder

- Keep `.env` out of version control
- Do not hardcode API keys in Python files
- Rotate any key that has been exposed in logs, screenshots, or shared chat

## Next Practice Ideas

- add a subtraction tool and division tool
- make the chatbot keep conversation history
- add error handling for missing API keys
- stream token output in the terminal
- experiment with conditional routing in notebooks