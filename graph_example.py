# LangGraph equivalent of your agent example with enhancements
import dotenv
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

dotenv.load_dotenv()

# --- Tool Definitions (using LangGraph-compatible @tool decorator) ---

@tool
def get_stock_price(ticker: str) -> str:
    """
    Get the current price of a given stock ticker symbol.
    Args:
    ticker (str): The stock ticker symbol.
    Returns:
    str: The current price of the stock.
    """
    prices = {"AAPL": 180.15, "TSLA": 210.23}
    return f"The current price of {ticker} is ${prices.get(ticker.upper(), 100.0)}"

@tool
def get_company_news(ticker: str) -> str:
    """
    Get the latest news headline for a given stock ticker symbol.
    Args:
    ticker (str): The stock ticker symbol.
    Returns:
    str: The latest news headline.
    """
    articles = {
        "AAPL": "Apple announces new AI chip for iPhones. Investors react positively.",
        "TSLA": "Tesla expands into India. Market responds with optimism.",
    }
    return articles.get(ticker.upper(), "No recent news found.")

@tool
def summarize(text: str) -> str:
    """
    Summarize a given text string.
    Args:
    text (str): The text to be summarized.
    Returns:
    str: A summary of the text.
    """
    return f"Summary: {text.split('.')[0]}."

# --- State Schema ---
class AgentState(TypedDict):
    messages: list

# --- LLM Definition ---
tools = [get_stock_price, get_company_news, summarize]
llm = ChatOpenAI(model="gpt-4", temperature=0, tools=tools)

# --- Define logic nodes ---
def call_llm(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}

def call_tool(state: AgentState) -> AgentState:
    tool_call = state["messages"][-1].tool_calls[0]
    tool_name = tool_call['name']
    tool_input = tool_call['args']
    result = TOOL_REGISTRY[tool_name].invoke(tool_input)
    tool_msg = ToolMessage(tool_call_id=tool_call['id'], content=result)
    return {"messages": state["messages"] + [tool_msg]}

# --- Tool registry ---
TOOL_REGISTRY = {
    "get_stock_price": get_stock_price,
    "get_company_news": get_company_news,
    "summarize": summarize,
}

# --- Graph Definition ---
graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("tool", call_tool)

# Branching logic: if next step is tool call, go to tool; else END
def router(state: AgentState):
    last = state["messages"][-1]
    if hasattr(last, 'tool_calls') and last.tool_calls:
        return "tool"
    else:
        return END

graph.set_entry_point("llm")
graph.add_conditional_edges("llm", router)
graph.add_edge("tool", "llm")

# --- Compile the graph ---
app = graph.compile()

# --- Run the graph ---
response = app.invoke({
    "messages": [
        HumanMessage(content="What is the current price of AAPL and summarize the latest news about it?")
    ]
})

print(response["messages"][-1].content)
