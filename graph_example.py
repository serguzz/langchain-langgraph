# LangGraph equivalent of your agent example with enhancements
import dotenv
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

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
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Create the agent node using LangGraph's built-in utility
agent_node = create_react_agent(llm, tools)

# Build the graph
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.set_entry_point("agent")
graph.add_edge("agent", END)

# --- Compile the graph ---
app = graph.compile()

# --- Run the graph ---
response = app.invoke({
    "messages": [
        HumanMessage(content="What is the current price of AAPL and summarize the latest news about it?")
    ]
})

print(response["messages"][-1].content)
