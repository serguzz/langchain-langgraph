from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_openai import ChatOpenAI

import dotenv
dotenv.load_dotenv()

# Fake tools
def get_stock_price(ticker: str) -> str:
    prices = {"AAPL": 180.15, "TSLA": 210.23}
    return f"The current price of {ticker} is ${prices.get(ticker.upper(), 100.0)}"

def get_company_news(ticker: str) -> str:
    articles = {
        "AAPL": "Apple announces new AI chip for iPhones. Investors react positively.",
        "TSLA": "Tesla expands into India. Market responds with optimism.",
    }
    return articles.get(ticker.upper(), "No recent news found.")

def summarize(text: str) -> str:
    return f"Summary: {text.split('.')[0]}."

# Define tools
tools = [
    Tool(
        name="Stock Price Tool",
        func=get_stock_price,
        description="Use this to get current stock price. Input should be a stock ticker like AAPL or TSLA."
    ),
    Tool(
        name="Company News Tool",
        func=get_company_news,
        description="Use this to get recent news about a company. Input should be a stock ticker like AAPL or TSLA."
    ),
    Tool(
        name="Summarizer Tool",
        func=summarize,
        description="Use this to summarize a paragraph of text. Input should be raw text."
    ),
]


llm = ChatOpenAI(model="gpt-4", temperature=0)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

response = agent.invoke("What is the current price of AAPL and summarize the latest news about it?")
print(response)
