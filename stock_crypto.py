from langchain.agents import initialize_agent, Tool
from langchain_openai import ChatOpenAI
from langchain.agents.agent_types import AgentType

import dotenv
dotenv.load_dotenv()

# Fake tools
def get_top_stocks(_input: str = "") -> str:
    stocks = [
        "AAPL: $172", "GOOG: $140", "AMZN: $130", "MSFT: $310", "TSLA: $700",
        "META: $280", "NVDA: $950", "BABA: $90", "ORCL: $120", "CSCO: $50",
        "ADBE: $500", "INTC: $40", "AMD: $120", "CRM: $230", "PYPL: $70",
        "NFLX: $400", "SHOP: $60", "UBER: $65", "DIS: $90", "BA: $180"
    ]
    return "\n".join(stocks)

def get_top_cryptos(_input: str = "") -> str:
    cryptos = [
        "BTC: $68000", "ETH: $3600", "SOL: $180", "BNB: $600", "XRP: $0.5",
        "ADA: $0.45", "DOGE: $0.07", "AVAX: $40", "DOT: $6", "SHIB: $0.00001",
        "MATIC: $0.9", "LTC: $90", "LINK: $16", "ATOM: $10", "XLM: $0.13",
        "TRX: $0.12", "NEAR: $7", "ARB: $1.2", "OP: $2.5", "APE: $1.4"
    ]
    return "\n".join(cryptos)


# Tool 2: calculate average price
def average_price(data: str) -> str:
    prices = []
    for line in data.splitlines():
        line = line.strip()
        if not line:
            continue
        if ":" in line:
            parts = line.split(":")
            if len(parts) == 2:
                price_str = parts[1].strip().replace("$", "").replace(",", "")
                try:
                    price = float(price_str)
                    prices.append(price)
                except ValueError:
                    continue
        else:
            # Handle comma-separated or space-separated numbers
            for price_str in line.replace(",", " ").split():
                try:
                    price = float(price_str)
                    prices.append(price)
                except ValueError:
                    continue
    if not prices:
        return "No valid prices found."
    avg = sum(prices) / len(prices)
    return f"Average price is ${avg:.2f}"


# Define tools
tools = [
    Tool(
        name="StockTool",
        func=get_top_stocks,
        description="Returns a list of top 20 most expensive stocks. Use when the query is about stocks."
    ),
    Tool(
        name="CryptoTool",
        func=get_top_cryptos,
        description="Returns a list of top 20 most expensive cryptocurrencies. Use when the query is about cryptos."
    ),
    Tool(
        name="AveragePriceTool",
        func=average_price,
        description="Calculates the average price from stocks or cryptos list. Use when the query is about average prices."
    ),
]

# Initialize Agent
llm = ChatOpenAI(temperature=0, model="gpt-4")  # or gpt-3.5
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Example calls
# agent.run("Give me the top 5 most expensive stocks")
agent.invoke("Give me the average price of top 5 most expensive stocks")
