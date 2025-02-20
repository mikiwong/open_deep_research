import asyncio
import uuid
from src.open_deep_research.graph import graph

async def run_graph():
    thread = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
            "search_api": "tavily",
            "planner_provider": "openai",
            "max_search_depth": 1,
            "planner_model": "o1-mini",
            "writer_model": "o1-mini"
        }
    }

    topic = """I am a research analyst for a trading and investment firm
    \n- I like to find out about the fundamentals of a company
    \n- I also like to know about recent events that have affected it (list them in negative and positive news)
    \n- I also like to know about upcoming events and earnings for it (list them in negative and positive news)
    \n- Based on the data you have gathered and analyzed, give me a score between 1 (weak fundamentals) -10 (strong fundamentals) for the fundamentals, and a ranking between Low, Medium and High risk for the event risk of the stock
    \nThe company I like to find out on is BBAI"""

    # Run the graph asynchronously
    async for event in graph.astream({"topic": topic}, thread, stream_mode="updates"):
        print(event)
        print("\n")

# Run the async function
asyncio.run(run_graph())
