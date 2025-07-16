"""Simple example of using the Open Deep Research system."""

import asyncio
import os
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
import uuid

# Load environment variables
load_dotenv()

# Import the deep researcher
from src.open_deep_research.deep_researcher import deep_researcher


async def simple_research_example():
    """Example of conducting simple research."""
    
    # Create a simple config
    config = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
            "allow_clarification": False,  # Skip clarification for this example
            "search_api": "tavily",
            "max_researcher_iterations": 2,
            "research_model": "openai:gpt-4o-mini",
            "final_report_model": "openai:gpt-4o-mini",
            "summarization_model": "openai:gpt-4o-mini",
            "compression_model": "openai:gpt-4o-mini"
        }
    }
    
    # Research question
    research_question = "What are the latest developments in renewable energy technology in 2024?"
    
    print(f"Starting research on: {research_question}")
    print("-" * 50)
    
    try:
        # Run the research
        graph = deep_researcher.compile(checkpointer=MemorySaver())
        result = await graph.ainvoke(
            {"messages": [{"role": "user", "content": research_question}]},
            config
        )
        
        print("Research Complete!")
        print("=" * 50)
        print(result["final_report"])
        
    except Exception as e:
        print(f"Error during research: {e}")
        print("Make sure you have set the required environment variables:")
        print("- OPENAI_API_KEY (if using OpenAI models)")
        print("- TAVILY_API_KEY (if using Tavily search)")


if __name__ == "__main__":
    asyncio.run(simple_research_example())