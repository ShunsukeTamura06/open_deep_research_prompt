"""Local evaluation example without LangSmith dependencies."""

import asyncio
import json
from typing import Dict, List
import uuid
from langgraph.checkpoint.memory import MemorySaver
from src.open_deep_research.deep_researcher import deep_researcher


class LocalEvaluator:
    """Simple local evaluation system."""
    
    def __init__(self):
        self.results = []
    
    async def evaluate_research(self, query: str, expected_topics: List[str] = None) -> Dict:
        """Evaluate a single research query."""
        
        config = {
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "allow_clarification": False,
                "search_api": "tavily",
                "max_researcher_iterations": 2,
                "research_model": "openai:gpt-4o-mini",
                "final_report_model": "openai:gpt-4o-mini",
                "summarization_model": "openai:gpt-4o-mini",
                "compression_model": "openai:gpt-4o-mini"
            }
        }
        
        try:
            graph = deep_researcher.compile(checkpointer=MemorySaver())
            result = await graph.ainvoke(
                {"messages": [{"role": "user", "content": query}]},
                config
            )
            
            # Simple evaluation metrics
            final_report = result.get("final_report", "")
            
            evaluation = {
                "query": query,
                "report_length": len(final_report),
                "has_sources": "### Sources" in final_report,
                "has_sections": final_report.count("##") > 0,
                "expected_topics_covered": 0,
                "success": True,
                "error": None
            }
            
            # Check if expected topics are covered
            if expected_topics:
                covered = sum(1 for topic in expected_topics if topic.lower() in final_report.lower())
                evaluation["expected_topics_covered"] = covered / len(expected_topics)
            
            print(f"✅ Evaluation completed for: {query[:50]}...")
            return evaluation
            
        except Exception as e:
            evaluation = {
                "query": query,
                "success": False,
                "error": str(e)
            }
            print(f"❌ Evaluation failed for: {query[:50]}... - {e}")
            return evaluation
    
    async def run_evaluation_suite(self):
        """Run a suite of evaluation tests."""
        
        test_cases = [
            {
                "query": "What are the latest developments in renewable energy technology?",
                "expected_topics": ["solar", "wind", "battery", "energy storage"]
            },
            {
                "query": "Analyze the current state of the AI inference market",
                "expected_topics": ["market size", "competition", "providers", "growth"]
            },
            {
                "query": "Compare the top 3 cloud computing providers",
                "expected_topics": ["AWS", "Azure", "Google Cloud", "comparison"]
            }
        ]
        
        print("Starting Local Evaluation Suite")
        print("=" * 50)
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\nTest {i}/{len(test_cases)}: {test_case['query'][:50]}...")
            
            result = await self.evaluate_research(
                test_case["query"],
                test_case.get("expected_topics")
            )
            
            self.results.append(result)
            
            # Print basic metrics
            if result["success"]:
                print(f"  📊 Report length: {result['report_length']} chars")
                print(f"  📚 Has sources: {result['has_sources']}")
                print(f"  📝 Has sections: {result['has_sections']}")
                if "expected_topics_covered" in result:
                    print(f"  🎯 Topics covered: {result['expected_topics_covered']:.1%}")
        
        self.print_summary()
    
    def print_summary(self):
        """Print evaluation summary."""
        print("\n" + "=" * 50)
        print("EVALUATION SUMMARY")
        print("=" * 50)
        
        successful = [r for r in self.results if r["success"]]
        failed = [r for r in self.results if not r["success"]]
        
        print(f"Total tests: {len(self.results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        print(f"Success rate: {len(successful)/len(self.results):.1%}")
        
        if successful:
            avg_length = sum(r["report_length"] for r in successful) / len(successful)
            sources_pct = sum(1 for r in successful if r["has_sources"]) / len(successful)
            sections_pct = sum(1 for r in successful if r["has_sections"]) / len(successful)
            
            print(f"\nAverage report length: {avg_length:.0f} chars")
            print(f"Reports with sources: {sources_pct:.1%}")
            print(f"Reports with sections: {sections_pct:.1%}")
        
        if failed:
            print("\nFailed tests:")
            for result in failed:
                print(f"  - {result['query'][:50]}... : {result['error']}")
    
    def save_results(self, filename: str = "evaluation_results.json"):
        """Save results to JSON file."""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {filename}")


async def main():
    """Run the local evaluation."""
    evaluator = LocalEvaluator()
    await evaluator.run_evaluation_suite()
    evaluator.save_results()


if __name__ == "__main__":
    asyncio.run(main())