import os
from enum import Enum
from dataclasses import dataclass, field, fields
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from typing_extensions import Annotated
from dataclasses import dataclass

DEFAULT_REPORT_STRUCTURE = """The report structure should focus on breaking-down the user-provided ticker:

1. Introduction (no research needed)
   - Short overview of the ticker/company

2. Fundamental Analysis
    - Fundamentals of the company
          - Include any key concepts and definitions
   - Provide real-world examples or case studies where applicable
   - Based on the fundamental data, give a score between 1(weakest)-10(strongest)

3. Recent events that have affected the company
    - Recent negative news
    - Recent positive news

4. Upcoming events and earnings
    - Negative news
    - Positive news
    - Based on the recent and upcoming news you have gathered and analyzed, give a ranking between Low, Medium and High risk for the event risk of the stock

5. Our Advanced Metrics 
    - Use the tool "call_api" to get our advanced metrics for this company
    - Provide a table of the metrics

6. Conclusion
   - Provide a concise summary of the report
   - Provide a table providing the scores and rankings

"""
C3_WAR_PROMPT = """ 
The report structure should focus on breaking-down the user-provided country: 

1. Executive Summary
   - Brief overview of the topic, key events and current status
   - High-level assessment of strategic and operational implications

2. Background and context
    - Origins of any conflict and key historical events.
    - Overview of parties involved, its military capabilities, and objectives.
    - Political and regional dynamics influencing the war.

3. Current Operational Situation
    - Overview of ongoing military operations.
    - Summary of major battles, airstrikes, and troop movements.

4. Timeline 
    - A timeline of key events

5. Future Scenarios and Recommendations
    - Short-term and long-term outcomes.
    - Military, political, and intelligence recommendations.
    - Implications for regional and global security.

6.  Our current stand on the country
    - Use the tool get_sentiment to get current stand on the country

"""

class SearchAPI(Enum):
    PERPLEXITY = "perplexity"
    TAVILY = "tavily"


class PlannerProvider(Enum):
    OPENAI = "openai"
    GROQ = "groq"


@dataclass(kw_only=True)
class Configuration:
    """The configurable fields for the chatbot."""

    report_structure: str = (
        C3_WAR_PROMPT  # Defaults to the default report structure
    )
    number_of_queries: int = 2  # Number of search queries to generate per iteration
    max_search_depth: int = 2  # Maximum number of reflection + search iterations
    planner_provider: PlannerProvider = (
        PlannerProvider.OPENAI
    )  # Defaults to OpenAI as provider
    planner_model: str = "o1-mini"  # Defaults to OpenAI o3-mini as planner model
    writer_model: str = (
        "gpt-4o"  # Defaults to Anthropic as provider, claude-3-5-sonnet-latest
    )
    search_api: SearchAPI = SearchAPI.TAVILY  # Default to TAVILY

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        return cls(**{k: v for k, v in values.items() if v})
