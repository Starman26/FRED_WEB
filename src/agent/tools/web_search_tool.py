"""
web_search_tool.py

Web search via Tavily API for industrial automation troubleshooting.
"""
import os
from langchain_core.tools import tool
from src.agent.utils.logger import logger


PRIORITY_DOMAINS = [
    "support.industry.siemens.com",
    "cache.industry.siemens.com",
    "automation.siemens.com",
    "new.siemens.com",
    "plcforum.uz.ua",
]


@tool
def web_search_diagnostic(query: str) -> str:
    """Search the internet for technical documentation, troubleshooting guides,
    and forum posts about industrial automation equipment (PLCs, Cobots, sensors).

    Use this tool when:
    - The equipment manual doesn't have information about the problem
    - You need updated information (firmware updates, known bugs, patches)
    - You want to find community solutions to similar problems
    - You need error code explanations not in the local manual

    Args:
        query: Search query in ENGLISH. Be specific: include equipment model,
               error code, and symptom. Example: "S7-1200 cannot download program
               TIA Portal communication error"

    Returns:
        Relevant search results with titles, URLs, and content snippets.
    """
    try:
        tavily_key = os.getenv("TAVILY_API_KEY")
        if not tavily_key:
            return "Web search not available: TAVILY_API_KEY not configured."

        from tavily import TavilyClient
        client = TavilyClient(api_key=tavily_key)

        if "siemens" not in query.lower() and "plc" not in query.lower():
            query = f"Siemens industrial automation {query}"

        use_domains = None
        if any(kw in query.lower() for kw in ["siemens", "s7", "plc", "tia"]):
            use_domains = PRIORITY_DOMAINS

        results = client.search(
            query=query,
            search_depth="advanced",
            max_results=5,
            include_domains=use_domains,
        )

        if not results or not results.get("results"):
            return "No relevant results found for this query."

        formatted = []
        for i, result in enumerate(results["results"][:5], 1):
            title = result.get("title", "No title")
            url = result.get("url", "")
            content = result.get("content", "")[:500]

            formatted.append(
                f"**Result {i}: {title}**\n"
                f"Source: {url}\n"
                f"{content}\n"
            )

        return "\n---\n".join(formatted)

    except ImportError:
        return "Web search not available: tavily package not installed. Run: pip install tavily-python"
    except Exception as e:
        logger.error("web_search_tool", f"Web search error: {e}")
        return f"Web search error: {str(e)}"


def get_web_search_tool():
    """Returns the web search tool if Tavily is configured, None otherwise."""
    if os.getenv("TAVILY_API_KEY"):
        return web_search_diagnostic
    return None
