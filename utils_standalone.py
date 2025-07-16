import os
import aiohttp
import asyncio
import logging
import warnings
from datetime import datetime, timedelta, timezone
from typing import Annotated, List, Literal, Dict, Optional, Any
from langchain_core.tools import BaseTool, StructuredTool, tool, ToolException, InjectedToolArg
from langchain_core.messages import HumanMessage, AIMessage, MessageLikeRepresentation, filter_messages
from langchain_core.runnables import RunnableConfig
from langchain_core.language_models import BaseChatModel
from langchain.chat_models import init_chat_model
from tavily import AsyncTavilyClient
from mcp import McpError
from langchain_mcp_adapters.client import MultiServerMCPClient
from open_deep_research.state import Summary, ResearchComplete
from open_deep_research.configuration import SearchAPI, Configuration
from open_deep_research.prompts import summarize_webpage_prompt

# LangSmithなしでのメモリベーストークン管理
_memory_store = {}

##########################
# Tavily Search Tool Utils
##########################
TAVILY_SEARCH_DESCRIPTION = (
    "A search engine optimized for comprehensive, accurate, and trusted results. "
    "Useful for when you need to answer questions about current events."
)
@tool(description=TAVILY_SEARCH_DESCRIPTION)
async def tavily_search(
    queries: List[str],
    max_results: Annotated[int, InjectedToolArg] = 5,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
    config: RunnableConfig = None
) -> str:
    """
    Fetches results from Tavily search API.

    Args
        queries (List[str]): List of search queries, you can pass in as many queries as you need.
        max_results (int): Maximum number of results to return
        topic (Literal['general', 'news', 'finance']): Topic to filter results by

    Returns:
        str: A formatted string of search results
    """
    search_results = await tavily_search_async(
        queries,
        max_results=max_results,
        topic=topic,
        include_raw_content=True,
        config=config
    )
    # Format the search results and deduplicate results by URL
    formatted_output = f"Search results: \n\n"
    unique_results = {}
    for response in search_results:
        for result in response['results']:
            url = result['url']
            if url not in unique_results:
                unique_results[url] = {**result, "query": response['query']}
    configurable = Configuration.from_runnable_config(config)
    max_char_to_include = 50_000   # NOTE: This can be tuned by the developer. This character count keeps us safely under input token limits for the latest models.
    model_api_key = get_api_key_for_model(configurable.summarization_model, config)
    summarization_model = init_chat_model(
        model=configurable.summarization_model,
        max_tokens=configurable.summarization_model_max_tokens,
        api_key=model_api_key,
        # LangSmithタグを削除
    ).with_structured_output(Summary).with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    async def noop():
        return None
    summarization_tasks = [
        noop() if not result.get("raw_content") else summarize_webpage(
            summarization_model, 
            result['raw_content'][:max_char_to_include],
        )
        for result in unique_results.values()
    ]
    summaries = await asyncio.gather(*summarization_tasks)
    summarized_results = {
        url: {'title': result['title'], 'content': result['content'] if summary is None else summary}
        for url, result, summary in zip(unique_results.keys(), unique_results.values(), summaries)
    }
    for i, (url, result) in enumerate(summarized_results.items()):
        formatted_output += f"\n\n--- SOURCE {i+1}: {result['title']} ---\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"SUMMARY:\n{result['content']}\n\n"
        formatted_output += "\n\n" + "-" * 80 + "\n"
    if summarized_results:
        return formatted_output
    else:
        return "No valid search results found. Please try different search queries or use a different search API."


async def tavily_search_async(search_queries, max_results: int = 5, topic: Literal["general", "news", "finance"] = "general", include_raw_content: bool = True, config: RunnableConfig = None):
    tavily_async_client = AsyncTavilyClient(api_key=get_tavily_api_key(config))
    search_tasks = []
    for query in search_queries:
            search_tasks.append(
                tavily_async_client.search(
                    query,
                    max_results=max_results,
                    include_raw_content=include_raw_content,
                    topic=topic
                )
            )
    search_docs = await asyncio.gather(*search_tasks)
    return search_docs

async def summarize_webpage(model: BaseChatModel, webpage_content: str) -> str:
    try:
        summary = await asyncio.wait_for(
            model.ainvoke([HumanMessage(content=summarize_webpage_prompt.format(webpage_content=webpage_content, date=get_today_str()))]),
            timeout=60.0
        )
        return f"""<summary>\n{summary.summary}\n</summary>\n\n<key_excerpts>\n{summary.key_excerpts}\n</key_excerpts>"""
    except (asyncio.TimeoutError, Exception) as e:
        print(f"Failed to summarize webpage: {str(e)}")
        return webpage_content


##########################
# MCP Utils - LangSmithなしでのメモリベース実装
##########################
async def get_mcp_access_token(
    supabase_token: str,
    base_mcp_url: str,
) -> Optional[Dict[str, Any]]:
    try:
        form_data = {
            "client_id": "mcp_default",
            "subject_token": supabase_token,
            "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
            "resource": base_mcp_url.rstrip("/") + "/mcp",
            "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                base_mcp_url.rstrip("/") + "/oauth/token",
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data=form_data,
            ) as token_response:
                if token_response.status == 200:
                    token_data = await token_response.json()
                    return token_data
                else:
                    response_text = await token_response.text()
                    logging.error(f"Token exchange failed: {response_text}")
    except Exception as e:
        logging.error(f"Error during token exchange: {e}")
    return None

async def get_tokens(config: RunnableConfig):
    """LangSmithなしでのメモリベーストークン管理"""
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id:
        return None
    user_id = config.get("metadata", {}).get("owner")
    if not user_id:
        return None
    
    token_key = f"{user_id}:tokens"
    if token_key not in _memory_store:
        return None
    
    token_data = _memory_store[token_key]
    expires_in = token_data.get("expires_in", 3600)  # Default 1 hour
    created_at = token_data.get("created_at")
    if not created_at:
        return None
    
    current_time = datetime.now(timezone.utc)
    expiration_time = created_at + timedelta(seconds=expires_in)
    if current_time > expiration_time:
        del _memory_store[token_key]
        return None

    return token_data.get("value")

async def set_tokens(config: RunnableConfig, tokens: dict[str, Any]):
    """LangSmithなしでのメモリベーストークン管理"""
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id:
        return
    user_id = config.get("metadata", {}).get("owner")
    if not user_id:
        return
    
    token_key = f"{user_id}:tokens"
    _memory_store[token_key] = {
        "value": tokens,
        "created_at": datetime.now(timezone.utc),
        "expires_in": tokens.get("expires_in", 3600)
    }
    return

async def fetch_tokens(config: RunnableConfig) -> dict[str, Any]:
    current_tokens = await get_tokens(config)
    if current_tokens:
        return current_tokens
    supabase_token = config.get("configurable", {}).get("x-supabase-access-token")
    if not supabase_token:
        return None
    mcp_config = config.get("configurable", {}).get("mcp_config")
    if not mcp_config or not mcp_config.get("url"):
        return None
    mcp_tokens = await get_mcp_access_token(supabase_token, mcp_config.get("url"))

    await set_tokens(config, mcp_tokens)
    return mcp_tokens

def wrap_mcp_authenticate_tool(tool: StructuredTool) -> StructuredTool:
    old_coroutine = tool.coroutine
    async def wrapped_mcp_coroutine(**kwargs):
        def _find_first_mcp_error_nested(exc: BaseException) -> McpError | None:
            if isinstance(exc, McpError):
                return exc
            if isinstance(exc, ExceptionGroup):
                for sub_exc in exc.exceptions:
                    if found := _find_first_mcp_error_nested(sub_exc):
                        return found
            return None
        try:
            return await old_coroutine(**kwargs)
        except BaseException as e_orig:
            mcp_error = _find_first_mcp_error_nested(e_orig)
            if not mcp_error:
                raise e_orig
            error_details = mcp_error.error
            is_interaction_required = getattr(error_details, "code", None) == -32003
            error_data = getattr(error_details, "data", None) or {}
            if is_interaction_required:
                message_payload = error_data.get("message", {})
                error_message_text = "Required interaction"
                if isinstance(message_payload, dict):
                    error_message_text = (
                        message_payload.get("text") or error_message_text
                    )
                if url := error_data.get("url"):
                    error_message_text = f"{error_message_text} {url}"
                raise ToolException(error_message_text) from e_orig
            raise e_orig
    tool.coroutine = wrapped_mcp_coroutine
    return tool

async def load_mcp_tools(
    config: RunnableConfig,
    existing_tool_names: set[str],
) -> list[BaseTool]:
    configurable = Configuration.from_runnable_config(config)
    if configurable.mcp_config and configurable.mcp_config.auth_required:
        mcp_tokens = await fetch_tokens(config)
    else:
        mcp_tokens = None
    if not (configurable.mcp_config and configurable.mcp_config.url and configurable.mcp_config.tools and (mcp_tokens or not configurable.mcp_config.auth_required)):
        return []
    tools = []
    # TODO: When the Multi-MCP Server support is merged in OAP, update this code.
    server_url = configurable.mcp_config.url.rstrip("/") + "/mcp"
    mcp_server_config = {
        "server_1":{
            "url": server_url,
            "headers": {"Authorization": f"Bearer {mcp_tokens['access_token']}"} if mcp_tokens else None,
            "transport": "streamable_http"
        }
    }
    try:
        client = MultiServerMCPClient(mcp_server_config)
        mcp_tools = await client.get_tools()
    except Exception as e:
        print(f"Error loading MCP tools: {e}")
        return []
    for tool in mcp_tools:
        if tool.name in existing_tool_names:
            warnings.warn(
                f"Trying to add MCP tool with a name {tool.name} that is already in use - this tool will be ignored."
            )
            continue
        if tool.name not in set(configurable.mcp_config.tools):
            continue
        tools.append(wrap_mcp_authenticate_tool(tool))
    return tools


##########################
# Tool Utils
##########################
async def get_search_tool(search_api: SearchAPI):
    if search_api == SearchAPI.ANTHROPIC:
        return [{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}]
    elif search_api == SearchAPI.OPENAI:
        return [{"type": "web_search_preview"}]
    elif search_api == SearchAPI.TAVILY:
        search_tool = tavily_search
        search_tool.metadata = {**(search_tool.metadata or {}), "type": "search", "name": "web_search"}
        return [search_tool]
    elif search_api == SearchAPI.NONE:
        return []
    
async def get_all_tools(config: RunnableConfig):
    tools = [tool(ResearchComplete)]
    configurable = Configuration.from_runnable_config(config)
    search_api = SearchAPI(get_config_value(configurable.search_api))
    tools.extend(await get_search_tool(search_api))
    existing_tool_names = {tool.name if hasattr(tool, "name") else tool.get("name", "web_search") for tool in tools}
    mcp_tools = await load_mcp_tools(config, existing_tool_names)
    tools.extend(mcp_tools)
    return tools

def get_notes_from_tool_calls(messages: list[MessageLikeRepresentation]):
    return [tool_msg.content for tool_msg in filter_messages(messages, include_types="tool")]


##########################
# Model Provider Native Websearch Utils
##########################
def anthropic_websearch_called(response):
    try:
        usage = response.response_metadata.get("usage")
        if not usage:
            return False
        server_tool_use = usage.get("server_tool_use")
        if not server_tool_use:
            return False
        web_search_requests = server_tool_use.get("web_search_requests")
        if web_search_requests is None:
            return False
        return web_search_requests > 0
    except (AttributeError, TypeError):
        return False

def openai_websearch_called(response):
    tool_outputs = response.additional_kwargs.get("tool_outputs")
    if tool_outputs:
        for tool_output in tool_outputs:
            if tool_output.get("type") == "web_search_call":
                return True
    return False


##########################
# Token Limit Exceeded Utils
##########################
def is_token_limit_exceeded(exception: Exception, model_name: str = None) -> bool:
    error_str = str(exception).lower()
    provider = None
    if model_name:
        model_str = str(model_name).lower()
        if model_str.startswith('openai: