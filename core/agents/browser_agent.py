from pydantic_ai import RunContext
from pydantic import BaseModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.settings import ModelSettings
import os
from dotenv import load_dotenv

from pydantic_ai import Agent
# from ae.core.skills.enter_text_and_click import enter_text_and_click
from core.skills.enter_text_using_selector import bulk_enter_text
from core.skills.enter_text_using_selector import entertext
from core.skills.get_dom_with_content_type import get_dom_field_func, get_dom_texts_func
from core.skills.get_url import geturl
from core.skills.open_url import openurl
from core.skills.pdf_text_extractor import extract_text_from_pdf
from core.skills.google_search import google_search
from core.skills.press_key_combination import press_key_combination
from core.skills.click_using_selector import click


from core.utils.openai_client import get_client

load_dotenv()


class current_step_class(BaseModel):
    current_step : str

#System prompt for Browser Agent
BA_SYS_PROMPT = """
<agent_role>
    Web navigation and automation agent responsible for executing browser actions within multi-agent system 
    (Planner -> Browser Agent -> Critique). Executes single actions per iteration using provided tools.
</agent_role>

<core_rules>
    - Always perform tool calls
    - Use DOM representations for element interactions
    - Use only "mmid" attribute for element selection
    - Extract mmid values from fetched DOM
    - Never expose webpage URLs directly
    - Operate on current page unless specified
    - Use get_dom_fields for field discovery
</core_rules>

<action_guidelines>
    <search>
        - Use google_search_tool for API-based searches
        - Use Enter key for search field submission
        - Use click for other form submissions
    </search>

    <navigation>
        - Use open_url_tool only when explicitly instructed
        - Request URL if not provided
        - Offer click actions instead of URL sharing
    </navigation>

    <input_handling>
        - Match input format requirements
        - Clear fields before new input:
            1. Enter empty string
            2. Press Ctrl+A
            3. Press Delete/Backspace
            4. Enter new value
        - Respect field types (dates, numbers, etc.)
        - Request clarification for ambiguous choices
    </input_handling>

    <response_format>
        - Summarize actions performed
        - Report success/failure status
        - Answer queries using DOM content only
        - Avoid repeating failed actions
        - Report cycling behavior to critique
    </response_format>
</action_guidelines>

<available_tools>
    1. google_search_tool(query: str, num: int = 10)
       - Performs Google search via API
       - Returns formatted results

    2. enter_text_tool(entry: EnterTextEntry)
       - Enters text into DOM element
       - Uses mmid selector
       - Simulates keyboard input

    3. bulk_enter_text_tool(entries: List[EnterTextEntry])
       - Bulk text entry operation
       - Multiple field support

    4. get_dom_text()
       - Returns page text content
       - Use for text extraction

    5. get_dom_fields()
       - Returns interactive elements
       - Use for field discovery

    6. get_url_tool()
       - Returns current page URL

    7. open_url_tool(url: str, timeout: int = 3)
       - Navigates to specified URL
       - Waits for page load

    8. extract_text_from_pdf_tool(pdf_url: str)
       - Extracts PDF text content

    9. press_key_combination_tool(keys: str)
       - Simulates keyboard shortcuts
       - Example: "Control+A"

    10. click_tool(selector: str, wait_before_execution: float = 0.0)
        - Executes click actions
        - Uses mmid selector
        - Handles select/option elements
</available_tools>
"""

# Setup BA
BA_client = get_client()
BA_model = OpenAIModel(model_name = os.getenv("AGENTIC_BROWSER_TEXT_MODEL"), openai_client=BA_client)
BA_agent = Agent(
    model=BA_model, 
    system_prompt=BA_SYS_PROMPT,
    deps_type=current_step_class,
    name="Browser Agent",
    retries=3,
    model_settings=ModelSettings(
        temperature=0.5,
    ),
)


# BA Tools
@BA_agent.tool_plain
async def google_search_tool(query: str, num: int = 10) -> str:
    """
    Performs a Google search using the query and num parameters.
    """
    return await google_search(query=query, num=num)

@BA_agent.tool_plain
async def bulk_enter_text_tool(entries) -> str:
    """
    This function enters text into multiple DOM elements using a bulk operation.
    It takes a list of dictionaries, where each dictionary contains a 'query_selector' and 'text' pair.
    The function internally calls the 'entertext' function to perform the text entry operation for each entry.
    """
    return await bulk_enter_text(entries=entries)

@BA_agent.tool_plain
async def enter_text_tool(entry) -> str:
    """
    Enters text into a DOM element identified by a CSS selector.
    """
    return await entertext(entry=entry)

@BA_agent.tool_plain
async def get_dom_text() -> str:

    return await get_dom_texts_func()

@BA_agent.tool
async def get_dom_fields(ctx: RunContext[current_step_class]) -> str:
    return await get_dom_field_func(ctx.deps.current_step)

@BA_agent.tool_plain
async def get_url_tool() -> str:
    """
    Returns the full URL of the current page
    """
    return await geturl()

@BA_agent.tool_plain
async def click_tool(selector: str, wait_before_execution: float = 0.0) -> str:
    """
    Executes a click action on the element matching the given query selector string within the currently open web page.
    
    Parameters:
    - selector: The query selector string to identify the element for the click action
    - wait_before_execution: Optional wait time in seconds before executing the click event logic
    
    Returns:
    - A message indicating success or failure of the click action
    """
    return await click(selector=selector, wait_before_execution=wait_before_execution)

@BA_agent.tool_plain
async def open_url_tool(url: str, timeout:int = 3) -> str:
    """
    Opens the specified URL in the browser.
    """
    return await openurl(url=url, timeout=timeout)

@BA_agent.tool_plain
async def extract_text_from_pdf_tool(pdf_url: str) -> str:
    """
    Extracts the text content from a PDF file available at the specified URL.
    """
    return await extract_text_from_pdf(pdf_url=pdf_url)


@BA_agent.tool_plain
async def press_key_combination_tool(keys: str) -> str:
    """
    Presses the specified key combination in the browser.
    Parameter:
    - keys (str): Key combination as string, e.g., "Control+C" for Ctrl+C, "Control+Shift+I" for Ctrl+Shift+I
    Returns:
    - str: Status of the operation
    """
    return await press_key_combination(key_combination=keys)

