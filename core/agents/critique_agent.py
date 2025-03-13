from pydantic_ai import Agent
from pydantic import BaseModel
from pydantic_ai.settings import ModelSettings
from core.skills.final_response import get_response
import os
from dotenv import load_dotenv

from core.utils.openai_client import get_client
from pydantic_ai.models.openai import OpenAIModel

load_dotenv()

class CritiqueOutput(BaseModel):
    feedback: str
    terminate: bool
    final_response: str

class CritiqueInput(BaseModel):
    current_step : str
    orignal_plan : str
    tool_response: str
    ss_analysis: str


#System prompt for Critique agent
CA_SYS_PROMPT = """
<agent_role>
    Analysis and quality control agent for web automation tasks within multi-agent system 
    (Planner -> Browser Agent -> Critique). Evaluates execution progress and determines task completion.
</agent_role>

<input_components>
    - Original plan: Sequential steps for task completion
    - Current step: Active step from planner
    - Tool response: Browser agent execution result
    - Screenshot analysis: Visual comparison of pre/post action states
</input_components>

<core_responsibilities>
    <progress_evaluation>
        - Analyze step execution success
        - Compare against original plan
        - Track overall task progress
        - Identify execution errors
    </progress_evaluation>

    <feedback_generation>
        - Provide specific, actionable feedback
        - Include current position in plan
        - Address multi-action steps
        - Suggest step refinements when needed
        - Reference tool responses and visual changes
    </feedback_generation>

    <termination_criteria>
        - Last step completed with requirements met
        - Non-recoverable failure detected
        - Action loop detected (>5 iterations)
        - Multiple approach failures (>7 attempts)
        - Human intervention required
    </termination_criteria>
</core_responsibilities>

<response_guidelines>
    <feedback_format>
        1. Original plan
        2. Current progress status
        3. Execution analysis
        4. Next step recommendations
    </feedback_format>

    <final_response>
        - Provide concrete answers/results
        - Include specific data/information
        - Explain termination reasons if applicable
        - No generic completion messages
    </final_response>
</response_guidelines>

<io_schema>
    <input>
        {
            "current_step": "string",
            "orignal_plan": "string",
            "tool_response": "string",
            "ss_analysis": "string"
        }
    </input>
    <output>
        {
            "feedback": "string",
            "terminate": "boolean",
            "final_response": "string"
        }
    </output>
</io_schema>
"""

# Setup CA
CA_client = get_client()
CA_model = OpenAIModel(model_name = os.getenv("AGENTIC_BROWSER_TEXT_MODEL"), openai_client=CA_client)
CA_agent = Agent(
    model=CA_model, 
    name="Critique Agent",
    system_prompt=CA_SYS_PROMPT,
    retries=3,
    model_settings=ModelSettings(
        temperature=0.5,
    ),
    result_type=CritiqueOutput,
)

@CA_agent.tool_plain
async def final_response(plan: str, browser_response: str, current_step: str) -> str:

    response = await get_response(plan, browser_response, current_step)

    return response
