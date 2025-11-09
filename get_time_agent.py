"""
=============================================================
Get Time Agent -- LLM Agent tool use and flow control example
=============================================================

The LLM uses a tool (Python function) to get the time from an API.

Pydantic AI uses the input type "TimeInput" to require a timezone in string
format.

The Pydantic AI output type "TimeOutput" forces the LLM to return the time
as a datetime object and the UTC offset as a string.

Flow control is orchestrated by Langgraph.

After the time is fetched by the tool, another LLM call is made asking
the LLM to identify what month it is and always reply with the name of the
month plus an emoji.

Then, flow control determines whether the time is AM or PM.

The result is enriched with the AM/PM info and the final output is provided
to the user.
"""

import urllib3
import json
from typing import TypedDict, Literal
from datetime import datetime
import logging
from dataclasses import dataclass

from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.models.openai import OpenAIChatModelSettings

logger = logging.getLogger(__name__)
logging.basicConfig(encoding="utf-8", level=logging.ERROR)


# Define state schema
class AgentState(TypedDict):
    get_time_prompt: str
    timezone: str
    time_data: datetime | None
    utc_offset: str | None
    month_name: str | None
    month_emoji: str | None
    final_answer: dict | None


# input schema
@dataclass
class TimeInput:
    timezone_input: str


# output schema
class TimeOutput(BaseModel):
    time_data_output: datetime
    utc_offset_output: str


class MonthNameOutput(BaseModel):
    month_name_output: str
    month_name_output_emoji: str


# Pydantic AI model + agent setup
model = OpenAIChatModel(
    model_name="gpt-oss:20b",
    settings=OpenAIChatModelSettings(temperature=0.0, openai_reasoning_effort="low"),
    provider=OllamaProvider(base_url="http://localhost:11434/v1"),
)
time_agent = Agent(
    model,
    deps_type=TimeInput,
    output_type=TimeOutput,
    system_prompt=(
        """Use the `get_time` function to get the time.
        Return the current time as a datetime object, and return the UTC offset."""
    ),
)
month_name_agent = Agent(
    model,
    output_type=MonthNameOutput,
    system_prompt=(
        """Return the name of the month and also return an emoji that most represents that month."""
    ),
)


# Define the get_time tool for the time_agent
@time_agent.tool
async def get_time(ctx: RunContext[TimeInput]) -> str:
    """return the time"""
    url = f"http://worldtimeapi.org/api/timezone/{ctx.deps.timezone_input}"
    response = urllib3.request("GET", url)
    return json.loads(response.data)["datetime"]


# Define nodes for the graph
def get_time_node(state: AgentState) -> AgentState:
    time_data = time_agent.run_sync(
        state["get_time_prompt"], deps=TimeInput(state["timezone"])
    )
    logger.info(f"time_data.output: {time_data.output}")
    state["time_data"] = time_data.output.time_data_output
    state["utc_offset"] = time_data.output.utc_offset_output
    return state


def get_month_name_node(state: AgentState) -> AgentState:
    month_name_data = month_name_agent.run_sync(
        f"What month is it according to this datetime object?: {state['time_data']}"
    )
    logger.info(f"month_name_data.output: {month_name_data.output}")
    state["month_name"] = month_name_data.output.month_name_output
    state["month_emoji"] = month_name_data.output.month_name_output_emoji
    return state


def format_response(state: AgentState) -> AgentState:
    state["final_answer"] = {
        "current_time": f"{state['time_data']}",  # this dumps the datetime object as a string into 'current_time'
        "timezone": f"{state['timezone']}",
        "utc_offset": state["utc_offset"],
        "month_name": state["month_name"],
        "month_emoji": state["month_emoji"],
    }
    return state


def add_AM(state: AgentState) -> AgentState:
    state["final_answer"]["AM"] = True
    state["final_answer"]["PM"] = False
    return state


def add_PM(state: AgentState) -> AgentState:
    state["final_answer"]["AM"] = False
    state["final_answer"]["PM"] = True
    return state


def is_AM_or_PM(state: AgentState) -> Literal["is_AM", "is_PM"]:
    """Enrich output with AM/PM markers"""

    # look at the original datetime object version of the time data
    if state["time_data"].hour < 12:
        return "is_AM"
    else:
        return "is_PM"


def main():
    # Create the graph
    workflow = StateGraph(AgentState)

    # Add nodes to graph
    workflow.add_node("get_time", get_time_node)
    workflow.add_node("get_month_name", get_month_name_node)
    workflow.add_node("format", format_response)
    workflow.add_node("is_AM", add_AM)
    workflow.add_node("is_PM", add_PM)

    # Define edges
    workflow.add_edge(START, "get_time")
    workflow.add_edge("get_time", "get_month_name")
    workflow.add_edge("get_month_name", "format")
    workflow.add_conditional_edges("format", is_AM_or_PM, ["is_AM", "is_PM"])
    workflow.add_edge("is_AM", END)
    workflow.add_edge("is_PM", END)

    # Compile the graph
    app = workflow.compile()

    # Run the workflow
    config = {
        "get_time_prompt": "What is the current time?",
        "timezone": "America/New_York",
        #"timezone": "Asia/Seoul",
    }
    result = app.invoke(config)
    print(result["final_answer"])


if __name__ == "__main__":
    main()
