from agents import (
    Agent,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    Runner,
    set_tracing_disabled,
    ModelSettings,
    function_tool,
    RunContextWrapper,
)
from agents.agent import StopAtTools
from dotenv import load_dotenv
from pydantic import BaseModel
import os

load_dotenv()
set_tracing_disabled(True)

gemini_api_key = os.getenv("GEMINI_API_KEY")
url = os.getenv("BASE_URL")

external_client = AsyncOpenAI(
    api_key = gemini_api_key,
    base_url = url
)

model = OpenAIChatCompletionsModel(
    model= 'gemini-2.5-flash',
    openai_client = external_client
)

class info(BaseModel):
    name : str
    gender: str

@function_tool
def myInfo(ctx:RunContextWrapper[info]) ->str:
    """Return name and gender of the user"""
    return f"Name: {ctx.context.name}\nGender: {ctx.context.gender}\n"

@function_tool
def sum_numbers(a: int, b: int) -> int:
    """Adds two numbers."""
    return a + b

agent = Agent[info](
    name = "assistant",
    instructions = "Help user with their query. Use tool [myInfo] if user asks anything about his/herself.",
    model = model,
    tools = [myInfo, sum_numbers],
    model_settings = ModelSettings(tool_choice = "auto"),
    tool_use_behavior = "stop_on_first_tool"
    # tool_use_behavior = StopAtTools(stop_at_tool_names=["myInfo"])
)

context = info(name = "Amraha", gender = "Female")
result = Runner.run_sync(
    agent,
    "Add 53 and 64 then tell me What do you know about me?",
    context = context
    )

print(result.final_output)


# OUTPUT (without setting StopAtTools property) 👇🏻
# The sum of 53 and 64 is 117. I know that your name is Amraha and you are female.


# OUTPUT (with setting StopAtTools(stop_at_tool_names=["sum_numbers"]) 👇🏻
# 117


# OUTPUT (with setting StopAtTools(stop_at_tool_names=["myInfo"]) 👇🏻
# Name: Amraha
# Gender: Female


# OUTPUT by setting (tool_use_behavior = "stop_on_first_tool") 👇🏻
# 117