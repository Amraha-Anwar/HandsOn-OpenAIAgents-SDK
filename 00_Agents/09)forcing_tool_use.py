from agents import (
    Agent,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    Runner,
    set_tracing_disabled,
    ModelSettings,
    function_tool,
    RunContextWrapper
)

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


agent = Agent[info](
    name = "assistant",
    instructions = "Help user with their query. Use tool [myInfo] if user asks anything about his/herself.",
    model = model,
    tools = [myInfo],
    model_settings = ModelSettings(tool_choice = "myInfo")
)

context = info(name = "Amraha", gender = "Female")
result = Runner.run_sync(
    agent,
    "What do you know about me?",
    context = context
    )

print(result.final_output)


# OUTPUT (tool_choice = "none") 👇🏻

# I cannot access any personal information about you.
# My purpose is to provide helpful and harmless information, and I am not designed to store 
# or recall private data about individuals.



# OUTPUT (tool_choice = "required") 👇🏻
# Your name is Amraha and you are female.



# OUTPUT (tool_choice = "auto") 👇🏻
# I know that your name is Amraha and you are female.