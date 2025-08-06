from agents import Agent, OpenAIChatCompletionsModel, AsyncOpenAI, Runner, set_tracing_disabled, RunContextWrapper
from dotenv import load_dotenv
from dataclasses import dataclass
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

@dataclass
class UserInfo:
    name : str
    age : int

def dynamic_instruction(context: RunContextWrapper[UserInfo], agent : Agent[UserInfo]) -> str:
    return f"The user's name is {context.context.name} who is {context.context.age} years old."


user_info = UserInfo(name = "Amraha", age = 18)

agent = Agent[UserInfo](
    name = "Assistant",
    instructions = dynamic_instruction,
    model = model
)

result = Runner.run_sync(
    agent,
    "What do you know about me?",
    context = user_info
)

print(result.final_output)


# OUTPUT 👇🏻

# I know your name is **Amraha** and you are **18 years old**.

# That's the only personal information I've been given about you!