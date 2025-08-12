from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, Runner, ModelSettings, RunConfig
from dotenv import load_dotenv
import os, asyncio

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
MODEL = "gemini-2.5-flash"
url = os.getenv("BASE_URL")


client = AsyncOpenAI(
    api_key = gemini_api_key,
    base_url = url
)

model = OpenAIChatCompletionsModel(
    model = MODEL,
    openai_client = client
)

config = RunConfig(
    model = model,
    model_provider = client,
    tracing_disabled = True,
    model_settings = ModelSettings(
        temperature = 0.5
    )
)

agent = Agent(
    name = "Assistant",
    instructions = "You are Assistant. Help user with their query.",
    model = model
)

result = Runner.run_sync(
    agent,
    "Write a brief description of 'ModelSettings in OpenAI Agents SDK'.",
    run_config = config
)

print(result.final_output)