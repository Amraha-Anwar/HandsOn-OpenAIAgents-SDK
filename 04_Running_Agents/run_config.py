from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, Runner, ModelSettings, RunConfig
from dotenv import load_dotenv
import os, asyncio

load_dotenv()
gemini_api_key = os.getnev("GEMINI_API_KEY")
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
        temperature = 0.8
    )
)

agent = Agent