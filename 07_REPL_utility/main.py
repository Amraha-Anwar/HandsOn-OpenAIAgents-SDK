from agents import (
    Agent,
    OpenAIChatCompletionsModel,
    AsyncOpenAI,
    Runner,
    set_tracing_disabled,
    run_demo_loop
)
import asyncio, os
from dotenv import load_dotenv

load_dotenv()
set_tracing_disabled(True)

gemini_api_key = os.getenv("GEMINI_API_KEY")
MODEL = 'gemini-2.5-flash'
url = os.getenv("BASE_URL")

client = AsyncOpenAI(
    api_key = gemini_api_key,
    base_url = url
)

model = OpenAIChatCompletionsModel(
    model = MODEL,
    openai_client = client
)

async def main() -> None:
    agent = Agent(name="Assistant", instructions="You are a helpful assistant.")
    await run_demo_loop(agent)        #not working will chekc later!

if __name__ == "__main__":
    asyncio.run(main())