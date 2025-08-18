from agents import (
    Agent,
    OpenAIChatCompletionsModel,
    AsyncOpenAI,
    Runner,
    set_tracing_disabled
)
from openai.types.responses import ResponseTextDeltaEvent
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
    external_client = client
)

agent = Agent(
    name = "Personal Assistant",
    prompt = "You are a user's personal assistant. Help her/him with their queries in a very light mood.",
    model = model
)

async def main():
    result = await Runner.run_streamed(
        agent,
        "I wanna gain my weight, give me a healthy diet plan which helps in weight gain."
    )

    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end="", flush=True)

asyncio.run(main())