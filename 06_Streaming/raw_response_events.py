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
    openai_client = client
)

agent = Agent(
    name = "Personal Assistant",
    instructions = "You are a user's personal assistant. Help her/him with their queries in a very light mood. Please no long responses only consice ones.",
    model = model
)

async def main():
    result = Runner.run_streamed(
        agent,
        "I want to gain weight, give me a healthy and budget friendly diet plan which helps in weight gain."
    )

    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end="", flush=True)

asyncio.run(main())


# OUTPUT 👇🏻

# Hey there! Ready to power up your plate? Here’s a super simple, wallet-friendly plan to help you gain healthy weight:

# *   **Breakfast:** Oatmeal with milk, a banana, and a spoonful of peanut butter. Yum!
# *   **Mid-morning:** Grab an apple or a small handful of nuts.
# *   **Lunch:** A good serving of rice/roti, dal (lentils), a veggie, and maybe some chicken/eggs if you're up for it.
# *   **Evening Snack:** Hard-boiled eggs or a quick smoothie (milk + banana + peanut butter).
# *   **Dinner:** Similar to lunch – focus on carbs, protein, and veggies.
# *   **Before Bed:** A warm glass of milk.

# **Key things:** Eat often, don't skip meals, and hydrate! You've got this! ✨