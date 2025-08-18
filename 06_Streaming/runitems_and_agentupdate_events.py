from openai.types.responses import ResponseTextDeltaEvent
from agents import (
    Agent,
    OpenAIChatCompletionsModel,
    AsyncOpenAI,
    Runner,
    set_tracing_disabled,
    function_tool,
    ItemHelpers
)
import asyncio, os
from dotenv import load_dotenv
from rich import print
import random

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

@function_tool
def random_no_of_jokes()-> int:
    return random.randint(1, 10)


agent = Agent(
    name = "Joker",
    instructions = "First call the `how_many_jokes` tool, then tell that many jokes. No lame jokes please.",
    model = model,
    tools = [random_no_of_jokes]
)

async def main():
    result = Runner.run_streamed(
        agent,
        "Hello!"
    )

    print("==== Run Starting =====")

    async for event in result.stream_events():
        # this will ignore the raw response deltas
        if event.type == "raw_response_event":
            continue
        # when the agent updates, print that
        elif event.type == "agent_updated_stream_event":
            print(f"Agent updated: {event.new_agent.name}")
            continue
        # When items are generated, print them
        elif event.type == "run_item_stream_event":
            if event.item.type == "tool_call_item":
                print("---Tool was called")
            elif event.item.type == "tool_call_output_item":
                print(f"---Tool Output: {event.item.output}")
            elif event.item.type == "message_output_item":
                print(f"---Message Output:\n{ItemHelpers.text_message_output(event.item)}")
            else:
                pass

    print("==== Run Complete ====")

asyncio.run(main())


# OUTPUT 👇🏻

# ==== Run Starting =====
# Agent updated: Joker
# ---Tool was called
# ---Tool Output: 3
# ---Message Output:
# Here are 3 jokes for you:

# 1. Why don't scientists trust atoms?
# Because they make up everything!

# 2. What do you call a fake noodle?
# An impasta!

# 3. Why did the scarecrow win an award?
# Because he was outstanding in his field!
# ==== Run Complete ====