from agents import Agent, Runner, trace, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled
from dotenv import load_dotenv
import asyncio, os


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
    name="Assistant",
    instructions="Reply very concisely.",
    model = model)

async def main():

    thread_id = "thread_123"
    with trace(workflow_name = "conversation", group_id = thread_id):
        result = await Runner.run(
            agent,
            "What City is the most populated in Pakistan?"
        )
    print(f"\nTurn 1: First Input\n{result.final_output}")

    new_input = result.to_input_list() + [{"role": "user", "content": "What are the Names of all seasons?"}]
    result = await Runner.run(
        agent,
        new_input
    )
    print(f"\nTurn 2: New Input\n{result.final_output}")

asyncio.run(main())


# OUTPUT 👇🏻

# Turn 1: First Input
# Karachi

# Turn 2: New Input
# Spring, Summer, Autumn (Fall), Winter