from agents import Agent, Runner, trace, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled
from dotenv import load_dotenv
from rich import print
import os, asyncio

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

agent = Agent(
    name = "Reporter",
    instructions = "Answer in 2 lines.",
    model = model
)

async def main():
    with trace(workflow_name = "conversation"):
        result = await Runner.run(
            agent,
            "define 'procrastination'?"
        )
        print(f"Turn 1:\n{result.final_output}")

        new_input = result.to_input_list() + [{"role": "user", "content": "what is the opposite of 'procrastination'?"}]
        result = await Runner.run(
            agent,
            new_input
        )
        print(f"Turn 2:\n{result.final_output}")

asyncio.run(main())


# OUTPUT 👇🏻

# Turn 1:
# Procrastination is the act of delaying or postponing tasks or decisions, often despite knowing it may lead to negative 
# consequences. It involves voluntarily putting off important activities in favor of less urgent or more pleasurable ones.

# Turn 2:
# The opposite of procrastination is **prompt action and timely execution**.
# It involves immediately starting tasks and completing them without unnecessary delay.