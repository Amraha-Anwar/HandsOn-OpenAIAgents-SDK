# BASIC EXAMPLE 

from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled, RunContextWrapper, AgentHooks, Runner
from dotenv import load_dotenv
import asyncio
import os

load_dotenv()
set_tracing_disabled(disabled = True)

gemini_api_key = os.getenv("GEMINI_API_KEY")


if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url=os.getenv("BASE_URL"),
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=external_client
)

class TestAgentHooks(AgentHooks):
    def __init__(self, agent_display_name):
        self.event_counter = 0
        self.agent_display_name = agent_display_name

    async def on_start(self, context: RunContextWrapper, agent: Agent) -> None:
        self.event_counter += 1
        print(f"{self.agent_display_name} {self.event_counter}:\n{agent.name} Agent Started. \nUSAGE: {context.usage}\n")

    async def on_end(self, context: RunContextWrapper, agent: Agent, output: any) -> None:
        self.event_counter += 1
        print(f"{self.agent_display_name} {self.event_counter}:\n{agent.name} Agent Ended. \nUSAGE: {context.usage},\n Output:\n\n{output}")

start_agent = Agent(
    name = "Story Generator",
    instructions = "You are a Story Generator specialist. Take Topic from user's input and generate a short story on that topic.",
    hooks = TestAgentHooks(agent_display_name="Story Generator"),
    model = model
)

async def main():
    result = await Runner.run(
        start_agent,
        "Write a story about a brave knight."
    )

    print(result.final_output)

asyncio.run(main())
print("\n\t\t---------------THE END-------------\n")


# OUTPUT👇🏻

# Story Generator 1:
# Story Generator Agent Started. 
# USAGE: Usage(requests=0, input_tokens=0, input_tokens_details=InputTokensDetails(cached_tokens=0), output_tokens=0, output_tokens_details=OutputTokensDetails(reasoning_tokens=0), total_tokens=0)

# Story Generator 2:
# Story Generator Agent Ended. 
# USAGE: Usage(requests=1, input_tokens=33, input_tokens_details=InputTokensDetails(cached_tokens=0), output_tokens=577, output_tokens_details=OutputTokensDetails(reasoning_tokens=0), total_tokens=894),
#  Output:

# Sir Kaelen was not the strongest knight in the realm, nor the wealthiest, but his name echoed loudest when whispers of danger filled the castle halls. For Sir Kaelen possessed an unwavering courage, a quiet fire in his heart that burned brightest when fear threatened to consume others.

#  ---------------------------story--------------------------------------

# Exhausted but victorious, Sir Kaelen emerged from the cave, the morning sun feeling warmer than ever before. He returned to the capital, not with boasts or fanfare, but with the silent gratitude of the relieved villagers and the profound respect of his King. Sir Kaelen proved that true bravery wasn't the absence of fear, but the courage to face it, no matter how terrifying the shadow, for the sake of others.

#                 ---------------THE END-------------
