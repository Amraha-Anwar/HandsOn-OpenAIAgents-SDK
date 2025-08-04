# TASK:
# You want to measure how long the Story Generator agent takes to process a request to optimize performance.
# Create a custom AgentHooks class that records the start and end times of the agent’s execution and calculates the duration.

from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, Runner, RunContextWrapper, AgentHooks, set_tracing_disabled
from dotenv import load_dotenv
import asyncio
from datetime import datetime
import os

load_dotenv()
set_tracing_disabled(disabled = True)

client = AsyncOpenAI(
    api_key = os.getenv("GEMINI_API_KEY"),
    base_url = os.getenv("BASE_URL")
)

model = OpenAIChatCompletionsModel(
    model = 'gemini-2.5-flash',
    openai_client = client
)

class TestAgentHooks(AgentHooks):
    def __init__(self, agent_name):
        self.event_counter = 0
        self.agent_name = agent_name
        self.start_time = None

    async def on_start(self, context: RunContextWrapper, agent:Agent) -> None:
        self.event_counter += 1
        self.start_time = datetime.now()
        usage = context.usage
        prompt_tokens = usage.input_tokens if usage else 0
        completion_tokens = usage.output_tokens if usage else 0
        total_tokens = prompt_tokens + completion_tokens
        print(f"{self.agent_name} {self.event_counter}:\n{agent.name} Agent started at {self.start_time}.\n"
              f"PROMPT TOKENS: {prompt_tokens},\nCOMPLETION TOKENS: {completion_tokens},\nTOTAL TOKENS: {total_tokens}\n")
        

    async def on_end(self, context:RunContextWrapper, agent: Agent, output: any) -> None:
        self.event_counter += 1
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        usage = context.usage
        prompt_tokens = usage.input_tokens if usage else 0
        completion_tokens = usage.output_tokens if usage else 0
        total_tokens = prompt_tokens + completion_tokens

        print(f"{self.agent_name} {self.event_counter}:\n{agent.name} Agent Ended after {duration:.2f} seconds.\n"
              f"PROMPT TOKENS: {prompt_tokens},\nCOMPLETION TOKENS: {completion_tokens},\nTOTAL TOKENS: {total_tokens}\n"
              f"\nSTORY:\n\n{output}")
        
            
agent = Agent(
    name = "Story Maker",
    instructions = "Make a short story on the topic given in Input text.",
    hooks = TestAgentHooks(agent_name= "Story Maker"),
    model = model
)

async def main():
    result = await Runner.run(
        agent,
        "Write a short story on 'A haunted night'. "
    )

    print(result.final_output)

asyncio.run(main())
print("\n\n\t\t------------------The End----------------")


# OUTPUT 👇🏻

# Story Maker 1:
# Story Maker Agent started at 2025-07-11 20:20:10.699188.
# PROMPT TOKENS: 0,
# COMPLETION TOKENS: 0,
# TOTAL TOKENS: 0

# Story Maker 2:
# Story Maker Agent Ended after 17.15 seconds.
# PROMPT TOKENS: 25,
# COMPLETION TOKENS: 898,
# TOTAL TOKENS: 923

# STORY:

# The moon, a skeletal claw against the bruised twilight, cast long, distorted shadows across the overgrown grounds of Blackwood Manor. Elara shivered, not just from the biting November wind, but from the foolish dare that had brought her to this desolate place. Her friends, safe in their warm beds, had bet her twenty dollars she couldn't spend a single night inside the infamous, long-abandoned mansion. Foolish pride, Elara thought, clutching her trembling flashlight.
#  --------------------------------------- Story Continued