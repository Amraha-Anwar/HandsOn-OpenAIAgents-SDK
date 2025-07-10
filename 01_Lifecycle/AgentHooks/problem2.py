# TASK:
# You want to monitor the token usage of your Story Generator agent to track costs and optimize performance.
#Create a custom AgentHooks class that logs the token usage (prompt and completion tokens) at the end of the agent’s execution. Include the total tokens used in the output.


from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, Runner, RunContextWrapper, AgentHooks, set_tracing_disabled
from dotenv import load_dotenv
import asyncio
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

    async def on_start(self, context: RunContextWrapper, agent:Agent) -> None:
        self.event_counter += 1
        usage = context.usage
        prompt_tokens = usage.input_tokens if usage else 0
        completion_tokens = usage.output_tokens if usage else 0
        total_tokens = prompt_tokens + completion_tokens
        print(f"{self.agent_name} {self.event_counter}:\n{agent.name} Agent started!\n"
              f"PROMPT TOKENS: {prompt_tokens},\nCOMPLETION TOKENS: {completion_tokens},\nTOTAL TOKENS: {total_tokens}\n")

    async def on_end(self, context:RunContextWrapper, agent: Agent, output: any) -> None:
        self.event_counter += 1
        usage = context.usage
        prompt_tokens = usage.input_tokens if usage else 0
        completion_tokens = usage.output_tokens if usage else 0
        total_tokens = prompt_tokens + completion_tokens

        print(f"{self.agent_name} {self.event_counter}:\n{agent.name} Agent Ended.\n"
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

# Story Maker Agent started!
# PROMPT TOKENS: 0,
# COMPLETION TOKENS: 0,
# TOTAL TOKENS: 0

# Story Maker 2:
# Story Maker Agent Ended.
# PROMPT TOKENS: 25,
# COMPLETION TOKENS: 565,
# TOTAL TOKENS: 590
# STORY:

# The old cabin sat nestled deep within the whispering pines, a relic from a time when silence was a luxury, not a burden. Liam had inherited it from his great-aunt, a woman whispered to have been "eccentric." Tonight was his first night alone there, a test of his urban nerves against rural isolation.
# ------------------------ story continued...