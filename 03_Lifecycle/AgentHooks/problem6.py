# TASK:
# You have two agents: a Story Generator and a Story Reviewer that checks the story’s quality. Use AgentHooks to log the interaction between the two agents.
# Create a custom AgentHooks class for both agents to log when each agent starts, ends, and passes data to the next agent. The Story Reviewer should output a structured review using AgentOutputSchema.


from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, Runner, set_tracing_disabled, AgentHooks, RunContextWrapper, AgentOutputSchema
from dotenv import load_dotenv
from pydantic import BaseModel
from rich import print
import asyncio
import os

load_dotenv()
set_tracing_disabled(True)

gemini_api_key = os.getenv("GEMINI_API_KEY")
url = os.getenv("BASE_URL")
MODEL = 'gemini-2.5-flash'

client = AsyncOpenAI(
    api_key = gemini_api_key,
    base_url = url
)

model = OpenAIChatCompletionsModel(
    model = MODEL,
    openai_client = client
)

class StoryReview(BaseModel):
    quality: str
    feedback: str

class ChainingHooks(AgentHooks):
    def __init__(self, agent_name):
        self.event_counter = 0
        self.agent_name = agent_name

    async def on_start(self, context: RunContextWrapper, agent:Agent)-> None:
        self.event_counter += 1
        print(f"{self.agent_name} {self.event_counter}:\n{agent.name} Agent Started.\n")


    async def on_end(self, context: RunContextWrapper, agent:Agent, output:any) -> None:
        self.event_counter += 1
        print(f"{self.agent_name} {self.event_counter}:\n{agent.name} Agent Ended.\nOUTPUT:{output}")
        
story_agent = Agent(
    name = "Story writer",
    instructions = "Write a short story on 'title' based on user's input.",
    hooks = ChainingHooks(agent_name= "Story Writer"),
    model = model,
)

reviewer_agent = Agent(
    name = "Story Reviewer",
    instructions = "Review the story for quality (Good, Average, Poor) and provide 1 line feedback.",
    model = model,
    output_type = AgentOutputSchema(output_type = StoryReview),
    hooks=ChainingHooks(agent_name="Story Reviewer")
)

async def main():
    story_result = await Runner.run(
        story_agent,
        input = "Write a story on 'an ambitious girl'."
    )
    story = story_result.final_output

    review_result = await Runner.run(
        reviewer_agent,
        f"Review this story: {story}"
    )
    

asyncio.run(main())


# OUTPUT 👇🏻

# Story Writer 1:
# Story writer Agent Started.

# Story Writer 2:
# Story writer Agent Ended.
# OUTPUT: Elara wasn't born into grandeur; her childhood was spent in a compact apartment where the only view was another 
# apartment building................................

# Story Reviewer 1:
# Story Reviewer Agent Started.

# Story Reviewer 2:
# Story Reviewer Agent Ended.
# OUTPUT:
# StoryReview(
#     quality='Good',
#     feedback='A well-crafted and inspiring story of ambition and perseverance, with vivid descriptions and a satisfying 
# conclusion.'