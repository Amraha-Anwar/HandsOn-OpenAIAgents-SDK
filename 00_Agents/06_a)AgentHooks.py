from agents import Agent, OpenAIChatCompletionsModel, AsyncOpenAI, Runner, set_tracing_disabled, AgentHooks, Tool, function_tool, RunContextWrapper
from dotenv import load_dotenv
from pydantic import BaseModel
from rich import print
import asyncio
import os

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

class TestAgentHook(AgentHooks):
    def __init__(self, agent_name):
        self.event_counter = 0
        self.agent_name = agent_name

    async def on_start(self, context: RunContextWrapper, agent:Agent) -> None:
        self.event_counter += 1
        usage = context.usage
        input_tokens = usage.input_tokens if usage else 0
        output_tokens = usage.output_tokens if usage else 0
        total_tokens = input_tokens + output_tokens
        print(f"\n{self.agent_name} {self.event_counter}\n{agent.name} Started!\nUSAGE:\n\tINPUT TOKENS: {input_tokens}\n\tOUTPUT TOKENS: {output_tokens}\n\tTOTAL TOKENS USED: {total_tokens}\n")

    async def on_tool_start(self, context:RunContextWrapper, agent: Agent, tool: Tool)-> None:
        self.event_counter += 1
        print(f"\n{self.agent_name} {self.event_counter}\n{agent.name} Started tool {tool.name}\n")

    async def on_tool_end(self, context:RunContextWrapper, agent: Agent, tool: Tool, result: str)-> None:
        self.event_counter += 1
        print(f"\n{self.agent_name} {self.event_counter}\n{agent.name} ended tool {tool.name} with result:\n{result}")
    
    async def on_handoff(self, context:RunContextWrapper, agent: Agent, source: Agent)-> None:
        self.event_counter += 1
        print(f"\n{self.agent_name} {self.event_counter}\n{source.name} handed off to {agent.name}\n")

    async def on_end(self, context:RunContextWrapper, agent: Agent, output:any) -> None:
        self.event_counter += 1
        usage = context.usage
        input_tokens = usage.input_tokens if usage else 0
        output_tokens = usage.output_tokens if usage else 0
        total_tokens = input_tokens + output_tokens
        print(f"\n{self.agent_name} {self.event_counter}\n{agent.name} Ended with output:\n{output}\n\nUsage:\n\tINPUT TOKENS: {input_tokens}\n\tOUTPUT TOKENS: {output_tokens}\n\tTOTAL TOKENS USED: {total_tokens}\n")

class myInfo(BaseModel):
    name : str
    profession : str

@function_tool
def userInfo(ctx: RunContextWrapper[myInfo])-> str:
    """Returns name and profession of the user"""
    return f"user's name is {ctx.context.name} who is a/an {ctx.context.profession} by profession."


movie_recommender = Agent(
    name = "Movie Recommender Agent",
    instructions = "You are a movie recommender agent. Take genre from user's input statement and suggest him 5 best  movies of that genre.",
    handoff_description = "Recommends movies on an specific genre.",
    model = model,
    hooks = TestAgentHook(agent_name="customer support agent")
)

agent = Agent[myInfo](
    name = "Customer support Agent",
    instructions = "You are a customer support agent who help user with their queries."
                   "Use tool 'userInfo' if user asks for the information about the user"
                   "If user asks for the movie(s) handoff to 'movie_recommender' agent.",
    tools = [userInfo],
    handoffs = [movie_recommender],
    hooks = TestAgentHook(agent_name="customer support Agent"),
    model = model
)

context: myInfo = myInfo(name = "Amraha", profession = "Agentic AI Developer")
async def main():
    await Runner.run(
        agent,
        # "What do you know about me?",
        "I'm feeling bored. Suggest me some horror movies.",
        context = context
    )

asyncio.run(main())


# OUTPUT 👇🏻

# SCENARIO 1
# ----------------------------------------
# customer support Agent 1
# Customer support Agent Started!
# USAGE:
#         INPUT TOKENS: 0
#         OUTPUT TOKENS: 0
#         TOTAL TOKENS USED: 0


# customer support Agent 2
# Customer support Agent Started tool userInfo


# customer support Agent 3
# Customer support Agent ended tool userInfo with result:
# user's name is Amraha who is a Agentic AI Developer by profession.

# customer support Agent 4
# Customer support Agent Ended with output:
# Your name is Amraha, and you are an Agentic AI Developer by profession.

# Usage:
#         INPUT TOKENS: 312
#         OUTPUT TOKENS: 26
#         TOTAL TOKENS USED: 338


# SCENARIO 2
# --------------------------------------

# customer support Agent 1
# Customer support Agent Started!
# USAGE:
#         INPUT TOKENS: 0
#         OUTPUT TOKENS: 0
#         TOTAL TOKENS USED: 0


# customer support Agent 2
# Customer support Agent handed off to Movie Recommender Agent


# customer support agent 1
# Movie Recommender Agent Started!
# USAGE:
#         INPUT TOKENS: 142
#         OUTPUT TOKENS: 18
#         TOTAL TOKENS USED: 160


# customer support agent 2
# Movie Recommender Agent Ended with output:
# Here are 5 horror movies you might enjoy:

# 1.  **The Exorcist (1973)** - A classic for a reason, known for its chilling atmosphere and groundbreaking scares.        
# 2.  **Hereditary (2018)** - A modern masterpiece of psychological horror, very unsettling and expertly crafted.
# 3.  **Get Out (2017)** - A brilliant and socially relevant horror film that blends suspense with sharp commentary.        
# 4.  **The Conjuring (2013)** - A highly effective supernatural horror film based on a true story, full of jump scares and 
# a creepy atmosphere.
# 5.  **Alien (1979)** - While often considered sci-fi, it's undeniably a horror film in space, pioneering creature design  
# and suspense.

# Usage:
#         INPUT TOKENS: 235
#         OUTPUT TOKENS: 187
#         TOTAL TOKENS USED: 422
