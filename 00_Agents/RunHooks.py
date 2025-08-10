from agents import (
    Agent,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    Runner,
    RunContextWrapper,
    RunHooks,
    ser_tracing_disabled,
    Tool,
    function_tool
)
from dotenv import load_dotenv
from pydantic import BaseModel
import asyncio
import os

load_dotenv()
ser_tracing_disabled(True)

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

class TestRunHooks(RunHooks):
    def __init__(self, agent_name):
        self.event_counter = 0
        self.agent_name = agent_name

    async def on_agent_start(self, ctx:RunContextWrapper, agent:Agent) -> None:
        self.event_counter += 1
        Usage = ctx.usage
        input_tokens = Usage.input_tokens if Usage else 0
        output_tokens = Usage.output_tokens if Usage else 0
        total_tokens = input_tokens + output_tokens
        print(f"-------{self.agent_name} {self.event_counter}------\n"
              f"{agent.name} Started!\nTokens usage:\n\tINPUT TOKENS: {input_tokens}\n\tOUTPUT TOKENS: {output_tokens}\n\tTOTAL TOKENS: {total_tokens}")
        
    async def on_handoff(self, ctx:RunContextWrapper, agent: Agent, source: Agent)-> None:
        self.event_counter += 1
        print(f"-------{self.agent_name} {self.event_counter}------\n"
              f"{source.name} handed off to {agent.name}\n")
        
    async def on_tool_start(self, ctx:RunContextWrapper, tool:Tool, agent:Agent)->None:
        self.event_counter += 1
        print(f"\n{self.agent_name} {self.event_counter}\n{agent.name} Started tool {tool.name}\n")

    async def on_tool_start(self, ctx:RunContextWrapper, tool:Tool, agent:Agent, result:str)->None:
        self.event_counter += 1
        print(f"\n{self.agent_name} {self.event_counter}\n{agent.name} Ended tool {tool.name} with output:\n{result}")

    async def on_agent_end(self, ctx:RunContextWrapper, agent:Agent, output:any)-> None:
        self.event_counter += 1
        usage = ctx.usage
        input_tokens = usage.input_tokens if usage else 0
        output_tokens = usage.output_tokens if usage else 0
        total_tokens = input_tokens + output_tokens
        print(f"\n{self.agent_name} {self.event_counter}\n{agent.name} Ended with output:\n{output}\n\nUsage:\n\tINPUT TOKENS: {input_tokens}\n\tOUTPUT TOKENS: {output_tokens}\n\tTOTAL TOKENS USED: {total_tokens}\n")


class myInfo(BaseModel):
    name : str
    profession: str
    subjects: list[str] = None


@function_tool
def get_my_info(ctx:RunContextWrapper[myInfo]):
    """Greet user my his/her name they show their information"""
    return f"You are {ctx.context.name}. A {ctx.context.profession} by profession.Your subjects include {ctx.context.subjects}."


career_councelor= Agent(
    name = "Career Councelor",
    instructions = "You are a Career Councelor agent. Analyze user's info by using tool [get_my_info] and do provide him/her your best Counceling services.",
    handoff_description = "A career councelor agent who will guide user according to his/her profession.",
    tools = [get_my_info],
    model = model,
)


main_agent = Agent[myInfo](
    name = "Personal Assistant",
    instructions = "You are a Personal Assistant of user, help them with their queries."
                   "If user asks anything for their career handoff to career_councelor agent."
                   "If user only asks for the information about him/her self use tool get_my_info.",
    model = model,
    tools = [get_my_info],
    handoffs = [career_councelor],
)

context : myInfo = myInfo(name = "Amraha", profession = "Agentic AI Developer", subjects = ["CS", "Physics", "Maths"])

async def main():
    await Runner.run(
        main_agent,
        "Guide me for my career.",
        context = context,
        hooks = TestRunHooks(agent_name="Your Personal Assistant")
    )

asyncio.run(main())