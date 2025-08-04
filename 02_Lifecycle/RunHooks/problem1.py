from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled, RunContextWrapper, RunHooks, Runner
from dotenv import load_dotenv
import asyncio
import os

load_dotenv()
set_tracing_disabled(disabled=True)

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


class SummarizeRunHooks(RunHooks):
    def __init__(self, run_name):
        self.event_counter = 0
        self.run_name = run_name

    async def on_agent_start(self, context: RunContextWrapper, start_agent: Agent) -> None:
        self.event_counter += 1
        print(f"{self.run_name} {self.event_counter}:\n{start_agent.name} started Run!\nUSAGE: {context.usage}\n")

    async def on_agent_end(self, context: RunContextWrapper, start_agent: Agent, output: any) -> None:
        self.event_counter += 1
        print(f"{self.run_name} {self.event_counter}:\n{start_agent.name} Ended Run\nUSAGE: {context.usage}\nOUTPUT:\n\n{output}")


start_agent = Agent(
    name="Text Summarizer",
    instructions="You are a Text Summarizer specialist. Summarize the user's input text into a short summary.",
    model=model
)

async def main():
    await Runner.run(
        start_agent,
        "This is a story about a brave knight who saved a village.",
        hooks=SummarizeRunHooks(run_name="Text Summarization")
    )
    print("\n\t\t---------------THE END-------------\n")

if __name__ == "__main__":
    asyncio.run(main())