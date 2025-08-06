# TASK:
# You’re building a to-do list app, and you want an agent to extract task details from user input.
# Create an agent that takes a user’s description of a task and extracts the task name, priority (High, Medium, Low), and due date into a structured format.

from agents import Agent, OpenAIChatCompletionsModel, AsyncOpenAI, Runner, AgentOutputSchema, set_tracing_disabled
from dotenv import load_dotenv
from pydantic import BaseModel
import asyncio
import os

load_dotenv()
set_tracing_disabled(disabled = True)

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

class TaskDetails(BaseModel):
    task_name: str
    priority: str
    due_date: str

agent = Agent(
    name = "Task Extractor",
    instructions = "Extract task name, priority level (High, Medium, Low) and due-date from the input text",
    model = model,
    output_type = AgentOutputSchema(output_type = TaskDetails)
)

async def main():
    result = await Runner.run(
        agent,
        input = "Finish project report by July 10, 2025, with high priority."
    )

    print(result.final_output)

asyncio.run(main())

# OUTPUT 👇🏻

# task_name='Finish project report' priority='High' due_date='July 10, 2025'