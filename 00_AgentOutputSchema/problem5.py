# TASK:
# You’re building a meeting scheduler that extracts meeting details, including attendees and their roles.
# Create an agent that extracts a meeting’s title, date, time, and a list of attendees with their names and roles (e.g., Organizer, Participant).



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

class Attendees(BaseModel):
    name : str
    role : str

class MeetingDetails(BaseModel):
    title : str
    date: str
    time: str
    attendees_list : list[Attendees]

agent = Agent(
    name = "Meeting Scheduler Agent",
    instructions = "Extract meeting title, date, time and attendees name with their role from the user Input.",
    model = model,
    output_type = AgentOutputSchema(output_type = MeetingDetails)
)

async def main():
    result = await Runner.run(
        agent,
        "Schedule a team sync on July 12, 2025, at 10 AM with Ali as Organizer, and Sara as Managing Director"
    )

    print(result.final_output)

asyncio.run(main())



# INPUT 👉🏻 "Schedule a team sync on July 12, 2025, at 10 AM with Ali as Organizer, and Sara as Managing Director"
# OUTPUT 👇🏻
# title='team sync' date='July 12, 2025' time='10 AM' attendees_list=[Attendees(name='Ali', role='Organizer'), Attendees(name='Sara', role='Managing Director')]


# INPUT WITH MISSING ROLES👉🏻 "Schedule a meeting on July 15, 2025, at 2 PM with Ahmed and Sana."
# OUTPUT 👇🏻
# title='Meeting Schedule' date='July 15, 2025' time='2 PM' attendees_list=[Attendees(name='Ahmed', role='Attendee'), Attendees(name='Sana', role='Attendee')]