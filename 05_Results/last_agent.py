from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled, Runner
from dotenv import load_dotenv
import asyncio, os

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

dev_agent = Agent(
    name = "Web Developer Agent",
    instructions = "You are a Web Developer Agent. Help user with their web development related queries.Answer concisely, not too long",
    model = model,
    handoff_description = 'Web Developer Agent who is an expert web developer.'
)


doctor_agent = Agent(
    name = "Doctor Smith",
    instructions = "You are Doctor Smith. You deal with the user as a Doctor.Answer concisely, not too long",
    model = model,
    handoff_description = 'Doctor Agent, who will deal user as a doctor'
)

triage_agent = Agent(
    name = "Triage Agent",
    instructions = "You are the main agent who will help user with their queries."
     "If user ask for some medicine or something medical related handoff to [doctor_agent]. "
     "If user ask for something related to web development handoff to [dev_agent].",
    model = model,
    handoffs = [dev_agent, doctor_agent]
)

async def main():
    result = await Runner.run(
        triage_agent,
        # "what is the complete roadmap of web development?"
        "what is the capital of Pakistan?"
    )

    print(result.final_output)
    print(result.last_agent.name)

asyncio.run(main())


# OUTPUT 👇🏻

# The capital of Pakistan is Islamabad.
# Triage Agent

