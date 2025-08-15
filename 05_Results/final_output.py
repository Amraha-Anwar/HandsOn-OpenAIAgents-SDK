from agents import Agent, OpenAIChatCompletionsModel, AsyncOpenAI, Runner, set_tracing_disabled
from dotenv import load_dotenv
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

agent = Agent(
    name = "Haiku Agent",
    instructions = "Always respond in Haiku form.",
    model = model
)

result = Runner.run_sync(
    agent,
    "Hello! This is Amraha. Wish me for my birthday in advance."
)

# print(result.final_output)
print(type(result.final_output))



# OUTPUT 👇🏻

# Future joy awaits,
# Amraha, your day is near,
# Happy birthday wish.
# New year's joy arrives,
# Amraha, future bright,
# Happy wish for you.