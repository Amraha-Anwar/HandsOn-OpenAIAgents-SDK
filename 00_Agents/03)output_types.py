from agents import Agent, OpenAIChatCompletionsModel, AsyncOpenAI, Runner, RunConfig
from dotenv import load_dotenv
from pydantic import BaseModel
import os

load_dotenv()

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

config = RunConfig(
    model = model,
    model_provider = external_client,
    tracing_disabled = True
)

class Competition(BaseModel):
    name : str
    date : str
    participants : list[str]

agent = Agent(
    name = "Assistant",
    instructions = "Extract competition name, date (when it's going to happen) and list of participants if given from the given statement.",
    model = model,
    output_type = Competition
)

result = Runner.run_sync(
    agent,
    "Reading Competition is going to happen on 25th August 2025. Current participants include Amraha, Minha, Raahib.",
    run_config = config
)

print(result.final_output)


# OUTPUT 👇🏻
# name='Reading Competition' date='25th August 2025' participants=['Amraha', 'Minha', 'Raahib']