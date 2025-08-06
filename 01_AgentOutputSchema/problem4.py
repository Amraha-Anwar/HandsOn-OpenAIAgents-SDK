# TASK:
# You’re creating a weather app that extracts structured forecast data from a description.
# Create an agent that extracts the city, temperature (in Celsius), and weather condition from a weather description.

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

class WeatherForecast(BaseModel):
    city: str
    temperature : float
    condition : str


agent  = Agent(
    name ="Weather Forecaster",
    instructions = "Ectract city, temperature in Celcius and weather condition from the input.",
    model = model,
    output_type=AgentOutputSchema(output_type=WeatherForecast)
)

async def main():
    result = await Runner.run(
        agent,
        "It's cloudy in Karachi with a temperature of 25 degrees Celsius."
    )
    print(result.final_output)

asyncio.run(main())


# OUTPUT👇🏻
# city='Karachi' temperature=25.0 condition='cloudy'