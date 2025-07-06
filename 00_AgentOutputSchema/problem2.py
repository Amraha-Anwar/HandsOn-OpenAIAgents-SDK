# TASK:
# You’re developing a customer feedback system that analyzes user comments and determines sentiment.
#Create an agent that takes a customer review and returns the sentiment (Positive, Negative, Neutral) along with a brief reasoning for the classification.

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

class SentimentAnalysis(BaseModel):
    sentiment: str
    reasoning: str

agent = Agent(
    name = "Sentiment Analizer",
    instructions = "Analize The sentiment of user ['Positive', 'Negative', 'Neutral'] from the given review as an input text and provide a brief reasoning.",
    model = model,
    output_type = AgentOutputSchema(output_type = SentimentAnalysis)
)

async def main():
    result = await Runner.run(
        agent,
        "The product was not that good."
    )

    print(result.final_output)

asyncio.run(main())


# INPUT 👉🏻 "The product was amazing and worked perfectly!"
# OUTPUT👇🏻
# sentiment='Positive' reasoning='The review uses strong positive adjectives such as "amazing" and states that the product "worked perfectly," indicating high satisfaction.'


# INPUT 👉🏻 "The product was just okay."
# OUTPUT👇🏻
# sentiment='Neutral' reasoning='The phrase "just okay" indicates a lack of strong positive or negative feelings, suggesting an average or acceptable experience without enthusiasm.'


# INPUT 👉🏻 "The product was not that good."
# OUTPUT👇🏻
# sentiment='Negative' reasoning="The user explicitly states the product was 'not that good,' indicating dissatisfaction with its quality."