# TASK:
# You’re building a grocery app that parses a user’s shopping list into structured items.
# Create an agent that extracts a list of items with quantities from a user’s input.

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

class ShoppingItem(BaseModel):
    item_name : str
    quantity : int

class ShoppingList(BaseModel):
    items :list[ShoppingItem]

agent = Agent(
    name = "Shopping List Parser",
    instructions = "Pull out the name and quantity of Items from the shopping list.",
    model = model,
    output_type = AgentOutputSchema(output_type = ShoppingList)
)

async def main():
    result = await Runner.run(
        agent,
        "I wanna buy 6 Abayas, 30 Strollers, 3 Heels."
    )

    print(result.final_output)

asyncio.run(main())


# OUTPUT👇🏻
# items=[ShoppingItem(item_name='Abayas', quantity=6), ShoppingItem(item_name='Strollers', quantity=30), ShoppingItem(item_name='Heels', quantity=3)]


# MESSY INPUT👉🏻 "I need apples (2), 3 bananas, milk."
# OUTPUT👇🏻
# items=[ShoppingItem(item_name='apples', quantity=2), ShoppingItem(item_name='bananas', quantity=3), ShoppingItem(item_name='milk', quantity=1)]


# MISSING QUANTITY INPUT👉🏻 "Buy apples and bananas"
#  OUTPUT👇🏻
# "items=[ShoppingItem(item_name='apples', quantity=1), ShoppingItem(item_name='bananas', quantity=1)]"