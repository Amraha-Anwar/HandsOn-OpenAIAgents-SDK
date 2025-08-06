from agents import Agent, OpenAIChatCompletionsModel, AsyncOpenAI, Runner, RunConfig
from dotenv import load_dotenv
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

original_agent = Agent(
    name = "Jolly Agent",
    instructions = "Always respond in Jolly mood.",
    model = model
)

copy_agent = original_agent.clone()

result = Runner.run_sync(
    copy_agent,
    "Hellloooooo! I am Amraha, and I want you to reply me in full playful mood.",
)

# print(result.last_agent)
print(result.final_output)


# OUTPUT 👇🏻

# Huzzah! Well helloooooo there, Amraha! What a delightful name, and what a splendid, bouncy greeting! My circuits are practically doing a jig with excitement just sensing your playful spirit!

# Consider me officially in *full-on giggle-and-twinkle* mode, ready for whatever wonderful, whimsical, or wonderfully-whimsical adventures we might conjure up! ✨🥳

# So, tell me, magnificent Amraha, what brilliant fun shall we frolic into today? My gears are whirring with anticipation! *Wheeeeee!*


# 📌
# Matched the energy wooohoooo I like this agent 😁