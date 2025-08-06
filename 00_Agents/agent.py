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

agent = Agent(
    name = "Haiku Agent",
    instructions = "Always respond in Haiku form.",
    model = model
)

result = Runner.run_sync(
    agent,
    "Hello! This is Amraha. Write something beautiful on me.",
    run_config = config
)

print(result)


# OUTPUT 👇🏻
# RunResult:
# - Last agent: Agent(name="Haiku Agent", ...)
# - Final output (str):
#     Amraha's soft glow,
#     Like petals in morning dew,
#     Pure beauty you show.
# - 1 new item(s)
# - 1 raw response(s)
# - 0 input guardrail result(s)
# - 0 output guardrail result(s)
# (See `RunResult` for more details)