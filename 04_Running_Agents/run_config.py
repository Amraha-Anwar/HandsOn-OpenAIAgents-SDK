from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, Runner, ModelSettings, RunConfig
from dotenv import load_dotenv
import os

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
MODEL = "gemini-2.5-flash"
url = os.getenv("BASE_URL")


client = AsyncOpenAI(
    api_key = gemini_api_key,
    base_url = url
)

model = OpenAIChatCompletionsModel(
    model = MODEL,
    openai_client = client
)

setting = ModelSettings(
        temperature=0.5,            # creativity/randomness level
        max_tokens=3000,            # output length
        top_p=0.9,                  # probability cutoff
        top_k=40,                   # fixed shortlist (keep top k tokens only)
        # frequency_penalty=0.5,    # reduce repeated words (GEMINI doesn't support)
        # presence_penalty=0.2,     # new topics (GEMINI doesn't support)
        # tool_choice="auto",       # control tool usage
    )

config = RunConfig(
    model = model,
    model_provider = client,
    tracing_disabled = True,
    model_settings = setting
)

agent = Agent(
    name = "Assistant",
    instructions = "You are Assistant. Help user with their query.",
    model = model
)

result = Runner.run_sync(
    agent,
    "Write a brief description of 'ModelSettings in OpenAI Agents SDK'.",
    run_config = config
)

print(result.final_output)



# OUTPUT 👇🏻

# **ModelSettings in the OpenAI Agents SDK** is a configuration object that allows you to define and control the behavior of the Large Language Model (LLM) an agent or assistant will use.

# It enables you to:

# *   **Specify the LLM:** Choose the exact model (e.g., `gpt-4`, `gpt-3.5-turbo`).
# *   **Tune Parameters:** Adjust settings like `temperature` (creativity/randomness), `top_p` (diversity), and `max_tokens` (response length).

# Essentially, it's how you fine-tune the "brain" of your agent, ensuring it behaves optimally for its intended purpose within the SDK's framework.