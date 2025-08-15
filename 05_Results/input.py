from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled, Runner
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
    name = "assistant Agent",
    instructions = "Help user with their query. Answer very concisely.",
    model = model,
)


result = Runner.run_sync(
    agent,
    "3 best ways to thank someone"
)

print(f"Output:\n{result.final_output}")
print(f"Your Input:\n{result.input}")


# OUTPUT 👇🏻

# Output:
# 1.  **Directly express thanks (in person or via call/message).**
# 2.  **Send a handwritten thank-you note.**
# 3.  **Perform a reciprocal kind act.**
# Your Input:
# 3 best ways to thank someone