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

print(result.final_output)
print(result.raw_responses)


# OUTPUT 👇🏻

# 1.  **Personalized verbal thanks:** Be specific about what they did and how it helped.
# 2.  **Handwritten note/card:** Shows extra effort and genuine appreciation.
# 3.  **Small, thoughtful gesture or favor in return:** Demonstrates gratitude through action.


# [ModelResponse(output=[ResponseOutputMessage(id='__fake_id__', content=[ResponseOutputText(annotations=[], text='1.  **Personalized verbal thanks:** Be specific about what they did and how it helped.\n2.  **Handwritten note/card:** Shows extra effort and genuine appreciation.\n3.  **Small, thoughtful gesture or favor in return:** Demonstrates gratitude through action.', type='output_text', logprobs=None)], role='assistant', status='completed', type='message')], usage=Usage(requests=1, input_tokens=18, input_tokens_details=InputTokensDetails(cached_tokens=0), output_tokens=58, output_tokens_details=OutputTokensDetails(reasoning_tokens=0), total_tokens=168), response_id=None)]