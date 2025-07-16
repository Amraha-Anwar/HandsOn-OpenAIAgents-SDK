# TASK:
# You want to handle errors gracefully in the Story Generator and log them using AgentHooks.
# Create a custom AgentHooks class that catches and logs any errors during the agent’s execution, including ModelBehaviorError for invalid outputs.



from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled, RunContextWrapper, AgentHooks, Runner
from dotenv import load_dotenv
import asyncio
import os

load_dotenv()
set_tracing_disabled(disabled=True)

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set.")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url=os.getenv("BASE_URL")
)


model = OpenAIChatCompletionsModel(
    model="invalid-model-name",            #intentional error
    openai_client=external_client
)


class ErrorHandlingHooks(AgentHooks):
    def __init__(self, agent_display_name):
        self.event_counter = 0
        self.agent_display_name = agent_display_name

    async def on_start(self, context: RunContextWrapper, agent: Agent) -> None:
        self.event_counter += 1
        print(f"{self.agent_display_name} {self.event_counter}:\n{agent.name} Agent Started.\nUSAGE: {context.usage}\n")

    async def on_end(self, context: RunContextWrapper, agent: Agent, output: any) -> None:
        self.event_counter += 1
        print(f"{self.agent_display_name} {self.event_counter}:\n{agent.name} Agent Ended.\nOutput:\n\n{output}")
    
    async def on_error(self, context: RunContextWrapper, agent: Agent ,error: Exception) -> None:
        self.event_counter += 1
        print(f"{self.agent_display_name} {self.event_counter}:\nError Occurred: {type(error).__name__} - {str(error)}")

agent = Agent(
    name = "Story Writer",
    instructions = "Generate a short story based on the user's given topic.",
    model = model,
    hooks = ErrorHandlingHooks(agent_display_name= "Story Writer")
)

async def main():
    try:
        result = await Runner.run(
            agent,
            "write a story on 'an ambitious girl'."
        )
        print(result.final_output)

    except Exception as e:
        print(f"Main caught Error : {e}")

asyncio.run(main())


# OUTPUT👇🏻
# Story Writer 1:
# Story Writer Agent Started.
# USAGE: Usage(requests=0, input_tokens=0, input_tokens_details=InputTokensDetails(cached_tokens=0), output_tokens=0, output_tokens_details=OutputTokensDetails(reasoning_tokens=0), total_tokens=0)


# Main caught Error : Error code: 404 - [{'error': {'code': 404, 'message': 'models/invalid-model-name is not found for API version v1main, or is not supported for generateContent. Call ListModels to see the list of available models and their supported methods.', 'status': 'NOT_FOUND'}}]