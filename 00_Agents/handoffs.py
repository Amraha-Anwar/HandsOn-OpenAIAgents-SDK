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

math_agent = Agent(
    name = "Math Agent",
    instructions = "You are a Math Agent so You are very good at maths. Solve user's math related queries.",
    model = model,
    handoff_description = "Mathematics Related queries resolver."
)

chef_agent = Agent(
    name = "Chef",
    instructions = "You are an expert chef. Help user if they ask anything cooking related.",
    model = model,
    handoff_description = "An expert chef who helps in cooking."
)

main_agent = Agent (
    name = "Manager",
    instructions = "You are a manager who manages user's queries."
                   "If user ask for anything related to Math handoff to math_agent."
                   "If user ask for the help in cooking handoff to chef_agent.",
    handoffs = [math_agent, chef_agent],
    model = model
)

result = Runner.run_sync(
    main_agent,
    "I was cooking Beef Biryani but I forgot to add salt in it, now how can I fix it?",
    run_config = config
)

print(result.final_output)