from agents import(
    Agent,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    Runner,
    RunContextWrapper,
    RunHooks,
    RunConfig
)
from dotenv import load_dotenv
import asyncio
import os

load_dotenv()
# set_tracing_disabled(disabled=True)

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url=os.getenv("BASE_URL"),
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=external_client
)

config = RunConfig(
    model = model,
    model_provider = external_client,
    tracing_disabled = True
)

class HookWriter(RunHooks):
    def __init__(self, run_name):
        self.event_counter = 0
        self.run_name = run_name

    async def on_agent_start(self, ctx: RunContextWrapper, agent: Agent)-> None:
        Usage = ctx.usage
        input_tokens = Usage.input_tokens if Usage else 0
        output_tokens = Usage.output_tokens if Usage else 0
        total_tokens = input_tokens + output_tokens
        self.event_counter += 1
        print(f"{self.run_name} {self.event_counter}:\n{agent.name} Started!\nTokens Used:\n\tPROMPT TOKENS: {input_tokens}"
              f"\n\tRESPONSE TOKENS: {output_tokens}\n\tTOTAL TOKENS USED: {total_tokens}\n")

    async def on_agent_end(self, ctx:RunContextWrapper, agent: Agent, output:any) ->None :
        Usage = ctx.usage
        input_tokens = Usage.input_tokens if Usage else 0
        output_tokens = Usage.output_tokens if Usage else 0
        total_tokens = input_tokens + output_tokens
        self.event_counter += 1
        print(f"{self.run_name} {self.event_counter}:\n{agent.name} Ended!\nTokens Used:\n\tPROMPT TOKENS: {input_tokens}"
              f"\n\tRESPONSE TOKENS: {output_tokens}\n\tTOTAL TOKENS USED: {total_tokens}\nOUTPUT:\n\n{output}")
        
agent = Agent(
    name = "Hook Writer Agent",
    instructions = "You are a hook writer agent for linkedIn posts. Take topic from user's input and write an attractive,playful one-lined hook on that topic",
    model = model
)

async def main():
    await Runner.run(
        agent,
        "I'm gonna post today on linkedIn about 'A graphic designer also needs a portfolio website'.",
        run_config = config,
        hooks = HookWriter("Hook Writer")
    )

asyncio.run(main())


# OUTPUT 👇🏻

# Hook Writer 1:
# Hook Writer Agent Started!
# Tokens Used:
#         PROMPT TOKENS: 0
#         RESPONSE TOKENS: 0
#         TOTAL TOKENS USED: 0

# Hook Writer 2:
# Hook Writer Agent Ended!
# Tokens Used:
#         PROMPT TOKENS: 54
#         RESPONSE TOKENS: 22
#         TOTAL TOKENS USED: 76
# OUTPUT:

# Graphic designers, your Behance is a great start, but your portfolio **website** is the main event!