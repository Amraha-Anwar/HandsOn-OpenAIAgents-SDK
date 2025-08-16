from agents import (
    Agent,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    set_tracing_disabled,
    input_guardrail,
    InputGuardrailTripwireTriggered,
    GuardrailFunctionOutput,
    TResponseInputItem,
    Runner,
    RunContextWrapper,
    RunConfig
)
from pydantic import BaseModel
from dotenv import load_dotenv
import asyncio, os


load_dotenv()
set_tracing_disabled(True)

gemini_api_key = os.getenv("GEMINI_API_KEY")
MODEL = 'gemini-2.5-flash'
url = os.getenv("BASE_URL")

client = AsyncOpenAI(
    api_key = gemini_api_key,
    base_url = url
)

model = OpenAIChatCompletionsModel(
    model = MODEL,
    openai_client = client
)

config = RunConfig(
    model = model,
    model_provider = client
)

class UrduOutput(BaseModel):
    is_urdu : bool
    reasoning: str

guardrail_agent = Agent(
    name = "Urdu Guardrail Agent",
    instructions = "Check if user asking for response in Urdu.",
    model = model,
    output_type = UrduOutput
)

@input_guardrail
async def urdu_guardrial(
    ctx: RunContextWrapper[None], agent: Agent, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    result = await Runner.run(
        guardrail_agent,
        input,
        context = ctx.context,
        run_config = config
    )
    return GuardrailFunctionOutput(
        output_info = result.final_output,
        tripwire_triggered = result.final_output.is_urdu,
    )

agent = Agent(  
    name="Customer support agent",
    instructions="You are a customer support agent. You help customers with their questions.",
    input_guardrails=[urdu_guardrial],
)

async def main():
    try:
        result = await Runner.run(
            agent,
            # "Write an essay on 'patriotism' in Urdu.",
            "Write a short description of 'Patriotism'.",
            run_config = config
        )
        print("Input Guardrail didn't trip")
        print(result.input_guardrail_results)
    except InputGuardrailTripwireTriggered:
        print("Input guardrail tripped! Input cannot include 'Urdu'")

asyncio.run(main())


