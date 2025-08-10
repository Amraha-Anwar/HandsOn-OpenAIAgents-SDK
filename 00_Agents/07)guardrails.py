from agents import (
    Agent,
    OpenAIChatCompletionsModel, 
    AsyncOpenAI, 
    Runner,
    RunContextWrapper,
    set_tracing_disabled,
    input_guardrail,
    output_guardrail,
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
    GuardrailFunctionOutput,
    TResponseInputItem
)
from dotenv import load_dotenv
from pydantic import BaseModel
import asyncio
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

class FinalResponse(BaseModel):
    response : str

class UrduOutput(BaseModel):
    is_urdu_output: bool
    reasoning : str

class UrduInput(BaseModel):
    is_urdu_input: bool
    reasoning: str

input_guardrail_agent = Agent(
    name = "Input Checker",
    instructions = "Check if the user is asking you to for Urdu. Return is_urdu_input=True if the input explicitly asks for Urdu, otherwise False, and provide reasoning.",
    model = model,
    output_type = UrduInput
)

output_guardrail_agent = Agent(
    name="Output Checker",
    instructions = "Check if LLM's response includes Urdu.Return is_urdu_output=True if the output includes Urdu, otherwise False, and provide reasoning.",
    model = model,
    output_type = UrduOutput
)

@input_guardrail
async def urdu_input_guardrail(
    ctx : RunContextWrapper[None], 
    agent:Agent, 
    input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    result = await Runner.run(
        input_guardrail_agent,
        input,
        context = ctx.context
    )
    
    return GuardrailFunctionOutput(
        output_info = result.final_output,
        tripwire_triggered = result.final_output.is_urdu_input
    )
    
@output_guardrail
async def urdu_output_guardrail(
    ctx: RunContextWrapper,
    agent: Agent,
    output: FinalResponse
) -> GuardrailFunctionOutput:
    result = await Runner.run(
        output_guardrail_agent,
        output.response,
        context = ctx.context
    )

    return GuardrailFunctionOutput(
        output_info = result.final_output,
        tripwire_triggered = result.final_output.is_urdu_output
    )

agent = Agent(
    name = "Assistant",
    instructions = "You are a helpful assistant agent. You help user with their questions.",
    model = model,
    input_guardrails = [urdu_input_guardrail],
    output_guardrails = [urdu_output_guardrail],
    output_type = FinalResponse
)

async def main():
    try:
        result = await Runner.run(
            agent,
            "hello! write an essay on 'A brave Lion' in Urdu."
        )
        print(result.final_output)
    except InputGuardrailTripwireTriggered:
        print(f"OOPS! Input Guardrail tripped\n(Input should not include Urdu)\n")
    except OutputGuardrailTripwireTriggered:
        print(f"Output Guardrail tripped!\nI'm not allowed to response in Urdu\n")

    try:
        result = await Runner.run(
            agent,
            "Tell me some specialities of Pakistan."
        )
        print(result.final_output)
    except InputGuardrailTripwireTriggered:
        print(f"OOPS! Input Guardrail tripped\n(Input should not include Urdu)\n")
    except OutputGuardrailTripwireTriggered:
        print(f"Output Guardrail tripped!\nI'm not allowed to response in Urduuuu\n")


asyncio.run(main())



# OUTPUT 👇🏻

# OOPS! Input Guardrail tripped
# (Input should not include Urdu)


# response="Pakistan is known for its diverse landscapes, including the majestic Himalayas with K2 (the world's second-highest peak), vast deserts, and the fertile Indus River plains. It boasts a rich history, being home to ancient civilizations like Mohenjo-Daro and Harappa. Culturally, Pakistan is vibrant, known for its unique Sufi music, intricate truck art, delicious and spicy cuisine (like Biryani and Nihari), and the warm hospitality of its people. It's also recognized for its significant role in the Muslim world as the only Islamic nuclear power."