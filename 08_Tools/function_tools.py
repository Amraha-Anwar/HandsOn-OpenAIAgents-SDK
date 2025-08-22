from agents import (
    Agent,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    Runner,
    FunctionTool,
    function_tool,
    RunContextWrapper,
    set_tracing_disabled
)
from dotenv import load_dotenv
from pydantic import BaseModel
from rich import print
import json, asyncio, os

load_dotenv()
set_tracing_disabled(True)

gemini_api_key = os.getenv("GEMINI_API_KEY")
url = os.getenv("BASE_URL")
MODEL = 'gemini-2.5-flash'

client = AsyncOpenAI(
    api_key = gemini_api_key,
    base_url = url
)

model = OpenAIChatCompletionsModel(
    model = MODEL,
    openai_client = client
)

class Location(BaseModel):
    latitude : float
    longitude : float

@function_tool
async def fetch_weather(location : Location)-> str:
    """Returns the weather of the given location
    
    Args:
        location: The location to fetch the weather for.
    """
    return "Sunny"

@function_tool(name_override = "fetch_data")
def read_file(ctx: RunContextWrapper[any], path: str, directory: str | None = None) -> str:
    """Read the contents of a file.

    Args:
        path: The path to the file to read.
        directory: The directory to read the file from.
    """
    
    return "<file contents>"


agent = Agent(
    name = "Assistant",
    instructions = "Help user with their query. Use Tools [fetch_weather, read_file] if needed.",
    model = model,
    tools = [fetch_weather, read_file]
)

async def main():
    result = await Runner.run(
        agent,
        "What is the weather at latitude 25.2 and longitude 42.0?"
    )

    print(f"\nYour Input: '{result.input}'")
    print(f"Agent's Output: '{result.final_output}'")

for tool in agent.tools:
    if isinstance(tool, FunctionTool):
        print(f"\n\nTool Name:{tool.name}")
        print(f"Tool Description: {tool.description}")
        print(f"\n\tTool's Schema in json format:\n{json.dumps(tool.params_json_schema, indent= 2)}\n")
        print()


asyncio.run(main())


# OUTPUT 👇🏻

# Tool Name:fetch_weather
# Tool Description: Returns the weather of the given location

#               Tool's Schema in json format:
# {
#   "$defs": {
#     "Location": {
#       "properties": {
#         "latitude": {
#           "title": "Latitude",
#           "type": "number"
#         },
#         "longitude": {
#           "title": "Longitude",
#           "type": "number"
#         }
#       },
#       "required": [
#         "latitude",
#         "longitude"
#       ],
#       "title": "Location",
#       "type": "object",
#       "additionalProperties": false
#     }
#   },
#   "properties": {
#     "location": {
#       "description": "The location to fetch the weather for.",
#       "properties": {
#         "latitude": {
#           "title": "Latitude",
#           "type": "number"
#         },
#         "longitude": {
#           "title": "Longitude",
#           "type": "number"
#         }
#       },
#       "required": [
#         "latitude",
#         "longitude"
#       ],
#       "title": "Location",
#       "type": "object",
#       "additionalProperties": false
#     }
#   },
#   "required": [
#     "location"
#   ],
#   "title": "fetch_weather_args",
#   "type": "object",
#   "additionalProperties": false
# }




# Tool Name:fetch_data
# Tool Description: Read the contents of a file.
#                 Tool's Schema in json format:
# {
#   "properties": {
#     "path": {
#       "description": "The path to the file to read.",
#       "title": "Path",
#       "type": "string"
#     },
#     "directory": {
#       "anyOf": [
#         {
#           "type": "string"
#         },
#         {
#           "type": "null"
#         }
#       ],
#       "description": "The directory to read the file from.",
#       "title": "Directory"
#     }
#   },
#   "required": [
#     "path",
#     "directory"
#   ],
#   "title": "fetch_data_args",
#   "type": "object",
#   "additionalProperties": false
# }



# Your Input: 'What is the weather at latitude 25.2 and longitude 42.0?'
# Agent's Output: 'The weather at latitude 25.2 and longitude 42.0 is sunny.'