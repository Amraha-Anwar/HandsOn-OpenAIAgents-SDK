from agents import Agent, OpenAIChatCompletionsModel, AsyncOpenAI, Runner, RunConfig, function_tool, RunContextWrapper
from dotenv import load_dotenv
from dataclasses import dataclass
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

@dataclass
class UserInfo:
    username : str
    userid : int
    gender : str
    is_fresher : bool

@function_tool
def get_user_info(wrapper : RunContextWrapper[UserInfo]) -> str:
    '''Returns the username, userid, gender and is_fresher information of the user.'''
    return f"\nUserName: {wrapper.context.username}\nUserID: {wrapper.context.userid}\nGender: {wrapper.context.gender}\nIs Fresher : {wrapper.context.is_fresher}\n"

user_info = UserInfo(username = "Amraha", userid = 2301, gender = "Female",  is_fresher = True) 

agent = Agent[user_info](
    name = "Assistant",
    instructions = "Help user with their query.",
    tools = [get_user_info],
    model = model
)

result = Runner.run_sync(
    agent,
    "Tell me the details of the user.",
    context = user_info
)

print(result.final_output)


# OUTPUT 👇🏻

# Here are the details of the user:
# UserName: Amraha
# UserID: 2301
# Gender: Female
# Is Fresher : True