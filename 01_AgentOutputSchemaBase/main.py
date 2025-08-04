from agents import (
    Agent,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    Runner,
    set_tracing_disabled,
    AgentOutputSchemaBase
)
from pydantic import BaseModel
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
    model= 'gemini-2.0-flash',
    openai_client = external_client
)

class BookReview(BaseModel):
    is_goodread : bool
    summary : str
    genre : str

class OutputSchema(AgentOutputSchemaBase):
    def json_schema(self):
        return {
            'is_goodread' : 'True',
            'summary' : 'A gripping Urdu novel centered on Faris Ghazi, a former intelligence officer falsely accused of murdering his family, whose nephew Saadi Yousuf fights relentlessly to prove his innocence. Alongside Zumar Yousuf, a principled district attorney initially convinced of his guilt, they gradually uncover the manipulative schemes of Hashim Kardar, navigating a web of betrayal, justice, and spiritual redemption',
            'genre' : 'Crime-thriller'
        }

    def is_strict_json_schema(self) -> bool:
        return True
    
agent = Agent(
    name = "Novel Review Analyzer",
    instructions = "Analyze the book review and determine if it's a good read, provide a brief summary and identify the genre of the novel.Also try to guess the name f the novel it's written by a famous Urdu Writer.",
    model = model,
    output_type = OutputSchema
)

result = Runner.run_sync(
    agent,
    "I love this novel, the characters the plot even the dialogues were just more than amazing. It's a masterpiece itself."
)

print(result.final_output)
