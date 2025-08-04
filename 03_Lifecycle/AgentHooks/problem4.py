# TASK:
# You want the Story Generator to produce structured output (e.g., story title and content) and use AgentHooks to log the structured output in a specific format.

# Create an agent that generates a story with a title and content, using AgentOutputSchema for structured output, and a custom AgentHooks class to log the title and word count of the story.


from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, Runner, set_tracing_disabled, AgentHooks, RunContextWrapper, AgentOutputSchema
from dotenv import load_dotenv
from pydantic import BaseModel
from rich import print
import asyncio
import os

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

class Story(BaseModel):
    title: str
    content: str

class StoryHooks(AgentHooks):
    def __init__(self, agent_name):
        self.event_counter = 0
        self.agent_name = agent_name

    async def on_start(self, context: RunContextWrapper, agent:Agent)-> None:
        self.event_counter += 1
        usage = context.usage
        prompt_tokens = usage.input_tokens if usage else 0
        completion_tokens = usage.output_tokens if usage else 0
        total_tokens = prompt_tokens + completion_tokens
        print(f"{self.agent_name} {self.event_counter}:\n{agent.name} Agent Started.\nPROMPT TOKENS: {prompt_tokens}\nCOMPLETION TOKENS: {completion_tokens}\nTOTAL TOKENS: {total_tokens}\n")


    async def on_end(self, context: RunContextWrapper, agent:Agent, output:any) -> None:
        self.event_counter += 1
        word_count = len(output.content.split()) if hasattr(output, 'content') else 0
        title = output.title if hasattr(output, 'title') else 'Unknown'
        usage = context.usage
        prompt_tokens = usage.input_tokens if usage else 0
        completion_tokens = usage.output_tokens if usage else 0
        total_tokens = prompt_tokens + completion_tokens
        print(f"{self.agent_name} {self.event_counter}:\n{agent.name} Agent Ended."
              f"\nPROMPT TOKENS: {prompt_tokens}\nCOMPLETION TOKENS: {completion_tokens}\nTOTAL TOKENS: {total_tokens}\n"
              f"TITLE: {title}\nWORD COUNT: {word_count}\nOUTPUT:")
        
agent = Agent(
    name = "Story writer",
    instructions = "Write a short story on 'title' based on user's input.",
    hooks = StoryHooks(agent_name= "Story Writer"),
    model = model,
    output_type = AgentOutputSchema(output_type = Story)
)

async def main():
    result = await Runner.run(
        agent,
        "Write a story on 'an ambitious girl'."
    )
    print(result.final_output)

asyncio.run(main())

# OUTPUT 👇🏻

# Story Writer 1:
# Story writer Agent Started.
# PROMPT TOKENS: 0
# COMPLETION TOKENS: 0
# TOTAL TOKENS: 0

# Story Writer 2:
# Story writer Agent Ended.
# PROMPT TOKENS: 26
# COMPLETION TOKENS: 415
# TOTAL TOKENS: 441
# TITLE: An Ambitious Girl
# WORD COUNT: 324
# OUTPUT:

# Story(
#     title='An Ambitious Girl',
#     content='Maya lived in the quiet village of Eldoria, a place where the tallest structures were ancient oak trees and
# the loudest sounds were the bleating of sheep. While other children dreamed of simpler lives within the village borders,
# Maya\'s gaze always stretched beyond the distant mountains, her mind filled with blueprints of towering bridges and     
# intricate machinery. Her ambition was a blazing star in the dim sky of local expectations. Resources were scarce. The   
# village school had few books and no science lab, but Maya made do. She devoured every dusty textbook, drew complex      
# diagrams on discarded parchment, and spent hours observing the mechanics of the old water mill, sketching its gears and 
# levers. Her parents, though supportive, often worried about the practicality of her grand dreams. "Engineering is for   
# the city, child," her father would say gently, "not for Eldoria." Yet, Maya was undeterred. She sought out information  
# from visiting merchants, corresponded with distant relatives who lived in larger towns, and even taught herself basic   
# coding from a worn-out manual she found in a forgotten attic trunk. Her evenings were spent under the moonlight, not    
# playing, but studying, calculating, and designing. When the time came for university applications, Maya faced a new     
# hurdle: funding. Scholarships were competitive, and her village had little to offer. But her ambition had made her      
# resourceful. She presented a portfolio of her designs, including a self-designed, sustainable irrigation system that    
# could vastly improve Eldoria\'s crops. Her passion and ingenuity shone through, earning her a full scholarship to the   
# nation\'s top engineering university. Leaving Eldoria, Maya carried not just her modest belongings, but the weight of   
# her village\'s hopes and the fire of her own relentless ambition. She knew the path ahead would be challenging, but for 
# Maya, challenges were simply more problems to engineer solutions for. Her journey had just begun, a testament to the    
# boundless power of a girl who dared to dream bigger than her world.'
# )