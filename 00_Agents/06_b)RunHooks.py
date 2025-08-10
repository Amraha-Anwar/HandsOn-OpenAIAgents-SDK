from agents import (
    Agent,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    Runner,
    RunContextWrapper,
    RunHooks,
    set_tracing_disabled,
    Tool,
    function_tool
)
from dotenv import load_dotenv
from pydantic import BaseModel
import asyncio
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

class TestRunHooks(RunHooks):
    def __init__(self, agent_name):
        self.event_counter = 0
        self.agent_name = agent_name

    async def on_agent_start(self, context:RunContextWrapper, agent:Agent) -> None:
        self.event_counter += 1
        Usage = context.usage
        input_tokens = Usage.input_tokens if Usage else 0
        output_tokens = Usage.output_tokens if Usage else 0
        total_tokens = input_tokens + output_tokens
        print(f"-------{self.agent_name} {self.event_counter}------\n"
              f"{agent.name} Started!\nTokens usage:\n\tINPUT TOKENS: {input_tokens}\n\tOUTPUT TOKENS: {output_tokens}\n\tTOTAL TOKENS: {total_tokens}")
        
    async def on_tool_start(self, context:RunContextWrapper, tool:Tool, agent:Agent)->None:
        self.event_counter += 1
        print(f"\n{self.agent_name} {self.event_counter}\n{agent.name} Started tool {tool.name}\n")

    async def on_tool_end(self, context:RunContextWrapper, tool:Tool, agent:Agent, result:str)->None:
        self.event_counter += 1
        print(f"\n{self.agent_name} {self.event_counter}\n{agent.name} Ended tool {tool.name} with output:\n{result}")

    async def on_handoff(self, context:RunContextWrapper, from_agent: Agent, to_agent: Agent)-> None:
        self.event_counter += 1
        print(f"-------{self.agent_name} {self.event_counter}------\n"
              f"{from_agent.name} handed off to {to_agent.name}\n")

    async def on_agent_end(self, context:RunContextWrapper, agent:Agent, output:any)-> None:
        self.event_counter += 1
        usage = context.usage
        input_tokens = usage.input_tokens if usage else 0
        output_tokens = usage.output_tokens if usage else 0
        total_tokens = input_tokens + output_tokens
        print(f"\n{self.agent_name} {self.event_counter}\n{agent.name} Ended with output:\n{output}\n\nUsage:\n\tINPUT TOKENS: {input_tokens}\n\tOUTPUT TOKENS: {output_tokens}\n\tTOTAL TOKENS USED: {total_tokens}\n")


class myInfo(BaseModel):
    name : str
    profession: str
    subjects: list[str] = None


@function_tool
def get_my_info(ctx:RunContextWrapper[myInfo]):
    """Greet user my his/her name they show their information"""
    return f"You are {ctx.context.name}. A {ctx.context.profession} by profession.Your subjects include {ctx.context.subjects}."


career_councelor= Agent(
    name = "Career Councelor",
    instructions = "You are a Career Councelor agent. Analyze user's info by using tool [get_my_info] and do provide him/her your best Counceling services.",
    handoff_description = "A career councelor agent who will guide user according to his/her profession.",
    tools = [get_my_info],
    model = model,
)


agent = Agent[myInfo](
    name = "Personal Assistant",
    instructions = "You are a Personal Assistant of user, help them with their queries."
                   "If user asks anything for their career handoff to career_councelor agent."
                   "If user only asks for the information about him/her self use tool get_my_info.",
    model = model,
    tools = [get_my_info],
    handoffs = [career_councelor],
)

context : myInfo = myInfo(name = "Amraha", profession = "Agentic AI Developer", subjects = ["CS", "Physics", "Maths"])

async def main():
    await Runner.run(
        agent,
        # "I've given you lil bit of my information. Guide me for my career as per my profession.",
        "What do you know about me?",
        context = context,
        hooks=TestRunHooks(agent_name="Your Personal Assistant")
    )

if __name__ == "__main__":
    asyncio.run(main())


# OUTPUT1 👇🏻

# -------Your Personal Assistant 1------
# Personal Assistant Started!
# Tokens usage:
#         INPUT TOKENS: 0
#         OUTPUT TOKENS: 0
#         TOTAL TOKENS: 0
# -------Your Personal Assistant 2------
# Personal Assistant handed off to Career Councelor

# -------Your Personal Assistant 3------
# Career Councelor Started!
# Tokens usage:
#         INPUT TOKENS: 165
#         OUTPUT TOKENS: 16
#         TOTAL TOKENS: 181

# Your Personal Assistant 4
# get_my_info Started tool Career Councelor


# Your Personal Assistant 5
# get_my_info Ended tool Career Councelor with output:
# You are Amraha. A Agentic AI Developer by profession.Your subjects include ['CS', 'Physics', 'Maths'].

# Your Personal Assistant 6
# Career Councelor Ended with output:
# Hello Amraha! It's great to know you're an Agentic AI Developer. This is a very exciting and rapidly evolving field with immense potential. Given your background in CS, Physics, and Maths, you have a strong foundation for this profession.      

# Here's some guidance to help you further your career:

# 1.  **Continuous Learning and Skill Enhancement:** The AI landscape changes quickly.
#     *   **Deepen your understanding of core AI/ML concepts:** Stay updated on advanced algorithms, neural network architectures (e.g., Transformers, GANs, Diffusion Models), reinforcement learning, and ethical AI principles.
#     *   **Programming Languages:** Ensure proficiency in Python, and consider exploring other languages like Julia or R if relevant to specific AI applications.
#     *   **Tools and Frameworks:** Master popular frameworks such as TensorFlow, PyTorch, scikit-learn, and Hugging Face.  
#     *   **Cloud Platforms:** Gain expertise in cloud AI services (AWS SageMaker, Google Cloud AI Platform, Azure ML) as many AI solutions are deployed in the cloud.

# 2.  **Specialization within AI:** While being a generalist is good, specializing can open up unique opportunities. Consider areas like:
#     *   **Natural Language Processing (NLP):** Working with language models, sentiment analysis, text generation.
#     *   **Computer Vision (CV):** Image recognition, object detection, video analysis.
#     *   **Robotics and Autonomous Systems:** Integrating AI into physical systems.
#     *   **Responsible AI/AI Ethics:** Focusing on fairness, transparency, and accountability in AI systems.
#     *   **AI for Science/Physics/Maths:** Applying AI to solve complex problems in your foundational subjects (e.g., simulating physical systems, optimizing mathematical models).

# 3.  **Build a Strong Portfolio:**
#     *   Work on personal projects that demonstrate your skills and interests.
#     *   Contribute to open-source AI projects.
#     *   Participate in Kaggle competitions or similar challenges.

# 4.  **Networking and Community Engagement:**
#     *   Attend AI conferences, workshops, and meetups (online and offline).
#     *   Join professional AI communities and forums.
#     *   Connect with other AI professionals on platforms like LinkedIn.
#     *   Consider sharing your knowledge through blogs, articles, or presentations.

# 5.  **Explore Career Paths:** As an Agentic AI Developer, your path can evolve:
#     *   **Senior AI/ML Engineer:** Leading development of complex AI systems.
#     *   **AI/ML Researcher:** Focusing on developing novel AI algorithms and models.
#     *   **AI Architect:** Designing large-scale AI infrastructures.
#     *   **Data Scientist/Machine Learning Scientist:** Depending on the focus on data analysis vs. model development.     
#     *   **AI Product Manager:** Guiding the development of AI-powered products.
#     *   **Consultant:** Providing AI expertise to various organizations.

# 6.  **Stay Updated with Trends:** Keep an eye on emerging trends like explainable AI (XAI), federated learning, quantum AI, and advancements in large language models (LLMs).

# Your combination of an Agentic AI Developer profession with a strong foundation in CS, Physics, and Maths is powerful. Leverage your analytical and problem-solving skills from Physics and Maths to tackle complex AI challenges, and your CS knowledge to build robust AI systems.

# Good luck with your career journey!

# Usage:
#         INPUT TOKENS: 499
#         OUTPUT TOKENS: 755
#         TOTAL TOKENS USED: 1254



# OUTPUT2 👇🏻

# -------Your Personal Assistant 1------
# Personal Assistant Started!
# Tokens usage:
#         INPUT TOKENS: 0
#         OUTPUT TOKENS: 0
#         TOTAL TOKENS: 0

# Your Personal Assistant 2
# get_my_info Started tool Personal Assistant


# Your Personal Assistant 3
# get_my_info Ended tool Personal Assistant with output:
# You are Amraha. A Agentic AI Developer by profession.Your subjects include ['CS', 'Physics', 'Maths'].

# Your Personal Assistant 4
# Personal Assistant Ended with output:
# You are Amraha. An Agentic AI Developer by profession. Your subjects include CS, Physics, and Maths.

# Usage:
#         INPUT TOKENS: 362
#         OUTPUT TOKENS: 36
#         TOTAL TOKENS USED: 398