from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, Runner, set_tracing_disabled
from dotenv import load_dotenv
from rich import print
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

agent = Agent(
    name = "Urdu Expert",
    instructions = "You are a pro at Urdu. No matter what user asks, reply back him/her in Urdu.",
    model = model
)

async def main():
    result = await Runner.run(
        agent,
        "Write a famous ghazal for me."
    )

    print(result.final_output)

asyncio.run(main())



# OUTPUT 👇🏻

# جی بالکل، حاضرِ خدمت ہے فیض احمد فیض کی ایک بہت مشہور غزل:

# **گلوں میں رنگ بھرے بادِ نوبہار چلے**
# **چلے بھی آؤ کہ گلشن کا کاروبار چلے**

# **قفس اداس ہے یارو صبا سے کچھ تو کہو**
# **کہیں تو بہرِ خدا آج ذکرِ یار چلے**

# **کبھی تو صبح ترے کنجِ لب سے ہو آغاز**
# **کبھی تو شب سرِ کاکل سے مشکبار چلے**

# **بڑا ہے درد کا رشتہ یہ دل غریب سہی**
# **تمہارے نام پہ آئیں گے غم گسار چلے**

# **جو ہم پہ گزری سو گزری مگر شبِ ہجراں**
# **ہمارے اشک تری عاقبت سنوار چلے**

# **حضورِ یار ہوئی چشمِ تر کی شرمساری**
# **کہیں سے اٹھ کے ترا آرزو نکھار چلے**

# **مقامِ فیض کوئی راہ میں جچا ہی نہیں**
# **جو کوئے یار سے نکلے سو سوئے دار چلے**

# امید ہے آپ کو پسند آئے گی۔
