from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, Runner, set_tracing_disabled
from dotenv import load_dotenv
from rich import print
import os

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

result = Runner.run_sync(
    agent,
    "Write a famous ghazal for me."
)

print(result.final_output)


# OUTPUT 👇🏻

# جی ضرور، ایک بہت مشہور غزل پیش خدمت ہے، جو میرزا غالب کی ہے۔ امید ہے آپ کو پسند آئے گی:

# **دلِ ناداں تجھے ہوا کیا ہے**
# **آخر اِس درد کی دوا کیا ہے**

# **ہم ہیں مشتاق اور وہ بیزار**
# **یا الٰہی یہ ماجرا کیا ہے**

# **میں بھی منہ میں زبان رکھتا ہوں**
# **کاش پوچھو کہ مدعا کیا ہے**

# **جبکہ تجھ بِن نہیں کوئی موجود**
# **پھر یہ ہنگامہ اے خدا کیا ہے**

# **یہ پری چہرہ لوگ کیسے ہیں**
# **غمزہ و عشوہ و ادا کیا ہے**

# **شکنِ زلفِ عنبریں کیوں ہے**
# **عطرِ پیراہنِ قبا کیا ہے**

# **جب کہ ہر حال میں وہ ساتھ ہے، غالبؔ**
# **پھر یہ محتاجیِ دعا کیا ہے**


