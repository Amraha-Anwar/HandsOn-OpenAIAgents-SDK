from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, Runner, set_tracing_disabled
from openai.types.responses import ResponseTextDeltaEvent
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
    result = Runner.run_streamed(
        agent,
        "write briefly top 5 qualities of Pakistan."
    )

    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end="", flush= True)


asyncio.run(main())



# OUTPUT 👇🏻
# یقیناً، پاکستان کی ٹاپ 5 خوبیاں یہ ہیں:

# 1.**قدرتی حُسن:** پاکستان بلند پہاڑوں، سرسبز و شاداب وادیوں، وسیع صحراؤں
#  اور خوبصورت ساحلوں سمیت بے پناہ قدرتی خوبصورتی کا مالک ہے۔

# 2.**شاندار مہمان نوازی:** پاکستانی عوام اپنی بے مثال مہمان نوازی 
# اور سخاوت کے لیے دنیا بھر میں مشہور ہیں۔

# 3.  **ثقافتی تنوع:** ملک مختلف ثقافتوں، زبانوں، اور روایات کا 
# ایک حسین امتزاج ہے، جو اسے ایک منفرد اور رنگا رنگ شناخت دیتا
# ہے۔

# 4. **تاریخی و تہذیبی ورثہ:** پاکستان قدیم تہذیبوں 
# (جیسے ہڑپہ اور موہنجو داڑو) اور اسلامی فنِ تعمیر کے عظیم شاہکاروں کا گہوارہ 
# ہے۔

# 5.  **اسٹریٹیجک جغرافیائی محل وقوع:** اس کا جغرافیائی محل وقوع اسے جنوبی ا
# یشیا، وسطی ایشیا اور مشرق وسطیٰ کے درمیان ایک کلیدی اہمیت دیتا ہے۔