from agents import Agent, OpenAIChatCompletionsModel, AsyncOpenAI, Runner, RunConfig, enable_verbose_stdout_logging
from dotenv import load_dotenv
import os

load_dotenv()
enable_verbose_stdout_logging()

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

math_agent = Agent(
    name = "Math Agent",
    instructions = "You are a Math Agent so You are very good at maths. Solve user's math related queries.",
    model = model,
    handoff_description = "Mathematics Related queries resolver."
)

chef_agent = Agent(
    name = "Chef",
    instructions = "You are an expert chef. Help user if they ask anything cooking related.",
    model = model,
    handoff_description = "An expert chef who helps in cooking."
)

main_agent = Agent (
    name = "Manager",
    instructions = "You are a manager who manages user's queries."
                   "If user ask for anything related to Math handoff to math_agent."
                   "If user ask for the help in cooking handoff to chef_agent.",
    handoffs = [math_agent, chef_agent],
    model = model
)

result = Runner.run_sync(
    main_agent,
    "I was cooking Beef Biryani but I forgot to add salt in it, now how can I fix it?",
    run_config = config
)

print(result.final_output)


# OUTPUT 👇🏻

# Ah, a common kitchen predicament, even for the best of us! Don't worry, Chef is here to help you fix that Beef Biryani. It's definitely salvageable. Adding salt directly to the finished dish often results in uneven distribution and a crunchy texture, so we need a more refined approach.

# Here are a few ways you can fix your Beef Biryani:

# 1.  **The "Salted Broth" Method (Most Recommended for Even Distribution):**
#     *   **What you need:** About 1/2 to 1 cup of hot water or beef/chicken broth, salt to taste.
#     *   **How to do it:**
#         1.  In a small saucepan, heat the water or broth.
#         2.  Add a generous amount of salt to it, tasting it until it's quite salty, but not overpoweringly so (remember, it needs to season the whole dish).
#         3.  Gently drizzle this salted liquid over the biryani.
#         4.  Using a fork or a very light hand, gently fluff and mix the biryani, trying to distribute the liquid as evenly as possible without mashing the rice.
#         5.  Cover the pot again and let it sit for 5-10 minutes on very low heat, or just with the residual heat, to allow the rice and meat to absorb the salted liquid.

# 2.  **Enhance with a Salted Accompaniment:**
#     *   This is a simpler fix if you don't want to disturb the biryani too much.
#     *   **Raita:** Make a batch of raita (yogurt dip) but season it more generously than usual with salt, black salt, roasted cumin powder, and a pinch of black pepper. The salt in the raita will compensate for the lack of salt in the biryani as people eat them together.
#     *   **Salan or Gravy:** If you're serving your biryani with a Mirchi ka Salan (chili and peanut gravy) or a similar side gravy, ensure that gravy is well-seasoned with salt.

# 3.  **The "Lemon/Lime & Salt" Spritz:**
#     *   **What you need:** Fresh lemon or lime juice, a small amount of water, salt.
#     *   **How to do it:** Mix a tablespoon or two of fresh lemon/lime juice with an equal amount of water and dissolve a good pinch of salt into it. Gently spritz this mixture over the biryani, then fluff. The acidity from the citrus also brightens flavors and makes the existing flavors (even the unsalted ones) pop a bit more.

# 4.  **Offer Salt on the Side (Least Ideal for Biryani, but an Option):**
#     *   While not ideal for a dish like biryani where integrated seasoning is key, you can always offer a small bowl of fine table salt alongside the biryani for people to add to their individual plates if they wish. This is more of a last resort.

# **Key Tips for Success:**

# *   **Be Gentle:** Biryani rice can break easily. Use a light hand when mixing or fluffing.
# *   **Taste as you go:** When making the salted liquid, taste it to ensure it's not too salty. It's easier to add more salt than to remove it.
# *   **Warmth Helps:** The biryani will absorb the liquid better if it's still warm.

# Choose the method that best suits your comfort level and how much biryani you have left. My top recommendation is the "Salted Broth" method for the most uniform results.

# Good luck, Chef! Your Biryani will be delicious!

# ---------------------------------------------------------------------------------------------------------------------
 
# POV 📌

# I'M IMPRESSED!!!! 
# THE AGENT'S RESPONSE 🫡