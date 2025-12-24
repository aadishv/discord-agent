import asyncio
from pydantic_ai import RunContext
import random
from agent import DiscordAgentContext, DiscordAgent
import os
import hikari
from dotenv import load_dotenv
from pydantic_ai.agent import Agent
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings

load_dotenv()

token = os.getenv("DISCORD_TOKEN")
if token is None:
    raise ValueError("DISCORD_TOKEN environment variable is not set")
bot = hikari.GatewayBot(
    token=token,
    intents=(
        hikari.Intents.GUILDS
        | hikari.Intents.GUILD_MESSAGES
        | hikari.Intents.MESSAGE_CONTENT
    ),
)

model = OpenRouterModel(
    # gemini 2.5 flash/-lite. CRAZY fast. might go back to deepseek/deepseek-v3.2-exp but doesn't support audio input. tradeoffs
    "google/gemini-2.5-flash-lite"
    # :floor uses the slowest one so we can test failure modes
    # "deepseek/deepseek-v3.2-exp"
)
# settings = OpenRouterModelSettings(openrouter_reasoning={"enabled": True})
agent = Agent(
    model,
    deps_type=DiscordAgentContext[float],
    instructions="You are a weather reporter. Call the get_weather tool to get information about the weather.",
    # instructions="""
    # You are an expert Chinese teacher.

    # Your responses should primarily be in English, as the learners are not yet fluent; only practice content etc. should be in Chinese.

    # You are operating as a Discord bot; your message should thus fit the context in terms of tone. An example message is:

    # USER:
    #     > hi, sorry it might sound stupid but how do i say like see u again?
    # ASSISTANT (you):
    #     > no worries, we're here to learn \\:) "see you again" in Chinese is "再见" (zài jiàn)

    # Respond in a polite and extremely concise tone. Do not add extra sentences deviating from the user's question. For each,
    # continuing the above example, do not include follow-up suggestions such as "Do you want to learn how to write 再见？".

    # Politely steer the conversation back to Chinese if the user deviates.
    # """,
    # model_settings=settings,
)

@agent.tool
async def get_weather(ctx: RunContext[DiscordAgentContext[float]], location: str):
    await ctx.deps.thread.send(f"INFO: `Fetching weather data for location {location}...`")
    return f"The temperature in {location} is {ctx.deps.data} degrees Fahrenheit."

my_agent = DiscordAgent(agent, 25.0)

my_agent.register(bot)

async def main():
    await bot.start()

    try:
        while True:
            await asyncio.sleep(1)
            my_agent.update_context(random.random() * 100)
    finally:
        await bot.join()

asyncio.run(main())
