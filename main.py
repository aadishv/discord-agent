import asyncio
import tempfile
from gtts import gTTS
from pydantic_ai import RunContext
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
    "google/gemini-2.0-flash-001"
    # "deepseek/deepseek-v3.2-exp:floor"
)
# settings = OpenRouterModelSettings(openrouter_reasoning={"enabled": True})
agent = Agent(
    model,
    deps_type=DiscordAgentContext[None],
    # model_settings=settings,
    instructions="""
    You are a Chinese language teacher for beginners focused on listening comprehension. Your primary goal is to generate short Chinese sentences or passages based on the user's level or specific vocabulary requests.

    Follow this workflow for every interaction:

    1. Generate a relevant Chinese sentence or passage.
    2. Immediately use the `tts` tool to convert that text to audio for the user.
    3. Present 3 to 4 English comprehension questions about the passage to the user. DO NOT state the English meanings of the texts.

    Of course, never present the user with the Chinese or English meanings of the texts until the questions have all been answered; it otherwise defeats the purpose of the exercise.

    Keep all responses extremely concise and optimized for Discord. Do not use bullet points in any part of your output. Use a back-and-forth conversational style to grade the user's answers before moving to the next exercise.

    Do not grade answers and move on the next question in the same step. Use a transition like "Ready to move on?" and then wait for the user to affirm.
    """
)

@agent.tool
async def tts(ctx: RunContext[DiscordAgentContext[None]], chinese_text: str):
    """Run TTS on a given input Chinese text and send the output to the user"""
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        mp3_path = f.name

    def generate_audio():
        tts = gTTS(chinese_text, lang="zh-CN")
        tts.save(mp3_path)

    await asyncio.to_thread(generate_audio)

    await ctx.deps.thread.send(attachment=hikari.File(mp3_path))

    os.unlink(mp3_path)

    # return something the agent is familiar with instead of something that will
    # clog up its context
    return "200 OK"

my_agent = DiscordAgent(agent, None)


async def main():
    my_agent.register(bot)

    await bot.start()

    try:
        await asyncio.Event().wait()
    finally:
        await bot.join()

asyncio.run(main())
