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
    "google/gemini-2.5-flash-lite"
    # "deepseek/deepseek-v3.2-exp:floor"
)
settings = OpenRouterModelSettings(openrouter_reasoning={"enabled": True})
agent = Agent(
    model,
    deps_type=DiscordAgentContext[None],
    model_settings=settings
)

# @agent.tool
async def text_to_speech(ctx: RunContext[DiscordAgentContext[None]], text: str, lang: str):
    """Convert text to speech and send as an audio file.

    Args:
        text: The text to convert to speech.
        lang: Language code - use 'en' for English, 'zh-CN' for Mandarin Chinese.
    """
    await ctx.deps.thread.send(f"INFO: `Generating TTS for: {text[:50]}{'...' if len(text) > 50 else ''}`")

    # Create temporary MP3 file
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        mp3_path = f.name

    def generate_audio():
        tts = gTTS(text, lang="zh-CN")
        tts.save(mp3_path)

    await asyncio.to_thread(generate_audio)

    # Send the MP3 file as an attachment
    await ctx.deps.thread.send(attachment=hikari.File(mp3_path))

    # Clean up temp file
    os.unlink(mp3_path)

    return "Successfully generated and sent TTS audio for the text."

my_agent = DiscordAgent(agent, None)


async def main():
    my_agent.register(bot)

    await bot.start()

    try:
        await asyncio.Event().wait()
    finally:
        await bot.join()

asyncio.run(main())
