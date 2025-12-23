from pydantic_core import to_jsonable_python, to_json

from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

import asyncio
import os
import struct
import time

import hikari
import lmdb
from dotenv import load_dotenv
from pydantic_ai import ModelResponse, ModelMessagesTypeAdapter, ModelRequest, capture_run_messages
from pydantic_ai.agent import Agent
from pydantic_ai.messages import BinaryContent, UserContent
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
    # "google/gemini-2.5-flash-lite"
    # :floor uses the slowest one so we can test failure modes
    "deepseek/deepseek-v3.2-exp"
)
settings = OpenRouterModelSettings(openrouter_reasoning={"enabled": True})
agent = Agent(
    model,
    deps_type=hikari.GuildThreadChannel,
    instructions="""
    You are an expert Chinese teacher.

    Your responses should primarily be in English, as the learners are not yet fluent; only practice content etc. should be in Chinese.

    You are operating as a Discord bot; your message should thus fit the context in terms of tone. An example message is:

    USER:
        > hi, sorry it might sound stupid but how do i say like see u again?
    ASSISTANT (you):
        > no worries, we're here to learn \\:) "see you again" in Chinese is "再见" (zài jiàn)

    Respond in a polite and extremely concise tone. Do not add extra sentences deviating from the user's question. For each,
    continuing the above example, do not include follow-up suggestions such as "Do you want to learn how to write 再见？".

    Politely steer the conversation back to Chinese if the user deviates.
    """,
    model_settings=settings,
)

IMAGE_MEDIA_TYPES = [
    "image/avif",
    "image/bmp",
    "image/gif",
    "image/jpeg",
    "image/png",
    "image/svg+xml",
    "image/webp",
]

AUDIO_MEDIA_TYPES = [
    "audio/aac",
    "audio/mpeg",
    "audio/wav",
    "audio/webm",
]

md_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN, chunk_size=2000, chunk_overlap=0
)

db = lmdb.open("threads.db", max_dbs=10)
thread_user_kv = db.open_db(b"user")
thread_contents_kv = db.open_db(b"contents")

active_streams: dict[hikari.Snowflake, asyncio.Task] = {}


async def respond_to_message(user: hikari.Message, thread: hikari.GuildThreadChannel, info_message: hikari.Message):
    # synchronous input procressing
    key = struct.pack(">Q", thread.id)

    with db.begin() as txn:
        raw_message_history: bytes | None = txn.get(key, db=thread_contents_kv)

    message_history: list[ModelRequest | ModelResponse] | None = ModelMessagesTypeAdapter.validate_json(raw_message_history.decode('utf-8')) if raw_message_history else None

    user_message: list[UserContent] = [
        user.content if user.content else "[No text provided by user]",
    ]

    ignored_attachments = []
    for attachment in user.attachments:
        if attachment.media_type in IMAGE_MEDIA_TYPES + AUDIO_MEDIA_TYPES + [
            "application/pdf"
        ]:
            data = await attachment.read()
            user_message.append(BinaryContent(data, media_type=attachment.media_type))
        else:
            ignored_attachments.append(attachment.filename)

    start_time = time.time()

    # periodically updates start message w/ info
    async def reporter():
        try:
            while True:
                msg = "warning: ignoring following attachments due to invalid type: {}\n".format(', '.join(ignored_attachments)) if ignored_attachments else ""
                msg += f"info: thinking for {int(time.time() - start_time)}s"
                await info_message.edit(msg)
                await asyncio.sleep(1.0)
        except asyncio.CancelledError as e:
            msg = f"info: thought for {int(time.time() - start_time)}s\n" + str(e)
            await info_message.edit(msg)

    report = asyncio.create_task(reporter())
    with capture_run_messages() as messages:
        try:
            async with thread.trigger_typing():
                res = await agent.run(user_message, deps=thread, message_history=message_history)

                for chunk in md_splitter.split_text(res.output):
                    await thread.send(chunk)

                report.cancel(f"info: cost ${res.response.cost().total_price}")
        except asyncio.CancelledError:
            report.cancel("[interrupted]")
        except Exception as e:
            print("Error occured in respond_to_message:", e)
        finally:
            with db.begin(write=True) as txn:
                txn.put(key, to_json(messages), db=thread_contents_kv)


@bot.listen()
async def ping(event: hikari.MessageCreateEvent) -> None:
    if not event.is_human:
        return
    me = bot.get_me()
    if me is None:
        return

    if event.message.content is None:
        return

    event.message.content = event.message.content.replace(f"<@{me.id}>", "").strip()

    if not event.message.content:
        return

    print("what")
    # either 1) this is in the right channel or 2) this is in a known thread
    if event.message.channel_id == 1452524008233762818:
        thread = await bot.rest.create_thread(
            event.channel_id,
            hikari.ChannelType.GUILD_PUBLIC_THREAD,
            event.message.content
            if len(event.message.content) < 100
            else event.message.content[:97] + "...",
            auto_archive_duration=60,  # archive after 1 hour
        )
        id = struct.pack(">Q", thread.id)
        with db.begin(write=True) as txn:
            user_raw = struct.pack(">Q", event.message.author.id)
            txn.put(id, user_raw, db=thread_user_kv)
    else:
        id = struct.pack(">Q", event.message.channel_id)
        with db.begin() as txn:
            user_raw: bytes | None = txn.get(id, db=thread_user_kv)
            if user_raw is None:
                # not a known thread / wrong user message
                print("not a known thread")
                await event.message.add_reaction(hikari.UnicodeEmoji("🚧"))
                return
            user: int = struct.unpack(">Q", user_raw)[0]
            if user != event.message.author.id:
                print("wrong user")
                await event.message.add_reaction(hikari.UnicodeEmoji("🚧"))
                return
        thread = bot.cache.get_thread(event.message.channel_id)
        if not thread:
            print("thread not in cache")
            await event.message.add_reaction(hikari.UnicodeEmoji("🚧"))
            return

    if thread.id in active_streams:
        active_streams[thread.id].cancel()
        try:
            await active_streams[thread.id]
        except asyncio.CancelledError:
            pass

    start_message = await thread.send("starting...")
    active_streams[thread.id] = asyncio.create_task(
        respond_to_message(event.message, thread, start_message)
    )


bot.run()
