from pydantic_core import to_json
from dataclasses import dataclass
import time
import struct
import asyncio
import hikari
import lmdb
from pydantic_ai import Agent, ModelRequest, ModelResponse, ModelMessagesTypeAdapter, UserContent, BinaryContent, capture_run_messages
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

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

@dataclass
class DiscordAgentContext[UserData]:
    data: UserData
    thread: hikari.GuildThreadChannel
    trigger: hikari.MessageCreateEvent

class DiscordAgent[UserData]:
    def __init__(self, agent: Agent[DiscordAgentContext[UserData], str], initial_context: UserData, db_path="threads.db"):
        self.agent = agent

        self.db = lmdb.open(db_path, max_dbs=10)
        self.thread_user_kv = self.db.open_db(b"user")
        self.thread_contents_kv = self.db.open_db(b"contents")

        self.active_streams: dict[hikari.Snowflake, asyncio.Task] = {}

        self.user_data = initial_context

    def update_context(self, user_data: UserData):
        self.user_data = user_data

    async def respond_to_message(self, event: hikari.MessageCreateEvent, thread: hikari.GuildThreadChannel, info_message: hikari.Message):
        user = event.message

        # synchronous input procressing
        key = struct.pack(">Q", thread.id)

        with self.db.begin() as txn:
            raw_message_history: bytes | None = txn.get(key, db=self.thread_contents_kv)

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
                    res = await self.agent.run(user_message, deps=DiscordAgentContext(self.user_data, thread, event), message_history=message_history)

                    for chunk in md_splitter.split_text(res.output):
                        await thread.send(chunk)

                    report.cancel(f"info: cost ${res.response.cost().total_price}")
            except asyncio.CancelledError:
                report.cancel("[interrupted]")
            except Exception as e:
                print("Error occured in respond_to_message:", e)
            finally:
                with self.db.begin(write=True) as txn:
                    txn.put(key, to_json(messages), db=self.thread_contents_kv)

    async def message_create_handler(self, event: hikari.MessageCreateEvent, bot: hikari.GatewayBot) -> None:
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
            with self.db.begin(write=True) as txn:
                user_raw = struct.pack(">Q", event.message.author.id)
                txn.put(id, user_raw, db=self.thread_user_kv)
        else:
            id = struct.pack(">Q", event.message.channel_id)
            with self.db.begin() as txn:
                user_raw: bytes | None = txn.get(id, db=self.thread_user_kv)
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

        if thread.id in self.active_streams:
            self.active_streams[thread.id].cancel()
            try:
                await self.active_streams[thread.id]
            except asyncio.CancelledError:
                pass

        start_message = await thread.send("starting...")
        self.active_streams[thread.id] = asyncio.create_task(
            self.respond_to_message(event, thread, start_message)
        )

    def register(self, bot: hikari.GatewayBot):
        print("yo")
        @bot.listen()
        async def callback(event: hikari.MessageCreateEvent):
            print(event.message.content)
            await self.message_create_handler(event, bot)
