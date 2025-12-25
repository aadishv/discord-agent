from pydantic_core import to_json
from dataclasses import dataclass
import time
import struct
import asyncio
import hikari
import lmdb
from pydantic_ai import Agent, ModelRequest, ModelResponse, ModelMessagesTypeAdapter, UserContent, BinaryContent, PartEndEvent, TextPart, AgentRunResultEvent, AgentRunResult
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
    agent: Agent[DiscordAgentContext[UserData], str]

    def __init__(self, agent: Agent[DiscordAgentContext[UserData], str], initial_context: UserData, db_path="threads.db"):
        self.agent = agent

        self.db = lmdb.open(db_path, max_dbs=10)
        self.thread_user_kv = self.db.open_db(b"user")
        self.thread_contents_kv = self.db.open_db(b"contents")

        # thread: queue of messages (first is the next in the list)
        self.active_streams: dict[hikari.Snowflake, list[hikari.MessageCreateEvent]] = {}

        self.user_data = initial_context

    def update_context(self, user_data: UserData):
        self.user_data = user_data

    # input + ignored attachments
    async def _parse_message(self, user: hikari.Message) -> tuple[list[UserContent], list[str]]:
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

        return user_message, ignored_attachments

    async def respond_to_message(self, event: hikari.MessageCreateEvent, thread: hikari.GuildThreadChannel, info_message: hikari.Message):
        # synchronous input procressing
        key = struct.pack(">Q", thread.id)

        with self.db.begin() as txn:
            raw_message_history: bytes | None = txn.get(key, db=self.thread_contents_kv)

        message_history: list[ModelRequest | ModelResponse] = ModelMessagesTypeAdapter.validate_json(raw_message_history.decode('utf-8')) if raw_message_history else []

        try:
            # periodically updates start message w/ info
            all_ignored_attachments = []
            cost: str | None = None
            start_time = time.time()
            async def reporter(done=False):
                def build_msg():
                    msg = "warning: ignoring following attachments due to invalid type: {}\n".format(', '.join(all_ignored_attachments)) if all_ignored_attachments else ""
                    msg += f"info: {'working' if not done else 'worked'} for {int(time.time() - start_time)}s"
                    if cost:
                        msg += f"\ninfo: cost ${cost}"
                    if thread.id in self.active_streams and self.active_streams[thread.id]:
                        msg += f"\ninfo: {len(self.active_streams[thread.id])} queued messages"
                    return msg
                try:
                    while True:
                        await info_message.edit(build_msg())
                        await asyncio.sleep(1.0)
                except asyncio.CancelledError:
                    await info_message.edit(build_msg())
                except Exception as e:
                    print("exception occured in reporter: ", e)
            report = asyncio.create_task(reporter())

            current_message = event.message

            while True:
                user_message, new_ignored_attachments = await self._parse_message(current_message)
                all_ignored_attachments = all_ignored_attachments + new_ignored_attachments

                res: AgentRunResult | None = None
                async for chunk in self.agent.run_stream_events(user_message, deps=DiscordAgentContext(self.user_data, thread, event), message_history=message_history):
                    if isinstance(chunk, AgentRunResultEvent):
                        cost = str(chunk.result.response.cost().total_price)
                        res = chunk.result
                    if not isinstance(chunk, PartEndEvent):
                        continue
                    if not isinstance(chunk.part, TextPart):
                        continue
                    for chunk in md_splitter.split_text(chunk.part.content):
                        await thread.send(chunk)
                if res:
                    message_history = res.all_messages()
                if self.active_streams[thread.id]:
                    current_message = self.active_streams[thread.id].pop(0).message
                else:
                    break
        finally:
            del self.active_streams[thread.id]
            report.cancel()
            with self.db.begin(write=True) as txn:
                txn.put(key, to_json(message_history), db=self.thread_contents_kv)



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
            print("adding to queue")
            self.active_streams[thread.id].append(event)
            return

        start_message = await thread.send("starting...")

        self.active_streams[thread.id] = []
        async with thread.trigger_typing():
            await self.respond_to_message(event, thread, start_message)

    def register(self, bot: hikari.GatewayBot):
        @bot.listen()
        async def callback(event: hikari.MessageCreateEvent):
            print(event.message.content)
            await self.message_create_handler(event, bot)
