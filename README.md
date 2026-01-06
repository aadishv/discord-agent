# Discord agent

A minimal, typesafe, and modular library which allows anyone to build AI agents that run in a Discord server using Hikari and Pydantic AI. Set up the bot to run in a server, after which it will monitor for messages from users and respond to them in a thread. Also supports built-in thread and user storage via a (somewhat) configurable file-based KV store. Bring your own model, tools, etc..

The library itself is in `agent.py` and only about 200 lines. This has no license so do wtv with it!!

There's also a Chinese teacher agent in `main.py`, which is what I initially set out to build. Thanks to the abstractions provided by the agent module, the whole thing is <100 LOC.

## Chinese agent TODOs

* test out more models
* improve conversational flow
* encourage model to use longer passages, e.g. 2-3 sentences

## More modularity stuff to do

* make the KV backend configurable using a `typing.Protocol`
* fix persistence of image-based chats
* allow users to configure when to respond to messages (this won't be fun because we do a lot of silly logic)
* add minimal support for streaming[^streaming]
* support for structured outputs -> custom UI? might be outside the scope of this library though

I'm not trying to turn this into a "general-purpose agent library" because that's a rather grandiose task; there are many other libraries that are better suited to have adapters for other platforms. See Mario Zechner's `mom` for an example.

[^streaming]: we currently don't support streaming because Discord has a strict (5 edits)/(5 seconds) rate limit for editing messages. Might get a minimal POC working soon.


## Contrived, 67-line example program

```python
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

# we use the Hikari library for setting up the discord bot,
# mainly for its typesafety and abstractions.
# 
# setting up the bot is outside the scope of this example
# but should be familiar if you have experience w/ bots
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

# i use openrouter but you can use any provider supported by pydantic ai
model = OpenRouterModel(
    # extremely fast+cheap model, plus supports all kinds of input modalities like audio.
    # thinking can also be enabled via a flag: see https://ai.pydantic.dev/thinking/#openrouter
    # 
    # it might not be smart enough for your use case; i recommend deepseek/deepseek-v3.2-exp as 
    # the next stepping stone.
    "google/gemini-2.5-flash-lite"
)

# initialize your Pydantic AI agent.
# see docs (https://ai.pydantic.dev/) for more
agent = Agent(
    model,
    deps_type=DiscordAgentContext[float],
    instructions="You are a weather reporter. Call the get_weather tool to get information about the weather.",
)

@agent.tool
async def get_weather(ctx: RunContext[DiscordAgentContext[float]], location: str):
    await ctx.deps.thread.send(f"INFO: `Fetching weather data for location {location}...`")
    return f"The temperature in {location} is {ctx.deps.data} degrees Fahrenheit."

my_agent = DiscordAgent(agent, 25.0)

my_agent.register(bot)

async def main():
    await bot.start()
    # a contrived example mainly to show how updating the context works
    try:
        while True:
            await asyncio.sleep(1)
            my_agent.update_context(random.random() * 100)
    finally:
        await bot.join()

asyncio.run(main())
```
