import asyncio
import logging
import random
from typing import Annotated

import aiohttp
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
    metrics,
)
from livekit.agents.pipeline import VoicePipelineAgent, AgentCallContext
from livekit.plugins import deepgram, openai, silero, cartesia
import urllib

import requests

load_dotenv()
logger = logging.getLogger("voice-assistant")


# sst+rag(接入电商)+tts

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):

    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            "你是一个由 LiveKit 创建的语音助手。你与用户的交互方式是语音。"
            "你应该使用简短明了的回答，并避免使用难以发音的标点符号。"
        ),
    )

    # 连接到房间
    logger.info(f"连接到房间 {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # 等待连接
    participant = await ctx.wait_for_participant()
    logger.info(f"启动语音助手为参与者提供服务，参与者： {participant.identity}")

    dg_model = "nova-2-general"
    # 如果是电话就用电话模型
    if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
        dg_model = "nova-2-phonecall"

    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        # stt=openai.STT(
        #     base_url="https://api2.aigcbest.top/v1",
        #     api_key="sk-0vc6oFIta4zDcjqS714dEd3d50854183902b7f857dEc07F4",
        #     language="zh",
        # ),
        stt=deepgram.STT(
            model=dg_model,
            api_key="0f70619a76a06439f36497c11047e457d9b2cc05",
            language="zh-CN",
        ),
        llm=openai.LLM(
            base_url="http://183.131.7.9:8083/api/application/68b32f0e-64e2-11ef-977b-26cf8447a8c9",
            api_key="application-df40f7b453cb74ee46da8dc31cc385f7",
        ),
        tts=openai.TTS(
            base_url="https://api2.aigcbest.top/v1",
            api_key="sk-0vc6oFIta4zDcjqS714dEd3d50854183902b7f857dEc07F4",
        ),
        # tts=cartesia.TTS(
        #     api_key="sk_car_rcIeEFR3LnNwqmuWNNZSr",
        #     model="sonic-2024-10-19",
        #     language="zh",
        # ),
    )

    agent.start(ctx.room, participant)

    usage_collector = metrics.UsageCollector()

    # 捕获agents中sst,llm,tts日志信息
    @agent.on("metrics_collected")
    def _on_metrics_collected(mtrcs: metrics.AgentMetrics):
        metrics.log_metrics(mtrcs)
        usage_collector.collect(mtrcs)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: ${summary}")

    ctx.add_shutdown_callback(log_usage)

    # 监听聊天消息，获取聊天消息并使用大模型做对应的回答
    chat = rtc.ChatManager(ctx.room)

    async def answer_from_text(txt: str):
        chat_ctx = agent.chat_ctx.copy()
        chat_ctx.append(role="user", text=txt)
        stream = agent.llm.chat(chat_ctx=chat_ctx)
        await agent.say(stream)

    @chat.on("message_received")
    def on_chat_received(msg: rtc.ChatMessage):
        if msg.message:
            asyncio.create_task(answer_from_text(msg.message))

    await agent.say("您好，我有什么能帮助你的", allow_interruptions=True)

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            api_key="APIdmk2CKNSHv9k",
            ws_url="wss://test-4toglnrx.livekit.cloud",
            api_secret="Ebn4x7wfeAtSSvdn6dmuYQ7FQZ9aNfe4ux8XGOHgcxYC",
        ),
        
    )
