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


# sst+rag(接入电商)+functioncall（支持多种工具函数）+tts
class AssistantFnc(llm.FunctionContext):
    """
    这个类定义了工具函数
    """

    @llm.ai_callable()
    async def get_weather(
        self,
        location: Annotated[str, llm.TypeInfo(description="查询天气的地点")],
    ):
        # 当问到天气的时候，调用函数
        # 函数调用过程中如何通知用户可能需要等待一段时间，有以下两种选项：
        # 在函数调用触发后立即使用 .say 发送填充消息。
        # 提示代理在进行函数调用时返回一个文本响应。

        call_ctx = AgentCallContext.get_current()
        filler_messages = [
            "让我帮你查询{location}的天气吧",
            "让我看看现在{location}的天气怎么样",
            "{location}当前的天气是",
        ]

        # filler_messages = [
        #     "Let me check the weather for {location} for you",
        #     "Let me see what the weather is like in {location} now",
        #     "The current weather in {location} is",
        # ]

        # 获取当前的上下文，随机挑一句加载天气的话放进去
        message = random.choice(filler_messages).format(location=location)

        # 把消息查询的结果放到聊天记录的末尾
        speech_handle = await call_ctx.agent.say(message, add_to_chat_ctx=True)

        logger.info(f"正在获取{location}的天气")
        url = f"https://wttr.in/{urllib.parse.quote(location)}?format=%C+%t"
        weather_data = ""
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    weather_data = f"{location}的天气是{await response.text()}."
                else:
                    raise Exception(f"获取不到天气，status code: {response.status}")

        # 等待语音处理完成后再返回函数调用的结果
        await speech_handle.join()
        return weather_data


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    async def _enrich_with_rag(agent: VoicePipelineAgent, chat_ctx: llm.ChatContext):
        # 拿到最后一句话去调rag检索
        user_msg = chat_ctx.messages[-1]
        logger.info(f"对话内容：{chat_ctx.messages}")
        query_text = user_msg.content
        similarity = 0.6
        top_number = 3
        search_mode = "blend"
        url = f"http://183.131.7.9:8003/api/application/68b32f0e-64e2-11ef-977b-26cf8447a8c9/hit_test?query_text={query_text}&similarity={similarity}&top_number={top_number}&search_mode={search_mode}"
        headers = {
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh-CN,zh;q=0.9",
            "Connection": "keep-alive",
            "Referer": "http://183.131.7.9:8003/ui/application/68b32f0e-64e2-11ef-977b-26cf8447a8c9/SIMPLE/hit-test",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        }

        response = requests.get(url, headers=headers, verify=False)
        if response.status_code == 200:
            result = response.json()
            data = result.get("data", [])
            logger.info(f"RAG检索的文档: {data}")

            if len(data) > 0:

                contents = [item.get("content", "") for item in data]
                combined_content = "\n".join(contents)

                # 将检索文章放到ai建议的对话内容中
                rag_msg = llm.ChatMessage.create(
                    text="参考上下文:\n" + combined_content,
                    role="assistant",
                )

                # 在最后一句用户对话之前插入建议对话
                chat_ctx.messages[-1] = rag_msg
                chat_ctx.messages.append(user_msg)
        else:
            logger.info(f"RAG检索失败，错误码：{response.status_code}")

    # initial_ctx = llm.ChatContext().append(
    #     role="system",
    #     text=(
    #         "You are a voice assistant created by LiveKit. Your interface with users will be voice. "
    #         "You should use short and concise responses, and avoiding usage of unpronouncable punctuation."
    #         "Use the provided context to answer the user's question if needed."
    #     ),
    # )

    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            "请记住，你是一个电商直播主播，当前你在直播并且回答弹幕问题，你与用户的交互方式是语音。你应使用简短明了的回答，并避免使用难以发音的标点符号。"
            """
            回答要求：
            
            - 请使用简洁且专业的语言来回答用户的问题。
            - 问答的语气需要符合直播主播的风格
            - 避免提及你是从已知信息中获得的知识。
            - 请保证答案与已知信息中描述的一致。
            - 请使用 Markdown 语法优化答案的格式。
            - 已知信息中的图片、链接地址和脚本语言请直接返回。
            - 请使用与问题相同的语言来回答。
            """
            # "You are a voice assistant created by LiveKit. Your interface with users will be voice. "
            # "You should use short and concise responses, and avoiding usage of unpronouncable punctuation."
        ),
    )
    # 创建functioncall实例
    fnc_ctx = AssistantFnc()

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
        stt=deepgram.STT(
            model=dg_model,
            api_key="0f70619a76a06439f36497c11047e457d9b2cc05",
            language="zh-CN",
        ),
        # llm=openai.LLM(
        #     base_url="https://api2.aigcbest.top/v1",
        #     api_key="sk-0vc6oFIta4zDcjqS714dEd3d50854183902b7f857dEc07F4",
        # ),
        llm=openai.LLM(
            base_url="http://183.131.7.9:8083/api/application/68b32f0e-64e2-11ef-977b-26cf8447a8c9",
            api_key="application-df40f7b453cb74ee46da8dc31cc385f7",
        ),
        tts=cartesia.TTS(
            api_key="sk_car_NNdRyLj3PSzFlBM7s7tNC",
            model="sonic-2024-10-19",
            language="zh",
        ),
        chat_ctx=initial_ctx,
        # fnc_ctx=fnc_ctx,
        # before_llm_cb=_enrich_with_rag,
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

    # listen to incoming chat messages, only required if you'd like the agent to
    # answer incoming messages from Chat
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
    # await agent.say("hello,how can i help you today?", allow_interruptions=True)


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
