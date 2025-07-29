import asyncio
import re
import random
import aiohttp
import json
from astrbot import logger
from astrbot.api.event import filter
from astrbot.api.provider import LLMResponse
from astrbot.api.star import Context, Star, register
from astrbot.core import AstrBotConfig
from astrbot.core.message.components import Record
from astrbot.core.message.message_event_result import MessageChain
from astrbot.core.platform import AstrMessageEvent
import astrbot.api.message_components as Comp
from pathlib import Path
from typing import Dict

SAVED_AUDIO_DIR = Path(__file__).resolve().parent / "temp"
SAVED_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

@register("GPT-SoVITS推理特化包连接器",
          "DITF16",
          "GPT-SoVITS 推理特化包astrbot连接器",
          "1.0.0",
          "https://github.com/DITF16/AstrBot-GPT-SoVITS-Inference-Connector")
class GSVPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        # 加载配置
        base_setting: Dict = config.get("base_setting", {})
        self.base_url: str = base_setting.get("base_url", "")
        auto_config: Dict = config.get("auto_config", {})
        self.send_record_probability: float = auto_config.get(
            "send_record_probability", 0.3
        )
        self.max_resp_text_len: int = auto_config.get("max_resp_text_len", 100)

        # 加载TTS参数
        self.tts_core_params: Dict = config.get("tts_params", {})
        self.tts_other_params: Dict = config.get("other_params", {})

        # 加载并启动定时清理任务
        cleanup_setting: Dict = config.get("cleanup_setting", {})
        self.cleanup_interval_hours: float = cleanup_setting.get("cleanup_interval", 1.0)

        if self.cleanup_interval_hours > 0:
            logger.info("TTS插件的定时清理任务已启用。")
            asyncio.create_task(self._periodic_cleanup())
        else:
            logger.info("TTS插件的定时清理任务已禁用。")

        if not self.base_url:
            logger.error("TTS插件未配置base_url，插件将无法工作！")

    def _perform_cleanup(self):
        """清理temp文件夹中的所有文件"""
        logger.info("开始执行TTS临时文件清理...")
        cleaned_count = 0
        try:
            for item in SAVED_AUDIO_DIR.iterdir():
                if item.is_file():
                    try:
                        item.unlink()
                        cleaned_count += 1
                    except Exception as e:
                        logger.error(f"删除文件 {item.name} 时出错: {e}")
            if cleaned_count > 0:
                logger.info(f"清理完成，共删除了 {cleaned_count} 个临时文件。")
            else:
                logger.info("没有需要清理的临时文件。")
        except Exception as e:
            logger.error(f"清理任务执行期间发生错误: {e}")

    async def _periodic_cleanup(self):
        """后台定时任务，用于周期性清理临时文件夹"""
        self._perform_cleanup()

        interval_seconds = self.cleanup_interval_hours * 3600
        while True:
            await asyncio.sleep(interval_seconds)
            self._perform_cleanup()

    async def _make_request(
            self,
            method: str,
            endpoint: str,
            json_data: Dict = None,
    ) -> None | bytes:
        """通用的异步请求方法"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(method.upper(), endpoint, json=json_data) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"请求失败，状态码: {response.status}, 错误信息: {error_text}")
                        return None
                    audio_bytes = await response.read()
                    return audio_bytes
        except aiohttp.ClientError as e:
            logger.error(f"网络请求时发生错误：{e}")
            return None
        except Exception as e:
            logger.error(f"发生未知错误：{e}")
            return None


    @filter.on_llm_response()
    async def on_llm_resp(self, event: AstrMessageEvent, resp: LLMResponse):
        """在LLM响应后，按概率将文本转为语音"""
        if random.random() > self.send_record_probability:
            return

        result_chain = resp.result_chain
        # MessageChain(chain=[Plain(type='Plain', text=' ', convert=True)], use_t2i_=None, type=None)

        original_text = result_chain.get_plain_text()

        # 解析LLM回复，提取纯语言信息用于TTS
        # 移除格式： （动作信息） 【附加信息】 [好感度标记]
        text_to_speak = re.sub(r"（.*?）|【.*?】|\[.*?\]", "", original_text).strip()

        # 长度判断
        if len(text_to_speak) > self.max_resp_text_len:
            return

        # 如果解析后没有可说的内容，则直接返回，发送原始文本
        if not text_to_speak:
            return

        # 尝试语音合成
        file_name = self.generate_file_name(event, text_to_speak)
        save_path = await self.tts_inference(text=text_to_speak, file_name=file_name)

        # 根据结果决定如何响应
        if save_path:
            # 成功：发送语音
            logger.info(f"LLM响应转语音成功，发送语音: {file_name}")

            chain = [
                Comp.Plain(original_text),
                Comp.Record(file=save_path)
            ]

            await event.send(MessageChain(chain))
            return

        else:
            # 失败：记录日志并放行，框架将发送原始文本
            logger.error("LLM响应转语音失败，将返回原始文本。")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("说")
    async def on_command(
            self, event: AstrMessageEvent, send_text: str | int | None = None
    ):
        """/说 xxx，直接调用TTS，发送合成后的语音"""
        if not send_text:
            yield event.plain_result("请在指令后输入要说的内容。")
            return

        text_to_speak = str(send_text)

        file_name = self.generate_file_name(event, text_to_speak)
        save_path = await self.tts_inference(text=text_to_speak, file_name=file_name)

        if save_path:
            chain = [Record.fromFileSystem(save_path)]
            yield event.chain_result(chain)
        else:
            logger.error("TTS指令任务执行失败！")
            yield event.plain_result("语音合成失败，请检查后台日志。")

    def generate_file_name(self, event: AstrMessageEvent, text: str) -> str:
        """生成文件名"""
        group_id = event.get_group_id() or "0"
        sender_id = event.get_sender_id() or "0"
        sanitized_text = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff\s]", "", text)
        limit_text = sanitized_text.strip()[:30]
        media_type = self.tts_core_params.get("response_format", "wav")
        file_name = f"{group_id}_{sender_id}_{limit_text}.{media_type}"
        return file_name

    async def tts_inference(self, text: str, file_name: str) -> str | None:
        """发送TTS请求，获取音频内容"""
        if not self.base_url:
            return None

        endpoint = f"{self.base_url}/v1/audio/speech"

        payload = {
            "model": self.tts_core_params.get("model", "tts-v4"),
            "input": text,
            "voice": self.tts_core_params.get("voice", ""),
            "response_format": self.tts_core_params.get("response_format", "wav"),
            "speed": self.tts_core_params.get("speed", 1.0),
            "other_params": self.tts_other_params.copy()
        }

        logger.debug(f"发送到TTS接口的请求体: {payload}")

        audio_bytes = await self._make_request(method="POST", endpoint=endpoint, json_data=payload)

        if audio_bytes:
            save_path = str(SAVED_AUDIO_DIR / file_name)
            with open(save_path, "wb") as audio_file:
                audio_file.write(audio_bytes)
            logger.info(f"成功生成语音文件: {save_path}")
            return save_path
        return None