import asyncio
from typing import Dict, Optional, Union, Tuple
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, Field

from src.chat_engine.common.client_handler_base import ClientSessionDelegate
from src.chat_engine.common.engine_channel_type import EngineChannelType
from src.chat_engine.contexts.handler_context import HandlerContext
from src.chat_engine.data_models.chat_data.chat_data_model import ChatData
from src.chat_engine.data_models.chat_data_type import ChatDataType
from src.chat_engine.data_models.chat_engine_config_data import HandlerBaseConfigModel
from src.chat_engine.data_models.chat_signal import ChatSignal
from src.chat_engine.data_models.runtime_data.data_bundle import DataBundleDefinition, DataBundle


class RtcClientSessionDelegate(ClientSessionDelegate):
    """
    rtc客户端会话代理，负责处理客户端与引擎间的数据交互：
        timestamp_generator：时间戳生成器
        data_submitter：数据提交器
        shared_states：共享状态
        output_queues：输出队列字典（音频、视频、文本）
        input_data_definitions：输入数据定义
        modality_mapping：模态映射（引擎通道类型到聊天数据类型）
    """

    def __init__(self):
        self.timestamp_generator = None
        self.data_submitter = None
        self.shared_states = None
        # 由 ClientHandlerDelegate.start_session 注入，指向当前会话的 ChatSession
        # 这里不做强类型依赖，避免循环 import
        self.chat_session = None
        self.output_queues = {
            EngineChannelType.AUDIO: asyncio.Queue(),
            EngineChannelType.VIDEO: asyncio.Queue(),
            EngineChannelType.TEXT: asyncio.Queue(),
        }
        self.input_data_definitions: Dict[EngineChannelType, DataBundleDefinition] = {}
        self.modality_mapping = {
            EngineChannelType.AUDIO: ChatDataType.MIC_AUDIO,
            EngineChannelType.VIDEO: ChatDataType.CAMERA_VIDEO,
            EngineChannelType.TEXT: ChatDataType.HUMAN_TEXT,
        }

    async def get_data(self, modality: EngineChannelType, timeout: Optional[float] = 0.1) -> Optional[ChatData]:
        data_queue = self.output_queues.get(modality)
        if data_queue is None:
            return None
        if timeout is not None and timeout > 0:
            try:
                data = await asyncio.wait_for(data_queue.get(), timeout)
            except asyncio.TimeoutError:
                return None
        else:
            data = await data_queue.get()
        return data

    def put_data(self, modality: EngineChannelType, data: Union[np.ndarray, str],
                 timestamp: Optional[Tuple[int, int]] = None, samplerate: Optional[int] = None, loopback: bool = False):
        if timestamp is None:
            timestamp = self.get_timestamp()
        if self.data_submitter is None:
            return
        definition = self.input_data_definitions.get(modality)
        chat_data_type = self.modality_mapping.get(modality)
        if chat_data_type is None or definition is None:
            return
        data_bundle = DataBundle(definition)
        if modality == EngineChannelType.AUDIO:
            data_bundle.set_main_data(data.squeeze()[np.newaxis, ...])
        elif modality == EngineChannelType.VIDEO:
            data_bundle.set_main_data(data[np.newaxis, ...])
        elif modality == EngineChannelType.TEXT:
            data_bundle.add_meta('human_text_end', True)
            data_bundle.add_meta('speech_id', str(uuid4()))
            data_bundle.set_main_data(data)
        else:
            return
        chat_data = ChatData(
            source="client",
            type=chat_data_type,
            data=data_bundle,
            timestamp=timestamp,
        )
        self.data_submitter.submit(chat_data)
        if loopback:
            self.output_queues[modality].put_nowait(chat_data)

    def get_timestamp(self):
        return self.timestamp_generator()

    def emit_signal(self, signal: ChatSignal):
        # 将信号转发给当前会话（如果存在）
        if self.chat_session is None:
            return
        try:
            self.chat_session.emit_signal(signal)
        except Exception:
            # 避免信号处理影响主流程
            return

    def clear_data(self):
        for data_queue in self.output_queues.values():
            while not data_queue.empty():
                data_queue.get_nowait()


class ClientRtcConfigModel(HandlerBaseConfigModel, BaseModel):
    """
    RTC客户端配置模型：
        connection_ttl：连接生存时间（默认900秒）
        turn_config：TURN服务器配置
    """
    connection_ttl: int = Field(default=900)
    turn_config: Optional[Dict] = Field(default=None)


class ClientRtcContext(HandlerContext):
    """
    RTC客户端上下文：
        config：RTC配置
        client_session_delegate：客户端会话代理
    """
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.config: Optional[ClientRtcConfigModel] = None
        self.client_session_delegate: Optional[RtcClientSessionDelegate] = None
