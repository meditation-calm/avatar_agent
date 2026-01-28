import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union, Tuple, Optional

import gradio
import numpy as np
from fastapi import FastAPI
from loguru import logger

from src.chat_engine.common.engine_channel_type import EngineChannelType
from src.chat_engine.common.handler_base import HandlerBase
from src.chat_engine.contexts.handler_context import HandlerContext
from src.chat_engine.contexts.session_context import SessionContext
from src.chat_engine.data_models.chat_data.chat_data_model import ChatData
from src.chat_engine.data_models.chat_signal import ChatSignal
from src.chat_engine.data_models.session_info_data import SessionInfoData


class ClientSessionDelegate(ABC):
    """
    客户端会话代理抽象基类
        为客户端会话提供标准化的数据交互接口
        实现客户端与引擎之间的数据双向传输
        提供会话控制和状态管理功能
    """
    @abstractmethod
    async def get_data(self, modality: EngineChannelType, timeout: Optional[float] = 0.1) -> Optional[ChatData]:
        """异步获取指定模态的数据"""
        pass

    @abstractmethod
    def put_data(self, modality: EngineChannelType, data: Union[np.ndarray, str],
                 timestamp: Optional[Tuple[int, int]] = None,
                 samplerate: Optional[int] = None, loopback: bool = False):
        """向指定模态发送数据"""
        pass

    @abstractmethod
    def get_timestamp(self) -> Tuple[int, int]:
        """获取当前时间戳"""
        pass

    @abstractmethod
    def emit_signal(self, signal: ChatSignal):
        """发送控制信号"""
        pass

    @abstractmethod
    def clear_data(self):
        """清空数据"""
        pass


class ClientHandlerDelegate:
    """
    客户端处理器代理类：
        作为客户端处理器与聊天引擎之间的桥梁
        管理会话的创建、销毁和查找
        协调客户端与引擎的交互
    """
    def __init__(self, engine_ref, client_handler):
        self.engine_ref = engine_ref
        self.client_handler_ref = weakref.ref(client_handler)

        self.session_delegates = {}

    def start_session(self, session_id: str, **kwargs) -> ClientSessionDelegate:
        """启动新会话"""
        logger.info(f"Starting session {session_id}")
        engine = self.engine_ref()
        handler = self.client_handler_ref()
        assert engine is not None
        assert handler is not None

        kwargs["session_id"] = session_id
        session_info = SessionInfoData.model_validate(kwargs)

        session, handler_env = engine.create_client_session(session_info, handler)
        session.start()
        if handler_env.handler_info.client_session_delegate_class is None:
            msg = f"Client handler {handler_env.handler_info.handler_name} does not provide a session delegate."
            raise RuntimeError(msg)
        session_delegate = handler_env.handler_info.client_session_delegate_class()
        # 注入 ChatSession 引用，方便在 datachannel 侧触发会话级信号/打断
        try:
            setattr(session_delegate, "chat_session", session)
        except Exception:
            pass
        handler_env.handler.on_setup_session_delegate(session.session_context, handler_env.context, session_delegate)
        self.session_delegates[session_id] = session_delegate
        return session_delegate

    def stop_session(self, session_id: str):
        """停止指定会话"""
        engine = self.engine_ref()
        assert engine is not None
        engine.stop_session(session_id)
        self.session_delegates.pop(session_id)

    def find_session_delegate(self, session_id: str):
        """查找会话代理"""
        return self.session_delegates.get(session_id)


@dataclass
class ClientHandlerInfo:
    """
    客户端处理器信息数据类：
        存储客户端处理器的配置信息
        定义会话代理类类型
    """
    session_delegate_class: type[ClientSessionDelegate]


class ClientHandlerBase(HandlerBase, ABC):
    """
    客户端处理器基类，继承自 HandlerBase：
        为所有客户端处理器提供统一的基础框架
        集成客户端处理逻辑和引擎交互机制
        提供Web应用和会话代理设置接口
    """
    def __init__(self):
        super().__init__()
        self.handler_delegate = ClientHandlerDelegate(self.engine, self)

    def on_before_register(self):
        self.handler_delegate.engine_ref = self.engine

    @abstractmethod
    def on_setup_app(self, app: FastAPI, ui: gradio.blocks.Block, parent_block: Optional[gradio.blocks.Block]=None):
        """设置Web应用接口"""
        pass

    @abstractmethod
    def on_setup_session_delegate(self, session_context: SessionContext, handler_context: HandlerContext,
                                  session_delegate: ClientSessionDelegate):
        """设置会话代理"""
        pass
