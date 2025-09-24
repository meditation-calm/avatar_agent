import os
import uuid
from typing import Optional, Dict

from dotenv import load_dotenv
from loguru import logger

from src.chat_engine.common.client_handler_base import ClientHandlerBase
from src.chat_engine.contexts.session_context import SessionContext
from src.chat_engine.core.chat_session import ChatSession
from src.chat_engine.core.handler_manager import HandlerManager
from src.chat_engine.data_models.chat_engine_config_data import ChatEngineConfigModel, EngineChannelType
from src.chat_engine.data_models.session_info_data import SessionInfoData, IOQueueType
from src.engine_utils.directory_info import DirectoryInfo


"""
    1. 系统初始化和配置管理
        通过 initialize 方法初始化整个聊天引擎
        加载环境变量配置
        配置模型根目录路径
        初始化并加载所有处理器（通过 HandlerManager）
    2. 会话管理
        管理所有活跃的聊天会话，存储在 sessions 字典中
        通过 _create_session 方法创建新的聊天会话
        支持创建客户端会话（create_client_session）
        通过 stop_session 方法停止和清理指定会话
    3. 处理器协调
        持有 HandlerManager 实例，负责所有处理器的生命周期管理
        在创建会话时，将会话与启用的处理器进行关联
        特别处理客户端处理器的初始化逻辑
    4. 系统关闭和资源清理
        通过 shutdown 方法优雅地关闭整个引擎
        调用处理器管理器的销毁方法清理所有处理器资源

    工作流程
    初始化阶段：
        加载配置和环境变量
        初始化处理器管理器
        加载所有启用的处理器模块
    运行阶段：
        根据需要创建聊天会话
        为会话分配和配置相应的处理器
        管理会话的生命周期
    关闭阶段：
        清理所有会话
        销毁所有处理器
        释放系统资源
    """


class ChatEngine(object):
    def __init__(self):
        self.inited = False
        self.engine_config: Optional[ChatEngineConfigModel] = None
        self.handler_manager: HandlerManager = HandlerManager(self)
        self.sessions: Dict[str, ChatSession] = {}

    def initialize(self, engine_config: ChatEngineConfigModel, app=None, ui=None, parent_block=None):
        if self.inited:
            return

        load_dotenv()

        self.engine_config = engine_config
        if not os.path.isabs(engine_config.model_root):
            engine_config.model_root = os.path.join(DirectoryInfo.get_project_dir(), engine_config.model_root)
        self.handler_manager.initialize(engine_config)
        self.handler_manager.load_handlers(engine_config, app, ui, parent_block)
        self.inited = True

    def _create_session(self, session_info: SessionInfoData,
                        input_queues: Dict[EngineChannelType, IOQueueType],
                        output_queues: Dict[EngineChannelType, IOQueueType]):
        if not session_info.session_id:
            session_info.session_id = str(uuid.uuid4())
        if session_info.session_id in self.sessions:
            raise RuntimeError(f"session {session_info.session_id} already exists")

        session_context = SessionContext(session_info=session_info,
                                         input_queues=input_queues,
                                         output_queues=output_queues)

        session = ChatSession(session_context, self.engine_config)
        handlers = self.handler_manager.get_enabled_handler_registries()
        for registry in handlers:
            if isinstance(registry.handler, ClientHandlerBase):
                # client create_context and data_sink creation of handler is not called here,
                # they are created by its internal logic after every other handlers are ready.
                continue
            # 为会话准备非客户端处理器（客户端处理器特殊处理）
            session.prepare_handler(registry.handler, registry.base_info, registry.handler_config)
        self.sessions[session_info.session_id] = session
        return session

    def create_client_session(self, session_info: SessionInfoData, client_handler: ClientHandlerBase):
        # 当前不允许在一个会话中使用多个客户端。
        if session_info.session_id in self.sessions:
            msg = f"Session {session_info.session_id} already exists."
            raise RuntimeError(msg)

        session = self._create_session(session_info, {}, {})

        # 查找指定的客户端处理器注册信息
        registry = self.handler_manager.find_client_handler(client_handler)
        if registry is None:
            raise RuntimeError(f"client handler {client_handler} not found")

        # 为会话准备客户端处理器并返回会话和处理器环境
        handler_env = session.prepare_handler(client_handler, registry.base_info, registry.handler_config)
        return session, handler_env

    def stop_session(self, session_id: str):
        session = self.sessions.pop(session_id)
        if session is None:
            logger.error(f"Session {session_id} is not found.")
            return
        session.stop()
    
    def shutdown(self):
        logger.info("Shutting down chat engine...")
        self.handler_manager.destroy()
