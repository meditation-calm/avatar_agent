import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict

from src.chat_engine.data_models.chat_data_type import ChatDataType
from src.chat_engine.contexts.handler_context import HandlerContext
from src.chat_engine.contexts.session_context import SessionContext
from src.chat_engine.data_models.chat_data.chat_data_model import ChatData
from src.chat_engine.data_models.chat_engine_config_data import ChatEngineConfigModel, HandlerBaseConfigModel
from src.chat_engine.data_models.runtime_data.data_bundle import DataBundleDefinition


class ChatDataConsumeMode(Enum):
    """
    定义聊天数据的消费模式：
        ONCE = -1：数据只被消费一次
        DEFAULT = 0：默认消费模式（允许多次消费）
    """
    ONCE = -1
    DEFAULT = 0


@dataclass
class HandlerBaseInfo:
    """
    存储处理器的基本信息：
        name：处理器名称
        config_model：处理器配置模型类型
        client_session_delegate_class：客户端会话代理类
        load_priority：加载优先级（数值越小优先级越高）
    """
    name: Optional[str] = None
    config_model: Optional[type[HandlerBaseConfigModel]] = None
    client_session_delegate_class: Optional[type] = None
    # Handler load priority, the smaller, the higher
    load_priority: int = 0


@dataclass
class HandlerDataInfo:
    """
    描述处理器输入/输出数据的信息：
        type：聊天数据类型
        definition：数据包定义
        input_priority：输入优先级
        input_consume_mode：输入消费模式
        实现了 __lt__ 方法用于排序比较
    """
    type: ChatDataType = ChatDataType.NONE
    definition: Optional[DataBundleDefinition] = None
    input_priority: int = 0
    input_consume_mode: ChatDataConsumeMode = ChatDataConsumeMode.DEFAULT

    def __lt__(self, other):
        if self.input_priority == other.input_priority:
            return self.type.value < other.type.value
        return self.input_priority < other.input_priority


@dataclass
class HandlerDetail:
    """
    描述处理器的详细信息：
        inputs：描述该处理器可以接收的数据类型，键为 ChatDataType，值为 HandlerDataInfo
        outputs：描述该处理器可以产生的数据类型，键为 ChatDataType，值为 HandlerDataInfo
    """
    inputs: Dict[ChatDataType, HandlerDataInfo] = field(default_factory=dict)
    outputs: Dict[ChatDataType, HandlerDataInfo] = field(default_factory=dict)


class HandlerBase(ABC):
    """
        所有处理器的抽象基类，定义了处理器必须实现的接口：
    """
    def __init__(self):
        # 对引擎的弱引用，避免循环引用
        self.engine: Optional[weakref.ReferenceType] = None
        # 处理器根目录路径
        self.handler_root: Optional[str] = None

    def on_before_register(self):
        # 在注册前调用的钩子方法
        pass

    @abstractmethod
    def get_handler_info(self) -> HandlerBaseInfo:
        # 返回处理器基本信息
        pass

    @abstractmethod
    def load(self, engine_config: ChatEngineConfigModel, handler_config: Optional[HandlerBaseConfigModel] = None):
        # 加载处理器，根据配置进行初始化
        pass

    @abstractmethod
    def create_context(self, session_context: SessionContext,
                       handler_config: Optional[HandlerBaseConfigModel] = None) -> HandlerContext:
        # 为会话创建处理器上下文
        pass

    @abstractmethod
    def start_context(self, session_context: SessionContext, handler_context: HandlerContext):
        # 启动处理器上下文
        pass

    @abstractmethod
    def get_handler_detail(self, session_context: SessionContext,
                           context: HandlerContext) -> HandlerDetail:
        # 获取处理器详细信息（输入输出定义）
        pass

    @abstractmethod
    def handle(self, context: HandlerContext, inputs: ChatData,
               output_definitions: Dict[ChatDataType, HandlerDataInfo]):
        # 处理输入数据并生成输出
        pass

    @abstractmethod
    def destroy_context(self, context: HandlerContext):
        # 销毁处理器上下文
        pass

    def interrupt(self, context: HandlerContext):
        """
        处理打断信号：
            当收到打断信号时，处理器应该停止当前处理任务
            清理相关资源，准备接收新的输入
            这是一个可选实现的方法，默认不做任何操作
        """
        # 默认实现为空，子类可以重写此方法来实现打断逻辑
        pass

    def destroy(self):
        # 处理器销毁时调用的方法
        pass
