from typing import Dict, Optional, List, Union

from pydantic import BaseModel, Field

from src.chat_engine.data_models.chat_data_type import ChatDataType
from src.chat_engine.common.engine_channel_type import EngineChannelType


class HandlerBaseConfigModel(BaseModel):
    """
    配置每个独立处理器的基本行为，控制是否启用、指定模块路径以及设置并发限制。
        enabled：布尔值，表示该处理器是否启用，默认为 True
        module：可选字符串，指定处理器模块的路径或名称，默认为 None
        concurrent_limit：整数，表示该处理器的并发限制，默认为 1
    """
    enabled: bool = Field(default=True)
    module: Optional[str] = Field(default=None)
    concurrent_limit: int = Field(default=1)


class ChatEngineOutputSource(BaseModel):
    """
    配置引擎的输出源，指定哪些处理器负责生成特定类型的数据输出
        handler：可选的字符串或字符串列表，指定提供输出的处理器名称
        type：ChatDataType 枚举值，指定输出的数据类型
    """
    handler: Optional[Union[str, List[str]]]
    type: ChatDataType


class ChatEngineConfigModel(BaseModel):
    """
    主配置模型
        model_root：字符串，指定模型文件的根目录路径，默认为空字符串
        concurrent_limit：整数，全局并发限制，默认为 1
        handler_search_path：字符串列表，指定处理器模块的搜索路径，默认为空列表
        handler_configs：可选字典，键为处理器名称，值为处理器配置字典，默认为 None
        outputs：字典，键为 EngineChannelType，值为 ChatEngineOutputSource，定义引擎输出配置，默认为空字典
        turn_config：可选字典，回合配置，默认为 None
    """
    model_root: str = ""
    concurrent_limit: int = Field(default=1)
    handler_search_path: List[str] = Field(default_factory=list)
    handler_configs: Optional[Dict[str, Dict]] = None
    outputs: Dict[EngineChannelType, ChatEngineOutputSource] = Field(default_factory=dict)
    turn_config: Optional[Dict] = Field(default=None)
