from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class EventType(str, Enum):
    """
    定义系统中可能发生的各种事件类型，继承自 str 和 Enum，使其既可以作为枚举使用，也可以作为字符串使用：
        EVT_START_AVATAR_SPEAKING：虚拟形象开始说话
        EVT_END_AVATAR_SPEAKING：虚拟形象结束说话
        EVT_START_HUMAN_SPEAKING：人类开始说话
        EVT_END_HUMAN_SPEAKING：人类结束说话
        EVT_HUMAN_TEXT：人类语音文本
        EVT_HUMAN_TEXT_END：人类语音文本结束
        EVT_AVATAR_TEXT：虚拟形象语音文本
        EVT_AVATAR_TEXT_END：虚拟形象语音文本结束
        EVT_SESSION_START：会话开始
        EVT_SESSION_STOP：会话停止
        EVT_INTERRUPT_SPEECH：语音中断
        EVT_SERVER_ERROR：服务器错误
    """
    EVT_START_AVATAR_SPEAKING = "start_avatar_speaking"
    EVT_END_AVATAR_SPEAKING = "end_avatar_speaking"
    EVT_START_HUMAN_SPEAKING = "start_human_speaking"
    EVT_END_HUMAN_SPEAKING = "end_human_speaking"
    EVT_HUMAN_TEXT = "human_speech_text"
    EVT_HUMAN_TEXT_END = "human_speech_text_end"
    EVT_AVATAR_TEXT = "avatar_speech_text"
    EVT_AVATAR_TEXT_END = "avatar_speech_text_end"
    EVT_SESSION_START = "session_start"
    EVT_SESSION_STOP = "session_stop"
    EVT_INTERRUPT_SPEECH = "interrupt_speech"
    EVT_SERVER_ERROR = "server_error"


class EventEmbeddingDataType(str, Enum):
    """
    定义事件数据的嵌入类型，指定事件携带的数据格式：
        NOT_SET：未设置数据类型
        TEXT：纯文本格式
        JSON：JSON格式
        BASE64：Base64编码格式
    """
    NOT_SET = "not_set"
    TEXT = "text"
    JSON = "json"
    BASE64 = "base64"


class EventData(BaseModel):
    """
    事件数据模型，继承自 Pydantic 的 BaseModel，用于表示和验证事件数据：
        event_type：事件类型，使用 EventType 枚举
        event_subtype：事件子类型，字符串形式的更细粒度分类
        event_data_type：事件数据类型，使用 EventEmbeddingDataType 枚举
        event_data：事件携带的具体数据，以字符串形式存储
        event_time：事件时间戳
            None 表示即时事件
            -1 表示由接收方确定事件时间
        event_time_unit：事件时间单位
    """
    event_type: Optional[EventType] = Field(default=None)
    event_subtype: Optional[str] = Field(default=None)
    event_data_type: Optional[EventEmbeddingDataType] = Field(default=None)
    event_data: Optional[str] = Field(default=None)
    # None means instant event, -1 means event time need to be determined by receiver
    event_time: Optional[int] = Field(default=None)
    event_time_unit: Optional[int] = Field(default=None)

    def is_valid(self):
        """验证事件是否有效（主要检查事件类型是否设置）"""
        return self.event_type is not None
