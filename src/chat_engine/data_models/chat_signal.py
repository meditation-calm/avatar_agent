from typing import Optional

from pydantic import BaseModel, Field

from src.chat_engine.data_models.chat_data_type import ChatDataType
from src.chat_engine.data_models.chat_signal_type import ChatSignalType, ChatSignalSourceType


class ChatSignal(BaseModel):
    """传递控制信号"""
    type: Optional[ChatSignalType] = Field(default=None)
    stream_type: Optional[ChatDataType] = Field(default=None)
    source_type: Optional[ChatSignalSourceType] = Field(default=None)
    source_name: str = Field(default="")
