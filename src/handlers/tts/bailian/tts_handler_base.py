import os
import time
from typing import Optional

from pydantic import BaseModel, Field

from src.chat_engine.contexts.handler_context import HandlerContext
from src.chat_engine.data_models.chat_engine_config_data import HandlerBaseConfigModel
from src.engine_utils.directory_info import DirectoryInfo


class TTSConfig(HandlerBaseConfigModel, BaseModel):
    model_name: str = Field(default="cosyvoice-v1")
    api_key: str = Field(default=os.getenv("DASHSCOPE_API_KEY"))
    voice: str = Field(default="longxiaochun")  # 说话人音色。
    volume: int = Field(default=50)  # 朗读音量，范围是0~100，默认50。
    speech_rate: float = Field(default=1.0)  # 朗读语速，范围是-500~500，默认是1.0。
    pitch_rate: float = Field(default=1.0)  # 朗读语调，范围是-500~500，默认是0。
    sample_rate: int = Field(default=24000)  # 音频采样率，默认为16000 Hz。


class TTSContext(HandlerContext):
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.config: Optional[TTSConfig] = None
        self.shared_states = None
        self.input_text = ''
        self.synthesizer = None
        self.synthesizer_idx = 0  # 合成索引
        self.ignore_text = False
