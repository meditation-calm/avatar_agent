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
    voice: str = Field(default="longxiaochun")
    volume: int = Field(default=50)
    speed: float = Field(default=1.0)
    pitch_rate: float = Field(default=1.0)
    sample_rate: int = Field(default=24000)


class TTSContext(HandlerContext):
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.config: Optional[TTSConfig] = None
        self.input_text = ''
        self.dump_audio = False
        self.audio_dump_file = None
        if self.dump_audio:
            localtime = time.localtime()
            dump_file_path = os.path.join(DirectoryInfo.get_cache_dir(),
                                          f"dump_avatar_audio_{self.session_id}_{localtime.tm_hour}_{localtime.tm_min}.pcm")
            self.audio_dump_file = open(dump_file_path, "wb")
        self.synthesizer = None
