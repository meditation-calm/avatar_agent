import os
import queue
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from pydantic import BaseModel, Field

from src.chat_engine.contexts.handler_context import HandlerContext
from src.chat_engine.data_models.chat_engine_config_data import HandlerBaseConfigModel
from src.engine_utils.directory_info import DirectoryInfo


class TTSConfig(HandlerBaseConfigModel, BaseModel):
    model_name: str = Field(default=None)
    ref_audio_path: str = Field(default=None)
    ref_audio_text: str = Field(default=None)
    spk_id: str = Field(default="中文女")
    speed: float = Field(default=1.0)
    sample_rate: int = Field(default=24000)
    process_num: int = Field(default=1)


@dataclass
class HandlerTask:
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    result_queue: queue.Queue = field(default_factory=queue.Queue)
    speech_id: str = field(default=None)
    speech_end: bool = field(default=False)


class TTSContext(HandlerContext):
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.config: Optional[TTSConfig] = None
        self.input_text = ''
        self.dump_audio = False
        self.audio_dump_file = None
        if self.dump_audio:
            localtime = time.localtime()
            dump_file_path = os.path.join(DirectoryInfo.get_project_dir(), 'cache',
                                          f"dump_avatar_audio_{self.session_id}_{localtime.tm_hour}_{localtime.tm_min}.pcm")
            self.audio_dump_file = open(dump_file_path, "wb")

        self.task_queue: deque[HandlerTask] = deque()
        self.task_consumer_thread = None
