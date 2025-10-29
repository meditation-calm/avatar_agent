import os
from typing import Optional

from pydantic import BaseModel, Field

from src.chat_engine.contexts.handler_context import HandlerContext
from src.chat_engine.data_models.chat_engine_config_data import HandlerBaseConfigModel
from src.engine_utils.directory_info import DirectoryInfo
from src.engine_utils.general_slicer import SliceContext


class ASRConfig(HandlerBaseConfigModel, BaseModel):
    model_name: str = Field(default="iic/SenseVoiceSmall")
    """ 推理阶段参数 """
    language: str = Field(default="auto")  # "zh"/"en"/"yue"/"ja"/"ko"/"auto"
    use_itn: bool = Field(default=False)
    batch_size_s: int = Field(default=10)


class ASRContext(HandlerContext):
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.config: Optional[ASRConfig] = None
        self.output_audios = []
        self.audio_slice_context = SliceContext.create_numpy_slice_context(
            slice_size=16000,
            slice_axis=0,
        )

        self.dump_audio = True
        self.audio_dump_file = None
        if self.dump_audio:
            dump_file_path = os.path.join(DirectoryInfo.get_cache_dir(), "dump_talk_audio.pcm")
            self.audio_dump_file = open(dump_file_path, "wb")
        self.shared_states = None
