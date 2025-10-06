from typing import Optional

from pydantic import BaseModel, Field

from src.chat_engine.contexts.handler_context import HandlerContext
from src.chat_engine.data_models.chat_engine_config_data import HandlerBaseConfigModel
from src.engine_utils.general_slicer import SliceContext


class KwsConfig(HandlerBaseConfigModel, BaseModel):
    model_name: str = Field(default="iic/speech_charctc_kws_phone-xiaoyun")
    """ 推理阶段参数 """
    keywords: str = Field(default="小云小云")
    speaking_threshold: float = Field(default=0.5)


class KwsContext(HandlerContext):
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.config: Optional[KwsConfig] = None
        self.output_audios = []
        self.audio_slice_context = SliceContext.create_numpy_slice_context(
            slice_size=16000,
            slice_axis=0,
        )
        self.shared_states = None
