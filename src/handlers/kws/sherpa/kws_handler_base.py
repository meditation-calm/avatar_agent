from typing import Optional

from pydantic import BaseModel, Field

from src.chat_engine.contexts.handler_context import HandlerContext
from src.chat_engine.data_models.chat_engine_config_data import HandlerBaseConfigModel


class KwsConfig(HandlerBaseConfigModel, BaseModel):
    model_name: str = Field(default="sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01")
    tokens: str = Field(default="tokens.txt")
    encoder: str = Field(default="encoder-epoch-12-avg-2-chunk-16-left-64.onnx")
    decoder: str = Field(default="decoder-epoch-12-avg-2-chunk-16-left-64.onnx")
    joiner: str = Field(default="joiner-epoch-12-avg-2-chunk-16-left-64.onnx")
    num_threads: int = Field(default=1)
    sample_rate: float = Field(default=16000)
    keywords_score: float = Field(default=1.0)  # 每个关键字标记的提升分数。越大越容易
    keywords_threshold: float = Field(default=0.25)  # 关键字的触发阈值（即概率）。越大更难触发。
    keywords_file: str = Field(default="keywords.txt")


class KwsContext(HandlerContext):
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.config: Optional[KwsConfig] = None
        self.sample_rate = 16000
        self.output_audios = []
        self.shared_states = None
