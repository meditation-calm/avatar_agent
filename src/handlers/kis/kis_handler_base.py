from typing import Optional, List
from pydantic import BaseModel, Field

from src.chat_engine.contexts.handler_context import HandlerContext
from src.chat_engine.data_models.chat_engine_config_data import HandlerBaseConfigModel


class KISConfig(HandlerBaseConfigModel, BaseModel):
    """KIS (Keyword-based Interrupt Switch) 配置"""
    model_name: str = Field(default="sherpa_kws")
    tokens: str = Field(default="tokens.txt")
    encoder: str = Field(default="encoder-epoch-12-avg-2-chunk-16-left-64.onnx")
    decoder: str = Field(default="decoder-epoch-12-avg-2-chunk-16-left-64.onnx")
    joiner: str = Field(default="joiner-epoch-12-avg-2-chunk-16-left-64.onnx")
    num_threads: int = Field(default=1)
    sample_rate: float = Field(default=16000)
    keywords_score: float = Field(default=1.0)  # 每个关键字标记的提升分数。越大越容易
    keywords_threshold: float = Field(default=0.25)  # 关键字的触发阈值（即概率）。越大更难触发。
    interrupt_keywords: List[str] = Field(default=["小云小云"])  # 打断关键词列表
    keywords_file: str = Field(default="keywords.txt")


class KISContext(HandlerContext):
    """KIS handler 上下文"""
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.config: Optional[KISConfig] = None
        self.sample_rate = 16000
        self.output_audios = []
        self.shared_states = None
        self.interrupt_pending = False  # 是否等待前端确认打断
        self.interrupt_keyword_detected = False  # 是否检测到打断关键词


