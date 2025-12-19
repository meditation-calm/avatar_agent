from typing import Optional

from pydantic import BaseModel, Field

from src.chat_engine.contexts.handler_context import HandlerContext
from src.chat_engine.data_models.chat_engine_config_data import HandlerBaseConfigModel


class DifyConfig(HandlerBaseConfigModel, BaseModel):
    api_url: str = Field(default="http://127.0.0.1/v1")
    api_key: str = Field(default="")
    response_mode: str = Field(default="streaming")
    timeout: int = Field(default=30)


class DifyContext(HandlerContext):
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.config: Optional[DifyConfig] = None
        self.input_texts = ""
        self.output_texts = ""
        self.current_image = None
        self.conversation_id = None
