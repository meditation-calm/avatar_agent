import re
from typing import Dict, Optional, cast

from loguru import logger
from openai import OpenAI, APIStatusError
from pydantic import BaseModel
from abc import ABC

from src.chat_engine.common.handler_base import HandlerBase, HandlerBaseInfo, HandlerDetail, HandlerDataInfo
from src.chat_engine.contexts.handler_context import HandlerContext
from src.chat_engine.contexts.session_context import SessionContext
from src.chat_engine.data_models.chat_data.chat_data_model import ChatData
from src.chat_engine.data_models.chat_data_type import ChatDataType
from src.chat_engine.data_models.chat_engine_config_data import ChatEngineConfigModel
from src.chat_engine.data_models.runtime_data.data_bundle import DataBundleDefinition, DataBundleEntry, DataBundle
from src.handlers.llm.chat_history_manager import ChatHistory, HistoryMessage
from src.handlers.llm.openai.llm_handler_base import LLMConfig, LLMContext


class LLMHandlerOpenAi(HandlerBase, ABC):
    def __init__(self):
        super().__init__()

    def get_handler_info(self) -> HandlerBaseInfo:
        return HandlerBaseInfo(
            name="LLM_OpenAI",
            config_model=LLMConfig,
        )

    def get_handler_detail(self, session_context: SessionContext,
                           context: HandlerContext) -> HandlerDetail:
        definition = DataBundleDefinition()
        definition.add_entry(DataBundleEntry.create_text_entry("avatar_text"))
        inputs = {
            ChatDataType.HUMAN_TEXT: HandlerDataInfo(
                type=ChatDataType.HUMAN_TEXT,
            ),
            # ChatDataType.CAMERA_VIDEO: HandlerDataInfo(
            #     type=ChatDataType.CAMERA_VIDEO,
            # ),
        }
        outputs = {
            ChatDataType.AVATAR_TEXT: HandlerDataInfo(
                type=ChatDataType.AVATAR_TEXT,
                definition=definition,
            )
        }
        return HandlerDetail(
            inputs=inputs, outputs=outputs,
        )

    def load(self, engine_config: ChatEngineConfigModel, handler_config: Optional[BaseModel] = None):
        if isinstance(handler_config, LLMConfig):
            if handler_config.api_key is None or len(handler_config.api_key) == 0:
                error_message = 'api_key is required in config/xxx.yaml, when use handler_llm'
                logger.error(error_message)
                raise ValueError(error_message)
            if handler_config.api_url is None or len(handler_config.api_url) == 0:
                error_message = 'api_url is required in config/xxx.yaml, when use handler_llm'
                logger.error(error_message)
                raise ValueError(error_message)

    def create_context(self, session_context, handler_config=None):
        if not isinstance(handler_config, LLMConfig):
            handler_config = LLMConfig()
        context = LLMContext(session_context.session_info.session_id)
        context.config = handler_config
        context.system_message = {'role': 'system', 'content': handler_config.system_prompt}
        context.history = ChatHistory(history_length=handler_config.history_length)
        context.client = OpenAI(
            api_key=handler_config.api_key,
            base_url=handler_config.api_url,
        )
        return context

    def start_context(self, session_context, handler_context):
        pass

    def handle(self, context: HandlerContext, inputs: ChatData,
               output_definitions: Dict[ChatDataType, HandlerDataInfo]):
        output_definition = output_definitions.get(ChatDataType.AVATAR_TEXT).definition
        context = cast(LLMContext, context)
        if inputs.type != ChatDataType.HUMAN_TEXT:
            return
        speech_id = inputs.data.get_meta("speech_id", context.session_id)
        text = inputs.data.get_main_data()
        if text is not None:
            context.input_texts += text

        text_end = inputs.data.get_meta("human_text_end", False)
        if not text_end:
            return

        chat_text = context.input_texts
        chat_text = re.sub(r"<\|.*?\|>", "", chat_text)
        if len(chat_text) < 1:
            return
        """ 加载历史记忆组建openai messages """
        current_content = context.history.generate_next_messages(chat_text, None)
        logger.info(f'llm openai input {context.config.model_name} {chat_text} {current_content} ')
        try:
            """ openai模型调用 """
            completion = context.client.chat.completions.create(
                model=context.config.model_name,
                messages=[context.system_message,] + current_content,
                stream=True,
                stream_options={"include_usage": True}
            )
            context.input_texts = ''
            context.output_texts = ''
            for chunk in completion:
                if chunk and chunk.choices and chunk.choices[0] and chunk.choices[0].delta.content:
                    """ 持续输出llm响应数据 """
                    output_text = chunk.choices[0].delta.content
                    context.output_texts += output_text
                    logger.info(output_text)
                    output = DataBundle(output_definition)
                    output.set_main_data(output_text)
                    output.add_meta("avatar_text_end", False)
                    output.add_meta("speech_id", speech_id)
                    yield output
            logger.info(f'llm openai output {context.config.model_name} {chat_text} {context.output_texts} ')
            context.history.add_message(HistoryMessage(role="human", content=chat_text))
            context.history.add_message(HistoryMessage(role="avatar", content=context.output_texts))
        except Exception as e:
            logger.error(f'llm openai error {context.config.model_name} {chat_text} {e} ')
            response = ''
            if isinstance(e, APIStatusError):
                response = e.body
                if isinstance(response, dict) and "message" in response:
                    response = f"{response['message']}"
            output = DataBundle(output_definition)
            output.set_main_data(response)
            output.add_meta("avatar_text_end", False)
            output.add_meta("speech_id", speech_id)
            yield output
        context.input_texts = ''
        context.output_texts = ''
        """ 标志llm结束响应 """
        end_output = DataBundle(output_definition)
        end_output.set_main_data('')
        end_output.add_meta("avatar_text_end", True)
        end_output.add_meta("speech_id", speech_id)
        yield end_output

    def destroy_context(self, context: HandlerContext):
        pass
