import re
from typing import Dict, Optional, cast
from loguru import logger
from pydantic import BaseModel
from abc import ABC
from src.chat_engine.contexts.handler_context import HandlerContext
from src.chat_engine.data_models.chat_engine_config_data import ChatEngineConfigModel, HandlerBaseConfigModel
from src.chat_engine.common.handler_base import HandlerBase, HandlerBaseInfo, HandlerDataInfo, HandlerDetail
from src.chat_engine.data_models.chat_data.chat_data_model import ChatData
from src.chat_engine.data_models.chat_data_type import ChatDataType
from src.chat_engine.contexts.session_context import SessionContext
from src.chat_engine.data_models.runtime_data.data_bundle import DataBundle, DataBundleDefinition, DataBundleEntry
from src.handlers.llm.dify.dify_request import DifyRequest
from src.handlers.llm.dify.llm_handler_base import DifyConfig, DifyContext


class LLMHandler(HandlerBase, ABC):
    def __init__(self):
        super().__init__()

    def get_handler_info(self) -> HandlerBaseInfo:
        return HandlerBaseInfo(
            config_model=DifyConfig,
        )

    def get_handler_detail(self, session_context: SessionContext,
                           context: HandlerContext) -> HandlerDetail:
        definition = DataBundleDefinition()
        definition.add_entry(DataBundleEntry.create_text_entry("avatar_text"))
        event_definition = DataBundleDefinition()
        event_definition.add_entry(DataBundleEntry.create_text_entry("human_event"))
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
            ),
            ChatDataType.HUMAN_EVENT: HandlerDataInfo(
                type=ChatDataType.HUMAN_EVENT,
                definition=event_definition,
            ),
        }
        return HandlerDetail(
            inputs=inputs, outputs=outputs,
        )

    def load(self, engine_config: ChatEngineConfigModel, handler_config: Optional[BaseModel] = None):
        if isinstance(handler_config, DifyConfig):
            if handler_config.api_key is None or len(handler_config.api_key) == 0:
                raise ValueError('api_key is required in config/xxx.yaml, when use handler_dify')

    def create_context(self, session_context, handler_config=None):
        if not isinstance(handler_config, DifyConfig):
            handler_config = DifyConfig()
        context = DifyContext(session_context.session_info.session_id)
        context.config = handler_config
        return context

    def start_context(self, session_context, handler_context):
        pass

    def handle(self, context: HandlerContext, inputs: ChatData,
               output_definitions: Dict[ChatDataType, HandlerDataInfo]):
        output_definition = output_definitions.get(ChatDataType.AVATAR_TEXT).definition
        event_definition = output_definitions.get(ChatDataType.HUMAN_EVENT).definition
        context = cast(DifyContext, context)
        if inputs.type != ChatDataType.HUMAN_TEXT:
            return
        speech_id = inputs.data.get_meta("speech_id", context.session_id)
        text = inputs.data.get_main_data()
        if text is not None:
            context.input_texts += text

        text_end = inputs.data.get_meta("human_text_end", False)
        if not text_end:
            return

        event = DataBundle(event_definition)
        event.set_main_data({"handler": "llm", "event": "start"})
        context.submit_data(ChatData(type=ChatDataType.HUMAN_EVENT, data=event))

        chat_text = context.input_texts
        chat_text = re.sub(r"<\|.*?\|>", "", chat_text)
        if len(chat_text) < 1:
            return

        try:
            for output_text in DifyRequest.chat_messages(context, chat_text,
                        [context.current_image] if context.current_image is not None else []):
                if output_text:
                    context.output_texts += output_text
                    logger.info(output_text)
                    output = DataBundle(output_definition)
                    output.set_main_data(output_text)
                    output.add_meta("avatar_text_end", False)
                    output.add_meta("speech_id", speech_id)
                    context.submit_data(ChatData(type=ChatDataType.AVATAR_TEXT, data=output))
        except Exception as e:
            logger.error(f"Error processing Dify response: {str(e)}")
            error_message = f"Error: {str(e)}"
            output = DataBundle(output_definition)
            output.set_main_data(error_message)
            output.add_meta("avatar_text_end", False)
            output.add_meta("speech_id", speech_id)
            context.submit_data(ChatData(type=ChatDataType.AVATAR_TEXT, data=output))

        context.input_texts = ''
        context.output_texts = ''
        context.current_image = None
        end_output = DataBundle(output_definition)
        end_output.set_main_data('')
        end_output.add_meta("avatar_text_end", True)
        end_output.add_meta("speech_id", speech_id)
        context.submit_data(ChatData(type=ChatDataType.AVATAR_TEXT, data=end_output))
        """ 结束事件 """
        event = DataBundle(event_definition)
        event.set_main_data({"handler": "llm", "event": "end"})
        context.submit_data(ChatData(type=ChatDataType.HUMAN_EVENT, data=event))

    def interrupt(self, context: HandlerContext):
        """处理打断信号：清空输入文本缓存"""
        context = cast(DifyContext, context)
        logger.info("LLM: Interrupt received, clearing input text")
        context.input_texts = ''
        context.current_image = None

    def destroy_context(self, context: HandlerContext):
        pass
