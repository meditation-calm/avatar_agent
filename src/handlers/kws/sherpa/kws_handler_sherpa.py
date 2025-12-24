import os
from abc import ABC
from typing import cast, Dict, Optional

import numpy as np
import sherpa_onnx
from loguru import logger

from src.chat_engine.common.handler_base import HandlerBase, HandlerBaseInfo, HandlerDetail, HandlerDataInfo
from src.chat_engine.contexts.handler_context import HandlerContext
from src.chat_engine.contexts.session_context import SessionContext
from src.chat_engine.data_models.chat_data.chat_data_model import ChatData
from src.chat_engine.data_models.chat_data_type import ChatDataType
from src.chat_engine.data_models.chat_engine_config_data import ChatEngineConfigModel
from src.chat_engine.data_models.runtime_data.data_bundle import DataBundleDefinition, DataBundleEntry, DataBundle
from src.handlers.kws.sherpa.kws_handler_base import KwsConfig, KwsContext


class KWSHandler(HandlerBase, ABC):
    """
    用于处理音频关键词检测（KWS）。
    加载 Silero VAD 模型，处理音频数据，检测语音活动，并生成相应的输出数据。
    """
    def __init__(self):
        super().__init__()
        self.model = None
        self.kws_stream = None

    def get_handler_info(self):
        """ 返回处理器的基本信息 """
        return HandlerBaseInfo(
            config_model=KwsConfig
        )

    def load(self, engine_config: ChatEngineConfigModel, handler_config=None):
        """ 加载 Sherpa KWS 模型 """
        if not isinstance(handler_config, KwsConfig):
            handler_config = KwsConfig()
        model_path = os.path.join(engine_config.model_root, handler_config.model_name)
        self.model = sherpa_onnx.KeywordSpotter(
            tokens=os.path.join(model_path, handler_config.tokens),
            encoder=os.path.join(model_path, handler_config.encoder),
            decoder=os.path.join(model_path, handler_config.decoder),
            joiner=os.path.join(model_path, handler_config.joiner),
            num_threads=handler_config.num_threads,
            sample_rate=handler_config.sample_rate,
            keywords_score=handler_config.keywords_score,
            keywords_threshold=handler_config.keywords_threshold,
            keywords_file=os.path.join(model_path, handler_config.keywords_file),
            provider="cpu"
        )
        self.kws_stream = self.model.create_stream()
        logger.info(f"Loaded Sherpa KWS model from {model_path}")

    def create_context(self, session_context: SessionContext, handler_config=None) -> HandlerContext:
        """
        创建 KwsContext 实例
            设置共享状态引用
            启用kws语言唤醒
        """
        context = KwsContext(session_context.session_info.session_id)
        context.shared_states = session_context.shared_states
        context.shared_states.enable_keyword = True
        if isinstance(handler_config, KwsConfig):
            context.config = handler_config
            context.sample_rate = handler_config.sample_rate
        return context

    def start_context(self, session_context, handler_context):
        pass

    def get_handler_detail(self, session_context: SessionContext,
                           context: HandlerContext) -> HandlerDetail:
        audio_definition = DataBundleDefinition()
        audio_definition.add_entry(DataBundleEntry.create_audio_entry(
            "kws_human_audio", 1, 16000
        ))
        keyword_definition = DataBundleDefinition()
        keyword_definition.add_entry(DataBundleEntry.create_text_entry("keyword_text"))
        inputs = {
            ChatDataType.HUMAN_AUDIO: HandlerDataInfo(
                type=ChatDataType.HUMAN_AUDIO
            )
        }
        outputs = {
            ChatDataType.HUMAN_AUDIO: HandlerDataInfo(
                type=ChatDataType.HUMAN_AUDIO,
                definition=audio_definition
            ),
            ChatDataType.KEYWORD_TEXT: HandlerDataInfo(
                type=ChatDataType.KEYWORD_TEXT,
                definition=keyword_definition
            )
        }
        return HandlerDetail(
            inputs=inputs,
            outputs=outputs,
        )

    def handle(self, context: HandlerContext, inputs: ChatData,
               output_definitions: Dict[ChatDataType, HandlerDataInfo]):
        """ 输入音频关键字检测 """
        context = cast(KwsContext, context)
        audio_output_definition = output_definitions.get(ChatDataType.HUMAN_AUDIO).definition
        keyword_output_definition = output_definitions.get(ChatDataType.KEYWORD_TEXT).definition
        if inputs.type != ChatDataType.HUMAN_AUDIO:
            return

        """ 收集音频，用于以下的关键词检测 """
        audio = inputs.data.get_main_data()
        if audio is not None:
            audio = audio.squeeze()
            context.output_audios.append(audio)
        speech_end = inputs.data.get_meta("human_speech_end", False)
        if not speech_end:
            return

        output_audio = np.concatenate(context.output_audios)
        context.output_audios.clear()

        kws_audio = output_audio
        if output_audio.dtype != np.float32:
            kws_audio = kws_audio.astype(np.float32)
        kws_audio = kws_audio / 32768.0 if kws_audio.max() > 1.0 else kws_audio

        """ keyword 关键词检测 """
        sample_rate = context.sample_rate
        tail_paddings = np.zeros(int(1 * sample_rate), dtype=np.float32)
        self.kws_stream.accept_waveform(sample_rate, kws_audio)
        self.kws_stream.accept_waveform(sample_rate, tail_paddings)
        self.kws_stream.input_finished()
        keyword: Optional[str] = None
        while self.model.is_ready(self.kws_stream):
            self.model.decode_stream(self.kws_stream)
            result = self.model.get_result(self.kws_stream)
            logger.info(f"KWS result: {result}")
            if result and result.strip() != '':
                keyword = result
        self.model.reset_stream(self.kws_stream)
        if keyword:
            """ 检测到关键词，返回指令，重新启用vad """
            logger.info(f"Keyword detected: {keyword}")
            data_bundle = DataBundle(keyword_output_definition)
            data_bundle.set_main_data(keyword)
            chat_data = ChatData(type=ChatDataType.KEYWORD_TEXT, data=data_bundle)
            context.submit_data(chat_data)
            context.shared_states.enable_vad = True
        else:
            logger.info("No keyword detected")
            speech_id = f"speech-{context.session_id}"
            data_bundle = DataBundle(audio_output_definition)
            if output_audio.dtype != np.float32:
                output_audio = output_audio.astype(np.float32)
            if output_audio.ndim == 1:
                output_audio = output_audio[np.newaxis, ...]
            elif output_audio.ndim == 2 and output_audio.shape[0] != 1:
                output_audio = output_audio[:1, ...]
            data_bundle.set_main_data(output_audio)
            data_bundle.add_meta("speech_id", speech_id)
            data_bundle.add_meta("human_speech_end", True)
            data_bundle.add_meta("keyword_audio", True)
            chat_data = ChatData(type=ChatDataType.HUMAN_AUDIO, data=data_bundle)
            context.submit_data(chat_data)

    def destroy_context(self, context: HandlerContext):
        pass
