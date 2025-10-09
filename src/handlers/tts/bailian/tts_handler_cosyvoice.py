import os
import re
from typing import Dict, Optional, cast
from abc import ABC

import dashscope
import numpy as np
from dashscope.audio.tts_v2 import SpeechSynthesizer, ResultCallback, AudioFormat
from loguru import logger
from pydantic import BaseModel

from src.chat_engine.common.handler_base import HandlerBase, HandlerBaseInfo, HandlerDetail, HandlerDataInfo
from src.chat_engine.contexts.handler_context import HandlerContext
from src.chat_engine.contexts.session_context import SessionContext
from src.chat_engine.data_models.chat_data.chat_data_model import ChatData
from src.chat_engine.data_models.chat_data_type import ChatDataType
from src.chat_engine.data_models.chat_engine_config_data import ChatEngineConfigModel
from src.chat_engine.data_models.runtime_data.data_bundle import DataBundle, DataBundleDefinition, DataBundleEntry
from src.handlers.tts.bailian.tts_handler_base import TTSContext, TTSConfig


class CosyvoiceCallBack(ResultCallback):
    def __init__(self, context: TTSContext, output_definition, speech_id):
        super().__init__()
        self.context = context
        self.output_definition = output_definition
        self.speech_id = speech_id
        self.temp_bytes = b''

    def on_open(self) -> None:
        logger.info('bailian cosyvoice connect')

    def on_event(self, message) -> None:
        pass

    def sendAudioData(self, output_audio: np.ndarray = None, avatar_speech_end: bool = False):
        if output_audio is not None:
            output = DataBundle(self.output_definition)
            output.set_main_data(output_audio)
            output.add_meta("avatar_speech_end", avatar_speech_end)
            output.add_meta("speech_id", self.speech_id)
            self.context.submit_data(output)

    def on_data(self, data: bytes) -> None:
        self.temp_bytes += data
        if len(self.temp_bytes) > 24000:
            # 实现接收合成二进制音频结果的逻辑
            output_audio = np.array(np.frombuffer(self.temp_bytes, dtype=np.int16)).astype(np.float32)/32767
            # librosa.load(io.BytesIO(self.temp_bytes), sr=None)[0]
            output_audio = output_audio[np.newaxis, ...]
            self.sendAudioData(output_audio, False)
            self.temp_bytes = b''

    def on_complete(self) -> None:
        if len(self.temp_bytes) > 0:
            output_audio = np.array(np.frombuffer(self.temp_bytes, dtype=np.int16)).astype(np.float32)/32767
            output_audio = output_audio[np.newaxis, ...]
            self.sendAudioData(output_audio, False)
            self.temp_bytes = b''
        self.sendAudioData(np.zeros(shape=(1, 240), dtype=np.float32), True)
        logger.info(f"bailian cosyvoice speech end")

    def on_error(self, message) -> None:
        logger.error(f'bailian cosyvoice 服务出现异常,请确保参数正确：${message}')
        self.sendAudioData(np.zeros(shape=(1, 240), dtype=np.float32), True)
        logger.info(f"bailian cosyvoice speech error end")

    def on_close(self) -> None:
        logger.info('bailian cosyvoice close')


class HandlerTTS(HandlerBase, ABC):
    def __init__(self):
        super().__init__()
        self.sample_rate = None

    def get_handler_info(self) -> HandlerBaseInfo:
        return HandlerBaseInfo(
            config_model=TTSConfig,
        )

    def get_handler_detail(self, session_context: SessionContext,
                           context: HandlerContext) -> HandlerDetail:
        definition = DataBundleDefinition()
        definition.add_entry(DataBundleEntry.create_audio_entry("avatar_audio", 1, self.sample_rate))
        inputs = {
            ChatDataType.AVATAR_TEXT: HandlerDataInfo(
                type=ChatDataType.AVATAR_TEXT,
            )
        }
        outputs = {
            ChatDataType.AVATAR_AUDIO: HandlerDataInfo(
                type=ChatDataType.AVATAR_AUDIO,
                definition=definition,
            )
        }
        return HandlerDetail(
            inputs=inputs, outputs=outputs,
        )

    def load(self, engine_config: ChatEngineConfigModel, handler_config: Optional[BaseModel] = None):
        config = cast(TTSConfig, handler_config)
        self.sample_rate = config.sample_rate
        if 'DASHSCOPE_API_KEY' in os.environ:
            # load API-key from environment variable DASHSCOPE_API_KEY
            dashscope.api_key = os.environ['DASHSCOPE_API_KEY']
        else:
            dashscope.api_key = config.api_key

    def create_context(self, session_context, handler_config=None):
        if not isinstance(handler_config, TTSConfig):
            handler_config = TTSConfig()
        context = TTSContext(session_context.session_info.session_id)
        context.config = handler_config
        return context

    def start_context(self, session_context, context: HandlerContext):
        pass

    def handle(self, context: HandlerContext, inputs: ChatData,
               output_definitions: Dict[ChatDataType, HandlerDataInfo]):
        output_definition = output_definitions.get(ChatDataType.AVATAR_AUDIO).definition
        context = cast(TTSContext, context)
        if inputs.type != ChatDataType.AVATAR_TEXT:
            return

        text = inputs.data.get_main_data()
        speech_id = inputs.data.get_meta("speech_id", context.session_id)

        if text is not None:
            text = re.sub(r"<\|.*?\|>", "", text)

        text_end = inputs.data.get_meta("avatar_text_end", False)
        try:
            if not text_end:
                if context.synthesizer is None:
                    callback = CosyvoiceCallBack(
                        context=context, output_definition=output_definition, speech_id=speech_id)
                    context.synthesizer = SpeechSynthesizer(
                        model=context.config.model_name,
                        voice=context.config.voice,
                        format=AudioFormat.PCM_24000HZ_MONO_16BIT,
                        volume=context.config.volume,
                        speech_rate= context.config.volume,
                        pitch_rate=context.config.pitch_rate,
                        callback=callback)
                logger.info(f'bailian cosyvoice streaming_call {text}')
                context.synthesizer.streaming_call(text)
            else:
                logger.info(f'bailian cosyvoice streaming_call last {text}')
                context.synthesizer.streaming_call(text)
                context.synthesizer.streaming_complete()
                context.synthesizer = None
                context.input_text = ''
        except Exception as e:
            logger.error(e)
            context.synthesizer.streaming_complete()
            context.synthesizer = None

    def destroy_context(self, context: HandlerContext):
        pass
