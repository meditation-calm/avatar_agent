import os
import re
from typing import Dict, Optional, cast
from abc import ABC

import dashscope
import numpy as np
from dashscope.audio.tts_v2 import SpeechSynthesizer, ResultCallback, AudioFormat
from loguru import logger
from pydantic import BaseModel
from pydub import AudioSegment

from src.chat_engine.common.handler_base import HandlerBase, HandlerBaseInfo, HandlerDetail, HandlerDataInfo
from src.chat_engine.contexts.handler_context import HandlerContext
from src.chat_engine.contexts.session_context import SessionContext
from src.chat_engine.data_models.chat_data.chat_data_model import ChatData
from src.chat_engine.data_models.chat_data_type import ChatDataType
from src.chat_engine.data_models.chat_engine_config_data import ChatEngineConfigModel
from src.chat_engine.data_models.runtime_data.data_bundle import DataBundle, DataBundleDefinition, DataBundleEntry
from src.engine_utils.directory_info import DirectoryInfo
from src.handlers.tts.bailian.tts_handler_base import TTSContext, TTSConfig


class CosyvoiceCallBack(ResultCallback):
    def __init__(self, context: TTSContext, output_definition, speech_id, audio_save_path):
        super().__init__()
        self.context = context
        self.output_definition = output_definition
        self.speech_id = speech_id
        self.temp_bytes = b''
        self.audio_buffer = b''
        self.audio_save_path = audio_save_path

    def on_open(self) -> None:
        logger.info('bailian cosyvoice connect')

    def on_event(self, message) -> None:
        pass

    def save_as_mp3(self):
        if self.audio_save_path is None:
            return
        os.makedirs(os.path.dirname(self.audio_save_path), exist_ok=True)
        audio_segment = AudioSegment(
            data=self.audio_buffer,
            sample_width=2,  # 16-bit
            frame_rate=24000,  # 24kHz
            channels=1  # mono
        )
        audio_segment.export(self.audio_save_path, format="mp3", bitrate="64k")

    def sendAudioData(self, output_audio: np.ndarray = None, avatar_speech_end: bool = False):
        if output_audio is not None:
            output = DataBundle(self.output_definition)
            output.set_main_data(output_audio)
            output.add_meta("avatar_speech_end", avatar_speech_end)
            output.add_meta("speech_id", self.speech_id)
            self.context.submit_data(output)

    def on_data(self, data: bytes) -> None:
        self.temp_bytes += data
        self.audio_buffer += data
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
        # 保存音频文件
        # self.save_as_mp3()

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
        definition.add_entry(DataBundleEntry.create_audio_entry(
            "avatar_audio", 1, self.sample_rate
        ))
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
        if not text_end:
            first_ignore_text = False
            left_idx = text.find("{")
            right_idx = text.find("}")
            if left_idx != -1 and right_idx != -1:
                left_text = text[:left_idx]
                right_text = text[right_idx + 1:]
                text = left_text + right_text
                context.ignore_text = False
            else:
                if left_idx != -1:
                    text = text[:left_idx]
                    first_ignore_text = True
                if right_idx != -1:
                    text = text[right_idx + 1:]
                    context.ignore_text = False
            if context.ignore_text and first_ignore_text is False:
                return
            if first_ignore_text:
                context.ignore_text = True

        try:
            if not text_end:
                context.input_text = context.input_text + text
                if context.synthesizer is None:
                    callback = CosyvoiceCallBack(
                        context=context,
                        output_definition=output_definition,
                        speech_id=speech_id,
                        audio_save_path=os.path.join(DirectoryInfo.get_cache_dir(), "output_audio",
                                                     speech_id + str(context.synthesizer_idx) + ".mp3")
                    )
                    context.synthesizer = SpeechSynthesizer(
                        model=context.config.model_name,
                        voice=context.config.voice,
                        format=AudioFormat.PCM_24000HZ_MONO_16BIT,
                        volume=context.config.volume,
                        speech_rate= context.config.speech_rate,
                        pitch_rate=context.config.pitch_rate,
                        callback=callback)
                logger.info(f'bailian cosyvoice streaming_call {text}')
                context.synthesizer.streaming_call(text)
            else:
                logger.info(f'bailian cosyvoice streaming_call last {text}')
                context.synthesizer.streaming_call(text)
                context.synthesizer.streaming_complete()
                context.synthesizer = None
                context.synthesizer_idx = context.synthesizer_idx + 1
                context.input_text = ''
                context.ignore_text = False
        except Exception as e:
            logger.error(e)
            context.synthesizer.streaming_complete()
            context.synthesizer = None
            context.ignore_text = False

    def destroy_context(self, context: HandlerContext):
        pass
