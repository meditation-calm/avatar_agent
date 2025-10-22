import os.path
from abc import ABC
from typing import cast, Dict

import numpy as np
import torch
from funasr import AutoModel
from loguru import logger

from src.chat_engine.common.handler_base import HandlerBase, HandlerBaseInfo, HandlerDataInfo, HandlerDetail
from src.chat_engine.contexts.handler_context import HandlerContext
from src.chat_engine.contexts.session_context import SessionContext
from src.chat_engine.data_models.chat_data.chat_data_model import ChatData
from src.chat_engine.data_models.chat_data_type import ChatDataType
from src.chat_engine.data_models.chat_engine_config_data import ChatEngineConfigModel
from src.chat_engine.data_models.runtime_data.data_bundle import DataBundleDefinition, DataBundleEntry, DataBundle
from src.engine_utils.directory_info import DirectoryInfo
from src.engine_utils.general_slicer import slice_data
from src.handlers.kws.xiaoyun.kws_handler_base import KwsConfig, KwsContext


class KwsHandler(HandlerBase, ABC):
    """
    用于处理音频语音唤醒检测（KWS）。
    检测音频数据关键词，唤醒数字人
    """

    def __init__(self):
        super().__init__()
        self.model = None
        self.model_name = 'iic/speech_charctc_kws_phone-xiaoyun'
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        logger.info(f"KWS xiaoyun Using device: {self.device}")

    def get_handler_info(self):
        """ 返回处理器的基本信息 """
        return HandlerBaseInfo(
            config_model=KwsConfig
        )

    def load(self, engine_config: ChatEngineConfigModel, handler_config=None):
        if not isinstance(handler_config, KwsConfig):
            handler_config = KwsConfig()
        self.model_name = handler_config.model_name
        self.model = AutoModel(model=self.model_name,
                               keywords=handler_config.keywords,
                               output_dir=os.path.join(DirectoryInfo.get_project_dir(), "cache"),
                               disable_update=True)
        logger.info(f"Loaded xiaoyun kWS model from {self.model_name}")

    def create_context(self, session_context: SessionContext, handler_config=None) -> HandlerContext:
        """
        创建 KwsContext 实例
            设置共享状态引用
            启用kws语言唤醒
        """
        context = KwsContext(session_context.session_info.session_id)
        context.shared_states = session_context.shared_states
        context.shared_states.enable_kws = True
        if isinstance(handler_config, KwsConfig):
            context.config = handler_config
        return context

    def start_context(self, session_context, handler_context):
        pass

    def get_handler_detail(self, session_context: SessionContext,
                           context: HandlerContext) -> HandlerDetail:
        definition = DataBundleDefinition()
        definition.add_entry(DataBundleEntry.create_audio_entry(
            "kws_human_audio", 1, 16000
        ))

        inputs = {
            ChatDataType.HUMAN_AUDIO: HandlerDataInfo(
                type=ChatDataType.HUMAN_AUDIO
            )
        }
        outputs = {
            ChatDataType.HUMAN_AUDIO: HandlerDataInfo(
                type=ChatDataType.HUMAN_AUDIO,
                definition=definition
            )
        }
        return HandlerDetail(
            inputs=inputs,
            outputs=outputs,
        )

    def handle(self, context: HandlerContext, inputs: ChatData,
               output_definitions: Dict[ChatDataType, HandlerDataInfo]):
        """ 处理输入音频数据，执行语音活动检测并生成输出 """
        context = cast(KwsContext, context)
        output_definition = output_definitions.get(ChatDataType.HUMAN_AUDIO).definition
        """ 检查是否启用 KWS，未启用则直接返回 """
        if not context.shared_states.enable_kws:
            return
        if inputs.type != ChatDataType.HUMAN_AUDIO:
            return
        audio = inputs.data.get_main_data()
        if audio is not None:
            audio = audio.squeeze()
            for audio_segment in slice_data(context.audio_slice_context, audio):
                if audio_segment is None or audio_segment.shape[0] == 0:
                    continue
                context.output_audios.append(audio_segment)
        speech_end = inputs.data.get_meta("human_speech_end", False)
        if not speech_end:
            return

        remainder_audio = context.audio_slice_context.flush()
        if remainder_audio is not None:
            if remainder_audio.shape[0] < context.audio_slice_context.slice_size:
                remainder_audio = np.concatenate(
                    [remainder_audio,
                     np.zeros(shape=(context.audio_slice_context.slice_size - remainder_audio.shape[0]))])
                context.output_audios.append(remainder_audio)
        output_audio = np.concatenate(context.output_audios)
        context.output_audios.clear()
        """ 检测语音唤醒 """
        res = self.model.generate(input=output_audio, cache={})
        logger.info(f"xiaoyun kws res {res}")
        try:
            if res and isinstance(res, list) and len(res) > 0:
                result = res[0]
                if 'text' in result:
                    text_parts = result['text'].split()
                    if len(text_parts) > 1:
                        confidence = float(text_parts[-1])
                        """ 语言唤醒达到概率阈值 """
                        # if confidence > context.config.speaking_threshold:
                        if confidence > 0:
                            logger.info(f"xiaoyun kws wake up")
                            context.shared_states.enable_kws = False
                            speech_id = f"speech-{context.session_id}"
                            output = DataBundle(output_definition)
                            if output_audio.dtype != np.float32:
                                output_audio = output_audio.astype(np.float32)
                            if output_audio.ndim == 1:
                                output_audio = output_audio[np.newaxis, ...]
                            elif output_audio.ndim == 2 and output_audio.shape[0] != 1:
                                output_audio = output_audio[:1, ...]
                            output.set_main_data(output_audio)
                            output.add_meta("speech_id", speech_id)
                            output.add_meta("human_speech_end", True)
                            return output
        except Exception as e:
            logger.info(f"xiaoyun kws rejected {e}")
        """ 重新启用vad和kws """
        context.shared_states.enable_vad = True
        context.shared_states.enable_kws = True

    def destroy_context(self, context: HandlerContext):
        pass
