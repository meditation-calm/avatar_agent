import re
from abc import ABC
from typing import Optional, Dict, cast

import numpy as np
import torch
from funasr import AutoModel
from loguru import logger
from pydantic import BaseModel

from src.chat_engine.common.handler_base import HandlerBase, HandlerBaseInfo, HandlerDetail, HandlerDataInfo
from src.chat_engine.contexts.handler_context import HandlerContext
from src.chat_engine.contexts.session_context import SessionContext
from src.chat_engine.data_models.chat_data.chat_data_model import ChatData
from src.chat_engine.data_models.chat_data_type import ChatDataType
from src.chat_engine.data_models.chat_engine_config_data import ChatEngineConfigModel
from src.chat_engine.data_models.runtime_data.data_bundle import DataBundleDefinition, DataBundleEntry, DataBundle
from src.engine_utils.general_slicer import slice_data
from src.handlers.asr.sensevoice.asr_handler_base import ASRConfig, ASRContext


class ASRHandler(HandlerBase, ABC):

    def __init__(self):
        super().__init__()
        self.model = None
        self.model_name = 'iic/SenseVoiceSmall'
        # 检测可用设备（CUDA、MPS或CPU）并设置设备
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")  # NVIDIA GPU（支持CUDA）
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")    # Apple M系列芯片（支持MPS）
        else:
            self.device = torch.device("cpu")    # 兜底使用CPU
        logger.info(f"ASR SenseVoice Using device: {self.device}")

    def get_handler_info(self) -> HandlerBaseInfo:
        return HandlerBaseInfo(
            name="ASR_SenseVoice",
            config_model=ASRConfig,
        )

    def get_handler_detail(self, session_context: SessionContext,
                           context: HandlerContext) -> HandlerDetail:
        definition = DataBundleDefinition()
        definition.add_entry(DataBundleEntry.create_audio_entry("avatar_audio", 1, 24000))
        inputs = {
            ChatDataType.HUMAN_AUDIO: HandlerDataInfo(
                type=ChatDataType.HUMAN_AUDIO,
            )
        }
        outputs = {
            ChatDataType.HUMAN_TEXT: HandlerDataInfo(
                type=ChatDataType.HUMAN_TEXT,
                definition=definition,
            )
        }
        return HandlerDetail(
            inputs=inputs, outputs=outputs,
        )

    def load(self, engine_config: ChatEngineConfigModel, handler_config: Optional[BaseModel] = None):
        if isinstance(handler_config, ASRConfig):
            self.model_name = handler_config.model_name
        self.model = AutoModel(model=self.model_name, disable_update=True)
        logger.info(f"Loaded SenseVoice model from {self.model_name}")

    def create_context(self, session_context, handler_config=None):
        if not isinstance(handler_config, ASRConfig):
            handler_config = ASRConfig()
        context = ASRContext(session_context.session_info.session_id)
        context.config = handler_config
        # 设置共享状态引用
        context.shared_states = session_context.shared_states
        return context

    def start_context(self, session_context, handler_context):
        pass

    def handle(self, context: HandlerContext, inputs: ChatData,
               output_definitions: Dict[ChatDataType, HandlerDataInfo]):
        """ 处理输入音频数据并生成文本输出 """
        output_definition = output_definitions.get(ChatDataType.HUMAN_TEXT).definition
        context = cast(ASRContext, context)
        """ 验证输入数据类型是否为 HUMAN_AUDIO """
        if inputs.type == ChatDataType.HUMAN_AUDIO:
            audio = inputs.data.get_main_data()
        else:
            return
        """ 获取语音ID，如果不存在则使用会话ID """
        speech_id = inputs.data.get_meta("speech_id")
        if speech_id is None:
            speech_id = context.session_id

        if audio is not None:
            audio = audio.squeeze()

            logger.info('audio in')
            """ 使用 slice_data 处理音频切片并存储到 output_audios """
            for audio_segment in slice_data(context.audio_slice_context, audio):
                if audio_segment is None or audio_segment.shape[0] == 0:
                    continue
                context.output_audios.append(audio_segment)

        """ 检查是否为语音结束标记，如果不是则返回继续收集音频 """
        speech_end = inputs.data.get_meta("human_speech_end", False)
        if not speech_end:
            return

        # prefill remainder audio in slice context
        """
            处理剩余音频数据，确保大小符合要求
            合并所有音频片段
        """
        remainder_audio = context.audio_slice_context.flush()
        if remainder_audio is not None:
            if remainder_audio.shape[0] < context.audio_slice_context.slice_size:
                remainder_audio = np.concatenate(
                    [remainder_audio,
                     np.zeros(shape=(context.audio_slice_context.slice_size - remainder_audio.shape[0]))])
                context.output_audios.append(remainder_audio)
        output_audio = np.concatenate(context.output_audios)
        """ 如果启用了音频转储，则将音频写入文件 """
        if context.audio_dump_file is not None:
            logger.info('dump audio')
            context.audio_dump_file.write(output_audio.tobytes())

        """ ASR模型生成文本结果 """
        res = self.model.generate(input=output_audio,
                                  language=context.config.language,
                                  use_itn=context.config.use_itn,
                                  batch_size_s=context.config.batch_size_s)
        logger.info(f"audio asr res {res}", )
        """ 清理音频缓存 """
        context.output_audios.clear()
        """ 处理识别结果，移除特殊标记 """
        output_text = re.sub(r"<\|.*?\|>", "", res[0]['text'])
        if len(output_text) == 0:
            """ 如果识别结果为空，则重新启用VAD """
            context.shared_states.enable_vad = True
            return
        """ 生成文本输出和结束标记输出 """
        """ 返回识别结果 """
        output = DataBundle(output_definition)
        output.set_main_data(output_text)
        output.add_meta('human_text_end', False)
        output.add_meta('speech_id', speech_id)
        yield output
        """ 标记结束 """
        end_output = DataBundle(output_definition)
        end_output.set_main_data('')
        end_output.add_meta("human_text_end", True)
        end_output.add_meta("speech_id", speech_id)
        yield end_output

    def destroy_context(self, context: HandlerContext):
        pass
