import math
import os
from abc import ABC
from typing import cast, Dict

import numpy as np
import onnxruntime
from loguru import logger

from src.chat_engine.common.handler_base import HandlerBase, HandlerDetail, HandlerDataInfo, HandlerBaseInfo
from src.chat_engine.data_models.chat_data_type import ChatDataType
from src.chat_engine.contexts.handler_context import HandlerContext
from src.chat_engine.contexts.session_context import SessionContext
from src.chat_engine.data_models.chat_data.chat_data_model import ChatData
from src.chat_engine.data_models.chat_engine_config_data import ChatEngineConfigModel
from src.chat_engine.data_models.runtime_data.data_bundle import DataBundle, DataBundleDefinition, DataBundleEntry
from src.engine_utils.general_slicer import SliceContext, slice_data
from src.handlers.vad.silero.vad_handler_base import VADConfig, VADContext


class HandlerAudioVAD(HandlerBase, ABC):
    """
    用于处理音频语音活动检测（VAD）。
    加载 Silero VAD 模型，处理音频数据，检测语音活动，并生成相应的输出数据。
    """
    def __init__(self):
        super().__init__()
        self.model = None
        self.providers = ["CPUExecutionProvider"]

    def get_handler_info(self):
        """ 返回处理器的基本信息 """
        return HandlerBaseInfo(
            config_model=VADConfig
        )

    def load(self, engine_config: ChatEngineConfigModel, handler_config=None):
        """ 加载 Silero VAD 模型，使用 ONNX Runtime 进行推理。 """
        model_name = "silero_vad.onnx"
        model_path = os.path.join(engine_config.model_root, "silero_vad", model_name)
        """  
        设置线程数为 1（inter_op_num_threads 和 intra_op_num_threads）
        设置日志级别为 3（仅显示警告和错误）
        创建 ONNX Runtime 推理会话，使用 CPU 执行提供程序
        """
        options = onnxruntime.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        options.log_severity_level = 3
        self.model = onnxruntime.InferenceSession(model_path,
                                                  providers=self.providers,
                                                  sess_options=options)
        logger.info(f"Loaded Silero VAD model from {model_path}")

    def create_context(self, session_context: SessionContext, handler_config=None) -> HandlerContext:
        """
        创建 HumanAudioVADContext 实例
            设置共享状态引用
            如果提供了有效的配置，则使用该配置替代默认配置
            初始化模型状态为零数组（形状为 (2, 1, 128)）
            创建数据切片上下文，用于处理音频数据切片
            计算并设置历史记录长度限制
        """
        context = VADContext(session_context.session_info.session_id)
        context.shared_states = session_context.shared_states
        if isinstance(handler_config, VADConfig):
            context.config = handler_config
        context.model_state = np.zeros((2, 1, 128), dtype=np.float32)
        context.slice_context = SliceContext.create_numpy_slice_context(
            slice_size=context.clip_size,
            slice_axis=0,
        )
        context.history_length_limit = math.ceil((context.config.start_delay + context.config.buffer_look_back)
                                                 / context.clip_size)
        return context

    def start_context(self, session_context, handler_context):
        pass

    def get_handler_detail(self, session_context: SessionContext,
                           context: HandlerContext) -> HandlerDetail:
        """ 返回处理器的输入输出定义，指定输入为麦克风音频，输出为人类语音。 """
        definition = DataBundleDefinition()
        definition.add_entry(DataBundleEntry.create_audio_entry("human_audio", 1, 16000))

        inputs = {
            ChatDataType.MIC_AUDIO: HandlerDataInfo(
                type=ChatDataType.MIC_AUDIO
            )
        }
        outputs = {ChatDataType.HUMAN_AUDIO: HandlerDataInfo(
                type=ChatDataType.HUMAN_AUDIO,
                definition=definition
            )
        }
        return HandlerDetail(
            inputs=inputs,
            outputs=outputs,
        )

    def _inference(self, context: VADContext, clip: np.ndarray, sr: int = 16000):
        """
        使用 Silero VAD 模型对音频片段进行推理
            预处理音频数据，确保为一维数组
            构造模型输入字典，包括音频数据、采样率和模型状态
            执行模型推理，获取语音概率和更新后的模型状态
            更新上下文中的模型状态
            返回语音概率值
        """
        clip = clip.squeeze()
        if clip.ndim != 1:
            logger.warning("Input audio should be 1-dim array")
            return 0
        clip = np.expand_dims(clip, axis=0)
        inputs = {
            "input": clip,
            "sr": np.array([sr], dtype=np.int64),
            "state": context.model_state
        }
        prob, state = self.model.run(None, inputs)
        context.model_state = state
        return prob[0][0]

    def handle(self, context: HandlerContext, inputs: ChatData,
               output_definitions: Dict[ChatDataType, HandlerDataInfo]):
        """ 处理输入音频数据，执行语音活动检测并生成输出 """
        context = cast(VADContext, context)
        output_definition = output_definitions.get(ChatDataType.HUMAN_AUDIO).definition
        """ 检查是否启用 VAD，未启用则直接返回 """
        if not context.shared_states.enable_vad:
            return
        if inputs.type != ChatDataType.MIC_AUDIO:
            return

        """ 提取音频数据并进行预处理 """
        audio = inputs.data.get_main_data()
        if audio is None:
            return
        audio_entry = inputs.data.get_main_definition_entry()
        sample_rate = audio_entry.sample_rate
        audio = audio.squeeze()

        timestamp = None
        if inputs.is_timestamp_valid():
            timestamp = inputs.timestamp

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32) / 32767

        context.slice_context.update_start_id(timestamp[0], force_update=False)

        """ 对音频数据进行切片处理 """
        for clip in slice_data(context.slice_context, audio):
            head_sample_id = context.slice_context.get_last_slice_start_index()
            """ 对每个切片执行语音检测推理 """
            speech_prob = self._inference(context, clip)
            """ 调用上下文的 update_status 方法更新状态并获取处理结果 """
            audio_clip, extra_args = context.update_status(speech_prob, clip, timestamp=head_sample_id)
            # FIXME this is a hack to disable VAD after human speech end,
            #  but it should be handled by client or downstream handlers
            human_speech_end = extra_args.get("human_speech_end", False)
            timestamp = extra_args.get("head_sample_id", head_sample_id)
            speech_id = f"speech-{context.session_id}-{context.speech_id}"
            """ 
                处理语音结束情况，禁用 VAD 并重置上下文
                如果有音频片段输出，则构造 ChatData 对象并生成输出 
            """
            if human_speech_end:
                context.shared_states.enable_vad = False
                context.reset()
            if audio_clip is not None:
                output = DataBundle(output_definition)
                output.set_main_data(np.expand_dims(audio_clip, axis=0))
                for flag_name, flag_value in extra_args.items():
                    output.add_meta(flag_name, flag_value)
                output.add_meta("speech_id", speech_id)
                output_chat_data = ChatData(
                    type=ChatDataType.HUMAN_AUDIO,
                    data=output
                )
                if timestamp >= 0:
                    output_chat_data.timestamp = timestamp, sample_rate
                yield output_chat_data

    def destroy_context(self, context: HandlerContext):
        pass
