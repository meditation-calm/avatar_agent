import enum
import math
from typing import Dict, Optional, Tuple

import numpy as np
from loguru import logger
from pydantic import BaseModel, Field

from src.chat_engine.contexts.handler_context import HandlerContext
from src.chat_engine.data_models.chat_engine_config_data import HandlerBaseConfigModel
from src.engine_utils.general_slicer import SliceContext


class VADConfig(HandlerBaseConfigModel, BaseModel):
    """
    定义 Silero VAD 模型的配置参数，存储 VAD 模型的配置参数，包括语音检测的阈值、延迟等参数。
        speaking_threshold: 判断为语音的概率阈值，默认为 0.5。
        start_delay: 语音开始前的延迟样本数，默认为 2048。
        end_delay: 语音结束后的延迟样本数，默认为 5000。
        buffer_look_back: 回溯缓冲区的样本数，默认为 1024。
        speech_padding: 语音前后填充的样本数，默认为 512。
    """
    speaking_threshold: float = Field(default=0.5)
    start_delay: int = Field(default=2048)
    end_delay: int = Field(default=5000)
    buffer_look_back: int = Field(default=1024)
    speech_padding: int = Field(default=512)


class SpeakingStatus(enum.Enum):
    """
    定义语音检测的状态，包括 PRE_START（预开始）、START（开始）和 END（结束）
    """
    PRE_START = enum.auto()
    START = enum.auto()
    END = enum.auto()


class VADContext(HandlerContext):
    """
    维护语音检测过程中的状态信息，如音频历史、语音长度、静音长度等。
        config: 配置模型实例。
        speaking_status: 当前语音状态。
        audio_history: 存储音频片段的历史记录。
        speech_length: 当前语音段的长度。
        silence_length: 当前静音段的长度。
        model_state: 模型状态。
        slice_context: 数据切片上下文。
        speech_id: 语音片段的唯一标识。
    """

    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.config: Optional[VADConfig] = None
        self.speaking_status = SpeakingStatus.END

        self.clip_size = 512

        self.audio_history = []
        self.history_length_limit = 0

        self.speech_length: int = 0
        self.silence_length: int = 0

        self.shared_states = None

        self.model_state: Optional[np.ndarray] = None
        self.slice_context: Optional[SliceContext] = None

        self.speech_id: int = 0

        self.is_started = False

    def reset(self):
        """ 重置上下文状态，清空音频历史和相关计数器，为下一次语音检测做准备。 """
        self.audio_history.clear()
        self.speech_length = 0
        self.silence_length = 0
        self.slice_context.flush()

    def _update_status_on_pre_start(self, clip: np.ndarray, _timestamp: Optional[int] = None):
        """ 处理预开始状态下的状态更新，当语音长度超过 start_delay 时，切换到 START 状态。 """
        if self.speech_length >= self.config.start_delay:
            """
            当语音长度达到 start_delay 阈值时：
            将状态切换为 START
            从历史记录中获取回溯的音频数据
            添加前导静音填充 speech_padding
            返回完整的语音片段和相关元数据
            """
            head_sample_id = None
            self.speaking_status = SpeakingStatus.START
            sample_num_to_fetch = self.config.buffer_look_back + self.config.start_delay
            slice_num_to_fetch = math.ceil(sample_num_to_fetch / self.clip_size)
            audio_clips = []
            for history_entry in self.audio_history[-slice_num_to_fetch:]:
                history_clip, history_timestamp = history_entry
                if head_sample_id is None:
                    head_sample_id = history_timestamp
                audio_clips.append(history_clip)
            output_audio = np.concatenate(audio_clips, axis=0)
            output_audio = np.concatenate([np.zeros(self.config.speech_padding, dtype=clip.dtype), output_audio],
                                          axis=0)
            self.speech_id += 1
            logger.info("Start of human speech")
            extra_args = {
                "human_speech_start": True,
                "pre_padding": self.config.speech_padding,
            }
            if head_sample_id is not None:
                extra_args["head_sample_id"] = head_sample_id
                logger.info(f"VAD pre_start to start got timestamp {head_sample_id}")
            return output_audio, extra_args
        else:
            """ 检测到静音，则回退到 END 状态 """
            if self.silence_length > 0:
                logger.info("Back to not started status")
                self.speaking_status = SpeakingStatus.END
            return None, {}

    def _update_status_on_start(self, clip: np.ndarray, timestamp: Optional[int] = None):
        """ 处理开始状态下的状态更新，当静音长度超过 end_delay 时，切换到 END 状态。 """
        if self.silence_length >= self.config.end_delay:
            """
            当静音长度达到 end_delay 阈值时：
            将状态切换为 END
            添加后导静音填充 speech_padding
            返回最终的语音片段和结束标记
            """
            self.speaking_status = SpeakingStatus.END
            output_audio = np.concatenate([clip, np.zeros(self.config.speech_padding, dtype=clip.dtype)], axis=0)
            logger.info("End of human speech")
            extra_args = {
                "human_speech_end": True,
                "post_padding": self.config.speech_padding,
            }
            if timestamp is not None:
                extra_args["head_sample_id"] = timestamp
                logger.info(f"VAD start to start got timestamp {timestamp}")
            return output_audio, extra_args
        else:
            """ 继续传输当前音频片段 """
            return clip, {"head_sample_id": timestamp}

    def _update_status_on_end(self, _clip: np.ndarray, _timestamp: Optional[int] = None):
        """ 处理结束状态下的状态更新，当检测到语音时，切换到 PRE_START 状态。 """
        if self.speech_length > 0:
            logger.info("Pre start of new human speech")
            self.speaking_status = SpeakingStatus.PRE_START
        return None, {}

    def _append_to_history(self, clip: np.ndarray, timestamp: Optional[int] = None):
        """ 将音频片段添加到历史记录中，并维护历史记录的长度限制。 """
        self.audio_history.append((clip, timestamp))
        while 0 < self.history_length_limit < len(self.audio_history):
            """ 根据 history_length_limit 限制历史记录长度，移除过期数据 """
            self.audio_history.pop(0)

    def update_status(self, speech_prob: float, clip: np.ndarray,
                      timestamp: Optional[int] = None) -> Tuple[Optional[np.ndarray], Dict]:
        """ 根据语音概率更新语音状态，并返回相应的音频片段和附加参数。 """
        self._append_to_history(clip, timestamp)
        if speech_prob > self.config.speaking_threshold:
            self.speech_length += self.clip_size
            self.silence_length = 0
        else:
            self.silence_length += self.clip_size
            self.speech_length = 0
        if self.speaking_status == SpeakingStatus.PRE_START:
            return self._update_status_on_pre_start(clip, timestamp)
        elif self.speaking_status == SpeakingStatus.START:
            return self._update_status_on_start(clip, timestamp)
        elif self.speaking_status == SpeakingStatus.END:
            return self._update_status_on_end(clip, timestamp)