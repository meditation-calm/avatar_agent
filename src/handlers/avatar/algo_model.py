from enum import Enum
from typing import Any

import av
from pydantic import BaseModel


class AvatarStatus(Enum):
    """
    SPEAKING（说话状态） 表示数字人当前正在说话
    LISTENING（监听状态） 表示数字人当前处于监听状态，没有在说话
    """
    SPEAKING = 0
    LISTENING = 1


class Tts2FaceEvent(Enum):
    """
    定义TTS到面部动画系统中的事件类型。具体包含以下事件：
    START: 值为1001，表示开始事件
    STOP: 值为1002，表示停止事件
    LISTENING_TO_SPEAKING: 值为2001，表示从聆听状态切换到说话状态的事件
    SPEAKING_TO_LISTENING: 值为2002，表示从说话状态切换到聆听状态的事件
    """
    START = 1001
    STOP = 1002
    LISTENING_TO_SPEAKING = 2001
    SPEAKING_TO_LISTENING = 2002


class AudioFrame(BaseModel):
    def __init__(self, speech_id: Any, speech_end: bool, frame: bytes | av.AudioFrame):
        super().__init__()
        self.speech_id = speech_id
        self.speech_end = speech_end
        self.frame = frame

    model_config = {
        "arbitrary_types_allowed": True
    }


class VideoFrame(BaseModel):
    def __init__(self, speech_id: Any, speech_end: bool, avatar_status: AvatarStatus, frame: Any | av.VideoFrame, bg_frame_id: int):
        super().__init__()
        self.speech_id = speech_id
        self.speech_end = speech_end
        self.avatar_status = avatar_status
        self.frame = frame
        self.bg_frame_id = bg_frame_id

    model_config = {
        "arbitrary_types_allowed": True
    }
