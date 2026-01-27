from enum import Enum

from src.chat_engine.common.engine_channel_type import EngineChannelType


class ChatDataType(Enum):

    def __init__(self, value: str, channel_type: EngineChannelType):
        self._value_ = value
        self.channel_type = channel_type

    """ 无数据或空数据类型 """
    NONE = ("none", EngineChannelType.NONE)
    """ 关键字文本数据 """
    KEYWORD_TEXT = ("keyword_text", EngineChannelType.TEXT)
    """ 人类用户输入的文本数据 """
    HUMAN_TEXT = ("human_text", EngineChannelType.TEXT)
    """ 流程处理事件(开始/结束) """
    HUMAN_EVENT = ("human_event", EngineChannelType.TEXT)
    """ 麦克风的原始音频数据 """
    MIC_AUDIO = ("mic_audio", EngineChannelType.AUDIO)
    """ 处理后的人类用户音频数据 """
    HUMAN_AUDIO = ("human_audio", EngineChannelType.AUDIO)
    """ 用于打断检测的音频数据（在数字人回复期间） """
    INTERRUPT_AUDIO = ("interrupt_audio", EngineChannelType.AUDIO)
    """ 摄像头的视频数据 """
    CAMERA_VIDEO = ("camera_video", EngineChannelType.VIDEO)
    """ 虚拟数字人生成的文本数据 """
    AVATAR_TEXT = ("avatar_text", EngineChannelType.TEXT)
    """ 虚拟数字人生成的音频数据 """
    AVATAR_AUDIO = ("avatar_audio", EngineChannelType.AUDIO)
    """ 虚拟数字人生成的视频数据 """
    AVATAR_VIDEO = ("avatar_video", EngineChannelType.VIDEO)
    """ 虚拟数字人生成的运动数据 """
    AVATAR_MOTION_DATA = ("avatar_motion_data", EngineChannelType.MOTION_DATA)
