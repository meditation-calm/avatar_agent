from enum import IntEnum


class TimeUnitType(IntEnum):
    # 无时间单位，表示数据不具有时间维度或时间无关
    NONE = 0
    # 帧单位，用于视频或图像序列数据，表示按帧计时
    FRAME = 1
    # 音频采样点单位，用于音频数据，表示按采样点计时
    AUDIO_SAMPLE = 2
    # 秒单位，用于一般时间度量
    SECOND = 3
