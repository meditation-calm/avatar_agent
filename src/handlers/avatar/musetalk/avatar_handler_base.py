import queue
import threading
from typing import Optional, Dict

from pydantic import BaseModel, Field

from src.chat_engine.contexts.handler_context import HandlerContext
from src.chat_engine.data_models.chat_data_type import ChatDataType
from src.chat_engine.data_models.chat_engine_config_data import HandlerBaseConfigModel
from src.chat_engine.data_models.runtime_data.data_bundle import DataBundleDefinition
from src.engine_utils.general_slicer import SliceContext


class AvatarConfig(HandlerBaseConfigModel, BaseModel):
    """ MuseTalk 推理相关配置 """
    bbox_shift: int = Field(default=0)  # 边界框偏移值，用于调整面部检测区域
    extra_margin: int = Field(default=10)  # 面部裁剪时的额外边距，默认10像素
    fps: int = Field(default=25)  # 输出视频的帧率，默认25
    audio_padding_length_left: int = Field(default=2)  # 音频左填充长度，默认2
    audio_padding_length_right: int = Field(default=2)  # 音频右填充长度，默认2
    batch_size: int = Field(default=5, ge=2)  # 推理时的批处理大小，默认5
    use_saved_coord: bool = Field(default=False)  # 使用已保存的坐标以节省时间
    saved_coord: bool = Field(default=False)  # 保存坐标供将来使用
    parsing_mode: str = Field(default="jaw")  # 面部融合解析模式
    left_cheek_width: int = Field(default=90)  # 左脸颊区域宽度，默认90
    right_cheek_width: int = Field(default=90)  # 右脸颊区域宽度，默认90
    gpu_id: int = Field(default=0)  # 指定使用的GPU ID，默认为0
    version: str = Field(default="v15")  # MuseTalk版本 v1 or v15
    preparation: bool = Field(default=False)  # 是否重新生成预处理数据
    vae_type: str = Field(default="sd-vae")  # VAE模型类型，默认为"sd-vae"
    """ MuseTalk 其他配置 """
    algo_audio_sample_rate: int = Field(default=16000)  # 输入音频采样率
    output_audio_sample_rate: int = Field(default=24000)  # 输出音频采样率
    avatar_video_path: str = Field(default="")  # avatar初始化头像文件路径
    multi_thread_inference: bool = Field(default=True)  # 多线程推理
    # avatar_model_dir: str = Field(default="models/musetalk/avatar_model")  # Directory for output results
    # debug: bool = Field(default=False)  # Enable debug mode
    # debug_save_handler_audio: bool = Field(default=False)  # Enable debug mode
    # debug_replay_speech_id: str = Field(default="")  # Enable debug mode


class AvatarContext(HandlerContext):
    """
    初始化 MuseTalk 头像处理器上下文：
    调用父类 HandlerContext 的构造函数
    设置各种队列：事件输入/输出队列、音频/视频输出队列
    初始化配置、共享状态和切片上下文
    启动两个后台线程：
        event_out_thread：用于处理事件输出
        media_out_thread：用于输出音视频数据
    """
    def __init__(self, session_id: str, event_in_queue: queue.Queue, event_out_queue: queue.Queue, audio_out_queue: queue.Queue, video_out_queue: queue.Queue, shared_status):
        super().__init__(session_id)
        self.config: Optional[AvatarConfig] = None
        self.event_in_queue = event_in_queue  # 输入事件队列
        self.event_out_queue = event_out_queue  # 输出事件队列
        self.audio_out_queue = audio_out_queue  # 音频输出队列
        self.video_out_queue = video_out_queue  # 视频输出队列
        self.shared_status = shared_status  # 共享状态对象，用于VAD和其他标志位

        self.slice_context: Optional[SliceContext] = None  # 音频切片上下文
        self.output_data_definitions: Dict[ChatDataType, DataBundleDefinition] = {}
        self.event_out_thread: threading.Thread = None  # 输出事件线程
        self.media_out_thread: threading.Thread = None  # 输出音视频线程
        self.loop_running = True  # Control flag for threads
