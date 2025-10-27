import queue
import threading
import time
from typing import Optional, Dict

import numpy as np
from loguru import logger
from pydantic import BaseModel, Field

from src.chat_engine.common.engine_channel_type import EngineChannelType
from src.chat_engine.contexts.handler_context import HandlerContext
from src.chat_engine.data_models.chat_data.chat_data_model import ChatData
from src.chat_engine.data_models.chat_data_type import ChatDataType
from src.chat_engine.data_models.chat_engine_config_data import HandlerBaseConfigModel
from src.chat_engine.data_models.runtime_data.data_bundle import DataBundleDefinition, DataBundle
from src.engine_utils.general_slicer import SliceContext
from src.handlers.avatar.algo_model import Tts2FaceEvent


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
    def __init__(self, session_id: str, event_in_queue: queue.Queue, event_out_queue: queue.Queue, audio_out_queue: queue.Queue, video_out_queue: queue.Queue, shared_states):
        super().__init__(session_id)
        self.config: Optional[AvatarConfig] = None
        self.event_in_queue = event_in_queue  # 输入事件队列
        self.event_out_queue = event_out_queue  # 输出事件队列
        self.audio_out_queue = audio_out_queue  # 音频输出队列
        self.video_out_queue = video_out_queue  # 视频输出队列
        self.shared_states = shared_states  # 共享状态对象，用于VAD和其他标志位

        self.slice_context: Optional[SliceContext] = None  # 音频切片上下文
        self.output_data_definitions: Dict[ChatDataType, DataBundleDefinition] = {}
        self.event_out_thread: threading.Thread = None  # 输出事件线程
        self.media_out_thread: threading.Thread = None  # 输出音视频线程
        self.loop_running = True
        try:
            self.media_out_thread = threading.Thread(target=self._media_out_loop)
            self.media_out_thread.start()
        except Exception as e:
            logger.opt(exception=True).error(f"Failed to start media_out_thread: {e}")
        try:
            self.event_out_thread = threading.Thread(target=self._event_out_loop)
            self.event_out_thread.start()
        except Exception as e:
            logger.opt(exception=True).error(f"Failed to start event_out_thread: {e}")

    def return_data(self, data: np.ndarray, chat_data_type: ChatDataType) -> None:
        definition = self.output_data_definitions.get(chat_data_type)
        if definition is None:
            logger.error(f"Definition is None, chat_data_type={chat_data_type}")
            return
        data_bundle = DataBundle(definition)
        if chat_data_type.channel_type == EngineChannelType.AUDIO:
            # Ensure audio data is float32 and has correct shape
            if data is not None:
                if data.dtype != np.float32:
                    logger.warning("Audio data dtype is not float32")
                    data = data.astype(np.float32)
                if data.ndim == 1:
                    logger.warning("Audio data ndim is 1")
                    data = data[np.newaxis, ...]
                elif data.ndim == 2 and data.shape[0] != 1:
                    logger.warning("Audio data shape is not [1, N]")
                    data = data[:1, ...]
            else:
                logger.error("Audio data is None")
                data = np.zeros([1, 0], dtype=np.float32)
            data_bundle.set_main_data(data)
        elif chat_data_type.channel_type == EngineChannelType.VIDEO:
            # Ensure video data has batch dimension
            data_bundle.set_main_data(data[np.newaxis, ...])
        else:
            return
        chat_data = ChatData(type=chat_data_type, data=data_bundle)
        self.submit_data(chat_data)

    def _media_out_loop(self) -> None:
        while self.loop_running:
            no_output = True
            if self.audio_out_queue.qsize() > 0:
                try:
                    audio = self.audio_out_queue.get_nowait()
                    self.return_data(audio, ChatDataType.AVATAR_AUDIO)
                    no_output = False
                except Exception as e:
                    logger.opt(exception=True).error(f"Exception when getting audio data: {e}")
            if self.video_out_queue.qsize() > 0:
                try:
                    video = self.video_out_queue.get_nowait()
                    if not isinstance(video, np.ndarray):
                        logger.error(f"video_out_queue got non-ndarray: {type(video)}, content: {str(video)[:100]}")
                        continue
                    self.return_data(video, ChatDataType.AVATAR_VIDEO)
                    no_output = False
                except Exception as e:
                    logger.opt(exception=True).error(f"Exception when getting video data: {e}")
            if no_output:
                time.sleep(0.01)
        logger.info("Media out loop exit")

    def _event_out_loop(self) -> None:
        while self.loop_running:
            try:
                event = self.event_out_queue.get(timeout=0.1)
                if isinstance(event, Tts2FaceEvent):
                    if event == Tts2FaceEvent.SPEAKING_TO_LISTENING:
                        self.shared_states.enable_vad = True
                else:
                    logger.warning(f"event_out_queue got unknown event type: {type(event)}, value: {event}")
            except queue.Empty:
                continue
            except Exception as e:
                logger.opt(exception=True).error(f"Exception: {e}")
        logger.info("Event out loop exit")

    def clear(self) -> None:
        logger.info("Clear MuseTalk Context")
        self.loop_running = False
        self.event_in_queue.put_nowait(Tts2FaceEvent.STOP)
        try:
            self.media_out_thread.join(timeout=5)
        except Exception as e:
            logger.opt(exception=True).error(f"Failed to join media_out_thread: {e}")
        try:
            self.event_out_thread.join(timeout=5)
        except Exception as e:
            logger.opt(exception=True).error(f"Failed to join event_out_thread: {e}")
