import asyncio
import json
import uuid
import weakref
from typing import Optional, Dict

import numpy as np
from fastrtc import AsyncAudioVideoStreamHandler, AudioEmitType, VideoEmitType
from loguru import logger

from src.chat_engine.common.client_handler_base import ClientHandlerDelegate, ClientSessionDelegate
from src.chat_engine.common.engine_channel_type import EngineChannelType
from src.chat_engine.data_models.chat_data.chat_data_model import ChatData
from src.chat_engine.data_models.chat_data_type import ChatDataType
from src.chat_engine.data_models.chat_signal import ChatSignal
from src.chat_engine.data_models.chat_signal_type import ChatSignalType, ChatSignalSourceType
from src.engine_utils.interval_counter import IntervalCounter
from aiortc.codecs import vpx

vpx.DEFAULT_BITRATE = 5000000
vpx.MIN_BITRATE = 1000000
vpx.MAX_BITRATE = 10000000


class RtcStream(AsyncAudioVideoStreamHandler):
    """
    WebRTC 流处理器，主要职责：
        处理客户端与服务器之间的音视频数据传输
        管理聊天会话的生命周期
        协调客户端会话委托与聊天引擎的交互
        处理文本聊天消息的发送和接收

        session_id: 会话ID，用于标识特定的RTC会话
        expected_layout: 音频声道布局，默认为"mono"(单声道)
        input_sample_rate: 输入音频采样率，16000Hz
        output_sample_rate: 输出音频采样率，24000Hz
        output_frame_size: 输出音频帧大小，480样本
        fps: 视频帧率，30帧/秒
        stream_start_delay: 流启动延迟，0.5秒
    """

    def __init__(self,
                 session_id: Optional[str],
                 expected_layout="mono",
                 input_sample_rate=16000,
                 output_sample_rate=24000,
                 output_frame_size=480,
                 fps=30,
                 stream_start_delay=0.5,
                 ):
        super().__init__(
            expected_layout=expected_layout,
            input_sample_rate=input_sample_rate,
            output_sample_rate=output_sample_rate,
            output_frame_size=output_frame_size,
            fps=fps
        )
        self.client_handler_delegate: Optional[ClientHandlerDelegate] = None
        self.client_session_delegate: Optional[ClientSessionDelegate] = None

        self.weak_factory: Optional[weakref.ReferenceType[RtcStream]] = None

        self.session_id = session_id
        self.stream_start_delay = stream_start_delay

        self.chat_channel = None
        self.first_audio_emitted = False

        self.quit = asyncio.Event()
        self.last_frame_time = 0

        self.emit_counter = IntervalCounter("emit counter")

        self.start_time = None
        self.timestamp_base = self.input_sample_rate

        self.streams: Dict[str, RtcStream] = {}

    # copy is used as create_instance in fastrtc
    def copy(self, **kwargs) -> AsyncAudioVideoStreamHandler:
        """
        会话创建,生成或获取会话ID,创建新的 RtcStream 实例
        会话委托初始化：
            调用 client_handler_delegate.start_session() 创建客户端会话委托
            将新会话添加到 streams 字典中管理
        """
        try:
            if self.client_handler_delegate is None:
                raise Exception("ClientHandlerDelegate is not set.")
            session_id = kwargs.get("webrtc_id", None)
            if session_id is None:
                session_id = uuid.uuid4().hex
            new_stream = RtcStream(
                session_id,
                expected_layout=self.expected_layout,
                input_sample_rate=self.input_sample_rate,
                output_sample_rate=self.output_sample_rate,
                output_frame_size=self.output_frame_size,
                fps=self.fps,
                stream_start_delay=self.stream_start_delay,
            )
            new_stream.weak_factory = weakref.ref(self)
            new_session_delegate = self.client_handler_delegate.start_session(
                session_id=session_id,
                timestamp_base=self.input_sample_rate,
            )
            new_stream.client_session_delegate = new_session_delegate
            if session_id in self.streams:
                msg = f"Stream {session_id} already exists."
                raise RuntimeError(msg)
            self.streams[session_id] = new_stream
            return new_stream
        except Exception as e:
            logger.opt(exception=True).error(f"Failed to create stream: {e}")
            raise

    async def emit(self) -> AudioEmitType:
        """
        负责从服务器向客户端发送音频数据：
        初始化处理：
            首次发送前清空客户端数据
            设置 first_audio_emitted 标志
        音频数据获取与发送：
            从客户端会话委托获取音频数据
            提取音频数组并返回给WebRTC框架发送
        性能监控：
            使用 IntervalCounter 记录发送性能
        """
        try:
            # if not self.args_set.is_set():
            # await self.wait_for_args()

            if not self.first_audio_emitted:
                self.client_session_delegate.clear_data()
                self.first_audio_emitted = True

            while not self.quit.is_set():
                chat_data = await self.client_session_delegate.get_data(EngineChannelType.AUDIO)
                if chat_data is None or chat_data.data is None:
                    continue
                audio_array = chat_data.data.get_main_data()
                if audio_array is None:
                    continue
                sample_num = audio_array.shape[-1]
                self.emit_counter.add_property("audio_emit", sample_num / self.output_sample_rate)
                return self.output_sample_rate, audio_array
        except Exception as e:
            logger.opt(exception=e).error(f"Error in emit: ")
            raise

    async def video_emit(self) -> VideoEmitType:
        """
        负责从服务器向客户端发送视频数据：
        同步处理：
            确保在首次音频发送后再发送视频
            添加启动延迟避免过早发送
        视频数据处理：
            从客户端会话委托获取视频帧数据
            提取并返回视频帧给WebRTC框架
        """
        try:
            if not self.first_audio_emitted:
                await asyncio.sleep(0.1)
            while not self.quit.is_set():
                video_frame_data: ChatData = await self.client_session_delegate.get_data(EngineChannelType.VIDEO)
                if video_frame_data is None or video_frame_data.data is None:
                    continue
                frame_data = video_frame_data.data.get_main_data().squeeze()
                if frame_data is None:
                    continue
                self.emit_counter.add_property("video_emit")
                return frame_data
        except Exception as e:
            logger.opt(exception=e).error(f"Error in video_emit: ")
            raise

    async def receive(self, frame: tuple[int, np.ndarray]):
        """
        处理从客户端接收到的音频数据：
        时间检查：
            检查是否超过启动延迟时间
            避免在连接建立初期处理数据
        数据转发：
            获取当前时间戳
            将音频数据通过会话委托的 put_data 方法提交给聊天引擎
        """
        if self.client_session_delegate is None:
            return
        timestamp = self.client_session_delegate.get_timestamp()
        if timestamp[0] / timestamp[1] < self.stream_start_delay:
            return
        _, array = frame
        self.client_session_delegate.put_data(
            EngineChannelType.AUDIO,
            array,
            timestamp,
            self.input_sample_rate,
        )

    async def video_receive(self, frame):
        """
        处理从客户端接收到的视频数据：
        时间同步：
            检查启动延迟
            确保连接稳定后再处理视频数据
        数据处理：
            获取时间戳
            通过会话委托提交视频数据
        """
        if self.client_session_delegate is None:
            return
        timestamp = self.client_session_delegate.get_timestamp()
        if timestamp[0] / timestamp[1] < self.stream_start_delay:
            return
        self.client_session_delegate.put_data(
            EngineChannelType.VIDEO,
            frame,
            timestamp,
            self.fps,
        )

    def set_channel(self, channel):
        """
        设置和管理数据通道，处理文本聊天功能：
        通道初始化：
            调用父类方法设置通道
            保存聊天通道引用
        文本聊天处理：
            创建异步任务处理文本数据发送
            监听通道消息接收
        消息类型处理：
            stop_chat: 处理中断信号
            chat: 处理用户聊天输入
            触发相应的聊天信号和数据提交
        """
        super().set_channel(channel)
        self.chat_channel = channel

        async def process_chat_history():
            role = None
            chat_id = None
            while not self.quit.is_set():
                chat_data = await self.client_session_delegate.get_data(EngineChannelType.TEXT)
                if chat_data is None or chat_data.data is None:
                    continue
                logger.info(f"Got chat data {str(chat_data)}")
                current_role = 'human' if chat_data.type == ChatDataType.HUMAN_TEXT else 'avatar'
                chat_id = uuid.uuid4().hex if current_role != role else chat_id
                role = current_role
                self.chat_channel.send(json.dumps({'type': 'chat', 'message': chat_data.data.get_main_data(),
                                                   'id': chat_id, 'role': current_role}))

        asyncio.create_task(process_chat_history())

        @channel.on("message")
        def _(message):
            logger.info(f"Received message Custom: {message}")
            try:
                message = json.loads(message)
            except Exception as e:
                logger.info(e)
                return

            if self.client_session_delegate is None:
                return
            timestamp = self.client_session_delegate.get_timestamp()
            if timestamp[0] / timestamp[1] < self.stream_start_delay:
                return
            logger.info(f'on_chat_datachannel: {message}')

            if message['type'] == 'stop_chat':
                self.client_session_delegate.emit_signal(
                    ChatSignal(
                        type=ChatSignalType.INTERRUPT,
                        source_type=ChatSignalSourceType.CLIENT,
                        source_name="rtc",
                    )
                )
            elif message['type'] == 'chat':
                channel.send(json.dumps({'type': 'avatar_end'}))
                if self.client_session_delegate.shared_states.enable_vad is False:
                    return
                self.client_session_delegate.shared_states.enable_vad = False
                self.client_session_delegate.emit_signal(
                    ChatSignal(
                        # begin a new round of responding
                        type=ChatSignalType.BEGIN,
                        stream_type=ChatDataType.AVATAR_AUDIO,
                        source_type=ChatSignalSourceType.CLIENT,
                        source_name="rtc",
                    )
                )
                self.client_session_delegate.put_data(
                    EngineChannelType.TEXT,
                    message['data'],
                    loopback=True
                )
            # else:

            # channel.send(json.dumps({"type": "chat", "unique_id": unique_id, "message": message}))

    async def on_chat_datachannel(self, message: Dict, channel):
        # {"type":"chat",id:"标识属于同一段话", "message":"Hello, world!"}
        # unique_id = uuid.uuid4().hex
        pass

    def shutdown(self):
        """ 优雅地关闭RTC流 """
        self.quit.set()
        factory = None
        if self.weak_factory is not None:
            factory = self.weak_factory()
        if factory is None:
            factory = self
        self.client_session_delegate = None
        if factory.client_handler_delegate is not None:
            factory.client_handler_delegate.stop_session(self.session_id)
