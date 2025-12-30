from typing import Dict, Optional, cast

import gradio
from fastapi import FastAPI
from loguru import logger

from src.chat_engine.common.client_handler_base import ClientHandlerBase, ClientSessionDelegate
from src.chat_engine.common.engine_channel_type import EngineChannelType
from src.chat_engine.common.handler_base import HandlerDataInfo, HandlerDetail, HandlerBaseInfo
from src.chat_engine.contexts.handler_context import HandlerContext
from src.chat_engine.contexts.session_context import SessionContext
from src.chat_engine.data_models.chat_data.chat_data_model import ChatData
from src.chat_engine.data_models.chat_data_type import ChatDataType
from src.chat_engine.data_models.chat_engine_config_data import HandlerBaseConfigModel, ChatEngineConfigModel
from src.chat_engine.data_models.runtime_data.data_bundle import DataBundleDefinition, DataBundleEntry, VariableSize
from src.handlers.client.stream import Stream
from src.service.rtc.rtc_provider import RTCProvider
from src.service.rtc.rtc_stream import RtcStream
from src.handlers.client.client_handler_base import RtcClientSessionDelegate, ClientRtcConfigModel, ClientRtcContext


class ClientHandlerRtc(ClientHandlerBase):
    """
    RTC客户端处理器
        engine_config：引擎配置
        handler_config：处理器配置
        rtc_streamer_factory：RTC流工厂
        output_bundle_definitions：输出数据包定义
    """
    def __init__(self):
        super().__init__()
        self.engine_config = None
        self.handler_config = None
        self.rtc_streamer_factory: Optional[RtcStream] = None

        self.output_bundle_definitions: Dict[EngineChannelType, DataBundleDefinition] = {}

    def get_handler_info(self) -> HandlerBaseInfo:
        return HandlerBaseInfo(
            config_model=ClientRtcConfigModel,
            client_session_delegate_class=RtcClientSessionDelegate,
        )

    def prepare_rtc_definitions(self):
        # RTC流处理器
        self.rtc_streamer_factory = RtcStream(
            session_id=None,
            expected_layout="mono",
            input_sample_rate=16000,
            output_sample_rate=24000,
            output_frame_size=480,
            fps=30,
            stream_start_delay=0.5,
        )
        self.rtc_streamer_factory.client_handler_delegate = self.handler_delegate

        # 文本、音频、视频的数据包定义
        # 文本定义
        text_output_definition = DataBundleDefinition()
        text_output_definition.add_entry(DataBundleEntry.create_text_entry(
            "human_text",
        ))
        text_output_definition.lockdown()
        self.output_bundle_definitions[EngineChannelType.TEXT] = text_output_definition

        # 音频定义
        audio_output_definition = DataBundleDefinition()
        audio_output_definition.add_entry(DataBundleEntry.create_audio_entry(
            "mic_audio",
            1,
            16000,
        ))
        audio_output_definition.lockdown()
        self.output_bundle_definitions[EngineChannelType.AUDIO] = audio_output_definition

        # 视频定义
        video_output_definition = DataBundleDefinition()
        video_output_definition.add_entry(DataBundleEntry.create_framed_entry(
            "camera_video",
            [VariableSize(), VariableSize(), VariableSize(), 3],
            0,
            30
        ))
        video_output_definition.lockdown()
        self.output_bundle_definitions[EngineChannelType.VIDEO] = video_output_definition

    def load(self, engine_config: ChatEngineConfigModel, handler_config: Optional[HandlerBaseConfigModel] = None):
        """
            加载引擎配置和处理器配置
            调用 prepare_rtc_definitions 准备RTC相关定义
        """
        self.engine_config = engine_config
        self.handler_config = cast(ClientRtcConfigModel, handler_config)
        self.prepare_rtc_definitions()

    def setup_rtc_ui(self, fastapi: FastAPI, ui, parent_block, avatar_config):
        """
        设置RTC用户界面和Web API：
            配置TURN服务器用于NAT穿透
            创建WebRTC流处理器并挂载到FastAPI
            设置初始化配置API端点
            配置前端静态文件服务和路由重定向
            设置Gradio界面组件
        """
        turn_entity = RTCProvider().prepare_rtc_configuration(self.handler_config.turn_config)
        if turn_entity is None:
            turn_entity = RTCProvider().prepare_rtc_configuration(self.engine_config.turn_config)

        webrtc = Stream(
            modality="audio-video",
            mode="send-receive",
            time_limit=self.handler_config.connection_ttl,
            rtc_configuration=turn_entity.rtc_configuration if turn_entity is not None else None,
            handler=self.rtc_streamer_factory,
            concurrency_limit=self.handler_config.concurrent_limit,
        )
        webrtc.mount(fastapi)
        logger.info(f"RTC streamer ready modality: ${webrtc.modality} mode: ${webrtc.mode}")

        @fastapi.get('/health')
        async def health():
            return {"status": "healthy", "service": "rtc"}

        @fastapi.get('/config')
        async def config():
            return {
                "avatar_config": avatar_config,
                "rtc_configuration": turn_entity.rtc_configuration if turn_entity is not None else None,
            }

        @fastapi.post('/webrtc/disconnect/{webrtc_id}')
        async def disconnect_rtc(webrtc_id: str):
            success = webrtc.disconnect_webrtc(webrtc_id)
            if success:
                return {"status": "success", "message": f"Disconnected {webrtc_id}"}
            else:
                return {"status": "failed", "message": f"Failed to disconnect {webrtc_id}"}

        @fastapi.get('/webrtc/active')
        async def rtc_active():
            connections = webrtc.get_active_connections()
            return {"connections": connections}

        # 渲染默认fastrtc页面
        with ui:
            webrtc.ui.render()

    def on_setup_app(self, app: FastAPI, ui: gradio.blocks.Block, parent_block: Optional[gradio.blocks.Block] = None):
        avatar_config = {}
        self.setup_rtc_ui(app, ui, parent_block, avatar_config)

    def create_context(self, session_context: SessionContext,
                       handler_config: Optional[HandlerBaseConfigModel] = None) -> HandlerContext:
        if not isinstance(handler_config, ClientRtcConfigModel):
            handler_config = ClientRtcConfigModel()
        context = ClientRtcContext(session_context.session_info.session_id)
        context.config = handler_config
        return context

    def start_context(self, session_context: SessionContext, handler_context: HandlerContext):
        pass

    def on_setup_session_delegate(self, session_context: SessionContext, handler_context: HandlerContext,
                                  session_delegate: ClientSessionDelegate):
        handler_context = cast(ClientRtcContext, handler_context)
        session_delegate = cast(RtcClientSessionDelegate, session_delegate)

        session_delegate.timestamp_generator = session_context.get_timestamp
        session_delegate.data_submitter = handler_context.data_submitter
        session_delegate.input_data_definitions = self.output_bundle_definitions
        session_delegate.shared_states = session_context.shared_states

        handler_context.client_session_delegate = session_delegate

    def create_handler_detail(self, _session_context, _handler_context):
        inputs = {
            ChatDataType.KEYWORD_TEXT: HandlerDataInfo(
                type=ChatDataType.KEYWORD_TEXT
            ),
            ChatDataType.HUMAN_TEXT: HandlerDataInfo(
                type=ChatDataType.HUMAN_TEXT
            ),
            ChatDataType.AVATAR_TEXT: HandlerDataInfo(
                type=ChatDataType.AVATAR_TEXT
            ),
            ChatDataType.AVATAR_AUDIO: HandlerDataInfo(
                type=ChatDataType.AVATAR_AUDIO
            ),
            ChatDataType.AVATAR_VIDEO: HandlerDataInfo(
                type=ChatDataType.AVATAR_VIDEO
            ),
        }
        outputs = {
            ChatDataType.HUMAN_TEXT: HandlerDataInfo(
                type=ChatDataType.HUMAN_TEXT,
                definition=self.output_bundle_definitions[EngineChannelType.TEXT]
            ),
            ChatDataType.MIC_AUDIO: HandlerDataInfo(
                type=ChatDataType.MIC_AUDIO,
                definition=self.output_bundle_definitions[EngineChannelType.AUDIO]
            ),
            ChatDataType.CAMERA_VIDEO: HandlerDataInfo(
                type=ChatDataType.CAMERA_VIDEO,
                definition=self.output_bundle_definitions[EngineChannelType.VIDEO]
            ),
        }
        return HandlerDetail(
            inputs=inputs,
            outputs=outputs
        )

    def get_handler_detail(self, session_context: SessionContext, context: HandlerContext) -> HandlerDetail:
        return self.create_handler_detail(session_context, context)

    def handle(self, context: HandlerContext, inputs: ChatData,
               output_definitions: Dict[ChatDataType, HandlerDataInfo]):
        """
            处理输入数据
            将接收到的数据（如音频/视频）转发到客户端会话委托的相应队列中
        """
        context = cast(ClientRtcContext, context)
        if context.client_session_delegate is None:
            return
        data_queue = context.client_session_delegate.output_queues.get(inputs.type.channel_type)
        if data_queue is not None:
            data_queue.put_nowait(inputs)

    def destroy_context(self, context: HandlerContext):
        pass
