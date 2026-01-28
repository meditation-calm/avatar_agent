import os
from abc import ABC
from typing import cast, Dict, Optional

import numpy as np
import sherpa_onnx
from loguru import logger
from uuid import uuid4

from src.chat_engine.common.handler_base import HandlerBase, HandlerBaseInfo, HandlerDetail, HandlerDataInfo
from src.chat_engine.contexts.handler_context import HandlerContext
from src.chat_engine.contexts.session_context import SessionContext
from src.chat_engine.data_models.chat_data.chat_data_model import ChatData
from src.chat_engine.data_models.chat_data_type import ChatDataType
from src.chat_engine.data_models.chat_engine_config_data import ChatEngineConfigModel
from src.chat_engine.data_models.runtime_data.data_bundle import DataBundleDefinition, DataBundleEntry, DataBundle
from src.handlers.kis.kis_handler_base import KISConfig, KISContext


class KISHandler(HandlerBase, ABC):
    """
    关键词语音打断处理器 (Keyword-based Interrupt Switch)
    在数字人回复期间检测打断关键词，触发打断流程
    """
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.kws_stream = None

    def get_handler_info(self):
        """返回处理器的基本信息"""
        return HandlerBaseInfo(
            config_model=KISConfig
        )

    def load(self, engine_config: ChatEngineConfigModel, handler_config=None):
        """加载 Sherpa KWS 模型用于打断关键词检测"""
        if not isinstance(handler_config, KISConfig):
            handler_config = KISConfig()
        model_path = os.path.join(engine_config.model_root, handler_config.model_name)
        self.model = sherpa_onnx.KeywordSpotter(
            tokens=os.path.join(model_path, handler_config.tokens),
            encoder=os.path.join(model_path, handler_config.encoder),
            decoder=os.path.join(model_path, handler_config.decoder),
            joiner=os.path.join(model_path, handler_config.joiner),
            num_threads=handler_config.num_threads,
            sample_rate=handler_config.sample_rate,
            keywords_score=handler_config.keywords_score,
            keywords_threshold=handler_config.keywords_threshold,
            keywords_file=os.path.join(model_path, handler_config.keywords_file),
            provider="cpu"
        )
        self.kws_stream = self.model.create_stream()
        logger.info(f"Loaded KIS handler with model from {model_path}")

    def create_context(self, session_context: SessionContext, handler_config=None) -> HandlerContext:
        """创建 KISContext 实例"""
        context = KISContext(session_context.session_info.session_id)
        context.shared_states = session_context.shared_states
        if isinstance(handler_config, KISConfig):
            context.config = handler_config
            context.sample_rate = handler_config.sample_rate
        return context

    def start_context(self, session_context, handler_context):
        pass

    def get_handler_detail(self, session_context: SessionContext,
                           context: HandlerContext) -> HandlerDetail:
        """返回处理器的输入输出定义"""
        event_definition = DataBundleDefinition()
        event_definition.add_entry(DataBundleEntry.create_text_entry("human_event"))
        
        inputs = {
            ChatDataType.INTERRUPT_AUDIO: HandlerDataInfo(
                type=ChatDataType.INTERRUPT_AUDIO
            )
        }
        outputs = {
            ChatDataType.HUMAN_EVENT: HandlerDataInfo(
                type=ChatDataType.HUMAN_EVENT,
                definition=event_definition
            )
        }
        return HandlerDetail(
            inputs=inputs,
            outputs=outputs,
        )

    def handle(self, context: HandlerContext, inputs: ChatData,
               output_definitions: Dict[ChatDataType, HandlerDataInfo]):
        """处理打断音频数据，检测打断关键词"""
        context = cast(KISContext, context)
        event_definition = output_definitions.get(ChatDataType.HUMAN_EVENT).definition
        
        if inputs.type != ChatDataType.INTERRUPT_AUDIO:
            return
        
        # 如果已经在等待前端确认，跳过处理
        if context.interrupt_pending:
            return
        
        # 收集音频数据
        audio = inputs.data.get_main_data()
        if audio is not None:
            audio = audio.squeeze()
            context.output_audios.append(audio)
        
        # 检查是否有足够的音频进行检测（至少1秒）
        if len(context.output_audios) < 1:
            return
        
        # 合并音频进行关键词检测
        output_audio = np.concatenate(context.output_audios)
        context.output_audios.clear()
        
        # 准备音频数据
        kws_audio = output_audio
        if output_audio.dtype != np.float32:
            kws_audio = kws_audio.astype(np.float32)
        kws_audio = kws_audio / 32768.0 if kws_audio.max() > 1.0 else kws_audio
        
        # 执行关键词检测
        sample_rate = context.sample_rate
        tail_paddings = np.zeros(int(1 * sample_rate), dtype=np.float32)
        self.kws_stream.accept_waveform(sample_rate, kws_audio)
        self.kws_stream.accept_waveform(sample_rate, tail_paddings)
        self.kws_stream.input_finished()
        
        keyword: Optional[str] = None
        while self.model.is_ready(self.kws_stream):
            self.model.decode_stream(self.kws_stream)
            result = self.model.get_result(self.kws_stream)
            logger.info(f"KIS detection result: {result}")
            if result and result.strip() != '':
                keyword = result
        
        self.model.reset_stream(self.kws_stream)
        
        # 检查是否检测到打断关键词
        if keyword and keyword in context.config.interrupt_keywords:
            logger.info(f"Interrupt keyword detected: {keyword}")
            context.interrupt_keyword_detected = True
            context.interrupt_pending = True
            request_id = uuid4().hex
            context.pending_request_id = request_id
            
            # 发送打断信号给前端
            event = DataBundle(event_definition)
            event.set_main_data({
                "handler": "kis",
                "event": "interrupt_request",
                "keyword": keyword,
                "request_id": request_id,
            })
            context.submit_data(ChatData(type=ChatDataType.HUMAN_EVENT, data=event))
            
            # 发送信号给会话管理器，通知需要打断
            # 注意：实际的信号发送需要通过 ChatSession，这里先通过事件通知前端
            # 前端确认后会调用 interrupt() 方法
            logger.info("KIS: Interrupt keyword detected, waiting for frontend confirmation")

    def interrupt(self, context: HandlerContext):
        """
        处理打断确认，触发后端各个 handler 的打断
        这个方法会在前端确认打断后，由 ChatSession 调用
        """
        context = cast(KISContext, context)
        if not context.interrupt_keyword_detected:
            return
        
        logger.info("KIS: Interrupt confirmed, triggering backend handlers")
        
        # 重置状态
        context.interrupt_pending = False
        context.interrupt_keyword_detected = False
        context.pending_request_id = None
        context.output_audios.clear()
        
        # 重新启用 VAD
        if context.shared_states:
            context.shared_states.enable_vad = True
        
        # 发送打断完成事件
        event_definition = DataBundleDefinition()
        event_definition.add_entry(DataBundleEntry.create_text_entry("human_event"))
        event = DataBundle(event_definition)
        event.set_main_data({
            "handler": "kis",
            "event": "interrupt_confirmed"
        })
        context.submit_data(ChatData(type=ChatDataType.HUMAN_EVENT, data=event))
        
        # 通过 session_context 发送打断信号，触发其他 handler 的打断
        # 注意：这里需要通过 ChatSession 来发送信号，但 ChatSession 的引用需要通过其他方式获取
        # 目前通过事件通知，实际的信号发送由 ChatSession 的 emit_signal 方法处理
        logger.info("KIS: Backend handlers will be interrupted by ChatSession")

    def destroy_context(self, context: HandlerContext):
        pass

