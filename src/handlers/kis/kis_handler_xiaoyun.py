import os.path
from abc import ABC
from typing import cast, Dict, Optional

import numpy as np
import torch
from funasr import AutoModel
from loguru import logger
from uuid import uuid4

from src.chat_engine.common.handler_base import HandlerBase, HandlerBaseInfo, HandlerDetail, HandlerDataInfo
from src.chat_engine.contexts.handler_context import HandlerContext
from src.chat_engine.contexts.session_context import SessionContext
from src.chat_engine.data_models.chat_data.chat_data_model import ChatData
from src.chat_engine.data_models.chat_data_type import ChatDataType
from src.chat_engine.data_models.chat_engine_config_data import ChatEngineConfigModel
from src.chat_engine.data_models.chat_signal import ChatSignal
from src.chat_engine.data_models.chat_signal_type import ChatSignalType, ChatSignalSourceType
from src.chat_engine.data_models.runtime_data.data_bundle import DataBundleDefinition, DataBundleEntry, DataBundle
from src.engine_utils.directory_info import DirectoryInfo
from src.handlers.kis.kis_handler_base import KISContext


class KISXiaoyunConfig:
    """Xiaoyun KIS 配置（简化版，因为 Xiaoyun 只支持单个关键词）"""
    def __init__(self, model_name="iic/speech_charctc_kws_phone-xiaoyun", 
                 interrupt_keyword="小云小云", speaking_threshold=0.1):
        self.model_name = model_name
        self.interrupt_keyword = interrupt_keyword  # 单个关键词
        self.speaking_threshold = speaking_threshold


class KISHandler(HandlerBase, ABC):
    """
    基于 Xiaoyun 的关键词语音打断处理器 (Keyword-based Interrupt Switch)
    在数字人回复期间检测打断关键词，触发打断流程
    注意：Xiaoyun 只支持单个关键词检测
    """
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.model_name = 'iic/speech_charctc_kws_phone-xiaoyun'
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        logger.info(f"KIS Xiaoyun Using device: {self.device}")

    def get_handler_info(self):
        """返回处理器的基本信息"""
        # 使用一个简单的配置模型，因为需要兼容现有的 KISConfig
        from src.handlers.kis.kis_handler_base import KISConfig
        return HandlerBaseInfo(
            config_model=KISConfig
        )

    def load(self, engine_config: ChatEngineConfigModel, handler_config=None):
        """加载 Xiaoyun KWS 模型用于打断关键词检测"""
        from src.handlers.kis.kis_handler_base import KISConfig
        
        if not isinstance(handler_config, KISConfig):
            handler_config = KISConfig()
        
        # 从 interrupt_keywords 中取第一个关键词（Xiaoyun 只支持单个）
        interrupt_keyword = handler_config.interrupt_keywords[0] if handler_config.interrupt_keywords else "小云小云"
        
        # 如果配置了 model_name 且是 xiaoyun 相关的，使用它；否则使用默认值
        if hasattr(handler_config, 'model_name') and 'xiaoyun' in handler_config.model_name.lower():
            self.model_name = handler_config.model_name
        else:
            # 默认使用 xiaoyun 模型
            self.model_name = 'iic/speech_charctc_kws_phone-xiaoyun'
        
        self.model = AutoModel(
            model=self.model_name,
            keywords=interrupt_keyword,
            output_dir=os.path.join(DirectoryInfo.get_cache_dir()),
            disable_update=True
        )
        logger.info(f"Loaded KIS Xiaoyun handler with model {self.model_name}, keyword: {interrupt_keyword}")

    def create_context(self, session_context: SessionContext, handler_config=None) -> HandlerContext:
        """创建 KISContext 实例"""
        from src.handlers.kis.kis_handler_base import KISConfig
        
        context = KISContext(session_context.session_info.session_id)
        context.shared_states = session_context.shared_states
        if isinstance(handler_config, KISConfig):
            context.config = handler_config
            context.sample_rate = 16000  # Xiaoyun 固定 16kHz
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
        if len(context.output_audios) == 0:
            return
        
        # 计算总音频长度（样本数）
        total_samples = sum(len(audio) for audio in context.output_audios)
        if total_samples < context.sample_rate:  # 16kHz 下，1秒 = 16000 个样本
            return
        
        # 合并音频进行关键词检测
        output_audio = np.concatenate(context.output_audios)
        context.output_audios.clear()
        
        # 准备音频数据
        if output_audio.dtype != np.float32:
            output_audio = output_audio.astype(np.float32)
        if output_audio.max() > 1.0:
            output_audio = output_audio / 32768.0
        
        # 执行关键词检测（Xiaoyun 方式）
        try:
            res = self.model.generate(input=output_audio, cache={})
            logger.info(f"KIS Xiaoyun detection result: {res}")
            
            if res and isinstance(res, list) and len(res) > 0:
                result = res[0]
                if 'text' in result:
                    text_parts = result['text'].split()
                    if len(text_parts) > 1:
                        confidence = float(text_parts[-1])
                        # 检查置信度是否达到阈值
                        speaking_threshold = 0.5  # 默认阈值，可以从配置读取
                        if hasattr(context.config, 'keywords_threshold'):
                            speaking_threshold = context.config.keywords_threshold
                        
                        if confidence > speaking_threshold:
                            # 检测到打断关键词（Xiaoyun 不返回具体关键词，只要检测到就触发）
                            interrupt_keyword = context.config.interrupt_keywords[0] if context.config.interrupt_keywords else "小云小云"
                            logger.info(f"KIS Xiaoyun interrupt keyword detected: {interrupt_keyword} (confidence: {confidence})")
                            context.interrupt_keyword_detected = True
                            context.interrupt_pending = True
                            request_id = uuid4().hex
                            context.pending_request_id = request_id
                            
                            # 发送打断信号给前端
                            payload = {
                                "handler": "kis",
                                "event": "interrupt_request",
                                "keyword": interrupt_keyword,  # 使用配置的第一个关键词
                                "request_id": request_id,
                            }
                            logger.info(f"KIS Xiaoyun sending interrupt request: {payload}")
                            event = DataBundle(event_definition)
                            event.set_main_data(payload)
                            context.submit_data(ChatData(type=ChatDataType.HUMAN_EVENT, data=event))
                            
                            logger.info("KIS Xiaoyun: Interrupt keyword detected, waiting for frontend confirmation")
        except Exception as e:
            logger.error(f"KIS Xiaoyun detection error: {e}")

    def interrupt(self, context: HandlerContext):
        """
        处理打断确认，触发后端各个 handler 的打断
        这个方法会在前端确认打断后，由 ChatSession 调用
        """
        context = cast(KISContext, context)
        if not context.interrupt_keyword_detected:
            return
        
        logger.info("KIS Xiaoyun: Interrupt confirmed, triggering backend handlers")
        
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
        logger.info("KIS Xiaoyun: Backend handlers will be interrupted by ChatSession")

        # FIX: 主动触发 ChatSession 的打断信号
        if self.engine:
            engine = self.engine()
            if engine and context.session_id in engine.sessions:
                session = engine.sessions[context.session_id]
                signal = ChatSignal(
                    type=ChatSignalType.INTERRUPT,
                    source_type=ChatSignalSourceType.HANDLER,
                    source_name="kis"
                )
                session.emit_signal(signal)
            else:
                logger.warning(f"Could not find session {context.session_id} to emit interrupt signal")
        else:
            logger.warning("Engine reference is missing in KISHandler")

    def destroy_context(self, context: HandlerContext):
        pass

