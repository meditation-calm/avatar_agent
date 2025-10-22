from collections import deque
import modelscope
from torch.multiprocessing import Manager, Queue
import os
import re
import threading
import time
from typing import Dict, Optional, cast
import numpy as np
from loguru import logger
from pydantic import BaseModel
from abc import ABC
import torch

from src.chat_engine.contexts.handler_context import HandlerContext
from src.chat_engine.data_models.chat_engine_config_data import ChatEngineConfigModel
from src.chat_engine.common.handler_base import HandlerBase, HandlerBaseInfo, HandlerDataInfo, HandlerDetail
from src.chat_engine.data_models.chat_data.chat_data_model import ChatData
from src.chat_engine.data_models.chat_data_type import ChatDataType
from src.chat_engine.contexts.session_context import SessionContext
from src.chat_engine.data_models.runtime_data.data_bundle import DataBundle, DataBundleDefinition, DataBundleEntry

from src.handlers.tts.cosyvoice.tts_handler_base import TTSConfig, TTSContext, HandlerTask
from src.handlers.tts.cosyvoice.tts_processor_cosyvoice import TTSCosyVoiceProcessor


class TTSHandler(HandlerBase, ABC):
    def __init__(self):
        super().__init__()
        self.mp = Manager()
        self.tts_input_queue = self.mp.Queue()
        self.tts_output_queue = self.mp.Queue()
        self.multi_process = []
        self.consume_thread = None
        self.task_queue_map = {}
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

    def get_handler_info(self) -> HandlerBaseInfo:
        return HandlerBaseInfo(
            config_model=TTSConfig,
        )

    def get_handler_detail(self, session_context: SessionContext,
                           context: HandlerContext) -> HandlerDetail:
        context = cast(TTSContext, context)
        definition = DataBundleDefinition()
        definition.add_entry(DataBundleEntry.create_audio_entry(
            "avatar_audio", 1, context.config.sample_rate
        ))
        inputs = {
            ChatDataType.AVATAR_TEXT: HandlerDataInfo(
                type=ChatDataType.AVATAR_TEXT,
            )
        }
        outputs = {
            ChatDataType.AVATAR_AUDIO: HandlerDataInfo(
                type=ChatDataType.AVATAR_AUDIO,
                definition=definition,
            )
        }
        return HandlerDetail(
            inputs=inputs, outputs=outputs,
        )

    def load(self, engine_config: ChatEngineConfigModel, handler_config: Optional[BaseModel] = None):
        if isinstance(handler_config, TTSConfig):  
            if not os.path.isabs(handler_config.model_name) and handler_config.model_name is not None:
                """ 下载模型 """
                modelscope.snapshot_download(handler_config.model_name)

            for i in range(handler_config.process_num):
                process = TTSCosyVoiceProcessor(self.handler_root, handler_config,
                                                self.tts_input_queue, self.tts_output_queue)
                process.start()
                self.multi_process.append(process)
            self.tts_output_queue.get()

        def consumer(task_queue_map: dict[str, deque], tts_output_queue: Queue):
            while True:
                logger.debug(f"tts output {len(task_queue_map.keys()), tts_output_queue.qsize()}")
                try:
                    output = tts_output_queue.get(timeout=1)
                except Exception as e:
                    logger.debug(e)
                    continue
                logger.debug(f'output {output}')
                key = output['key']
                audio = output['tts_speech']
                session_id = output['session_id']
                taskDeque = task_queue_map.get(session_id)
                if taskDeque is None:
                    continue
                for task in taskDeque:
                    if task is not None and task.id == key:
                        task.result_queue.put(audio)
                        break
        self.consume_thread = threading.Thread(target=consumer, args=[self.task_queue_map, self.tts_output_queue])
        self.consume_thread.start()
        
    @staticmethod
    def _create_message(text: str):
        if text is None:
            return None
        msg = {"role": "user", "content": text}
        return msg

    def create_context(self, session_context, handler_config=None):
        if not isinstance(handler_config, TTSConfig):
            handler_config = TTSConfig()
        context = TTSContext(session_context.session_info.session_id)
        context.config = handler_config
        return context

    def start_context(self, session_context, context: HandlerContext):
        context = cast(TTSContext, context)
        output_definition = self.get_handler_detail(session_context, context).outputs.get(ChatDataType.AVATAR_AUDIO).definition

        def task_consumer(task_inner_queue: deque, callback: callable):
            while True:
                if len(task_inner_queue) == 0:
                    time.sleep(0.03)
                    continue
                task = task_inner_queue[0]
                task = cast(HandlerTask, task)
                if task is None:
                    break
                logger.debug(f'get task audio {len(task_inner_queue), task.result_queue.qsize()}')
                try:
                    audio = task.result_queue.get(timeout=1)
                    if audio is not None:
                        output = DataBundle(output_definition)
                        output.set_main_data(audio)
                        output.add_meta("avatar_speech_end", False if not task.speech_end else True)
                        output.add_meta("speech_id", task.speech_id)
                        callback(output)
                        if context.dump_audio:
                            dump_audio = audio
                            context.audio_dump_file.write(dump_audio.tobytes())
                    else:
                        task_inner_queue.popleft()
                except Exception as e:
                    logger.debug(e)
  
        context.task_consumer_thread = threading.Thread(target=task_consumer, args=[context.task_queue, context.submit_data])
        context.task_consumer_thread.start()
        self.task_queue_map[context.session_id] = context.task_queue

    def filter_text(self, text):
        pattern = r"[^a-zA-Z0-9\u4e00-\u9fff,.\~!?，。！？ ]"  # 匹配不在范围内的字符
        filtered_text = re.sub(pattern, "", text)
        return filtered_text

    def handle(self, context: HandlerContext, inputs: ChatData,
               output_definitions: Dict[ChatDataType, HandlerDataInfo]):
        context = cast(TTSContext, context)
        if inputs.type == ChatDataType.AVATAR_TEXT:
            text = inputs.data.get_main_data()
        else:
            return
        speech_id = inputs.data.get_meta("speech_id", context.session_id)

        if text is not None:
            text = re.sub(r"<\|.*?\|>", "", text)
            context.input_text += self.filter_text(text)

        text_end = inputs.data.get_meta("avatar_text_end", False)
        if not text_end:
            sentences = re.split(r'(?<=[,.~!?，。！？])', context.input_text)
            if len(sentences) > 1:  # 至少有一个完整句子
                complete_sentences = sentences[:-1]  # 完整句子
                context.input_text = sentences[-1]  # 剩余的未完成部分

                # 对完整句子进行处理
                for sentence in complete_sentences:
                    if len(sentence.strip()) < 1:
                        continue
                    logger.info('current sentence' + sentence)
                    task = HandlerTask(speech_id=speech_id)
                    tts_info = {
                        "text": sentence + '。',
                        "key": task.id,
                        "session_id": context.session_id
                    }
                    self.tts_input_queue.put(tts_info)
                    context.task_queue.append(task)
        else:
            logger.info('last sentence' + context.input_text)
            if context.input_text is not None and len(context.input_text.strip()) > 0:
                task = HandlerTask(speech_id=speech_id)
                tts_info = {
                    "text": context.input_text,
                    "key": task.id,
                    "session_id": context.session_id
                }
                self.tts_input_queue.put(tts_info)
                context.task_queue.append(task)
            context.input_text = ''
            end_task = HandlerTask(speech_id=speech_id, speech_end=True)
            end_task.result_queue.put(np.zeros(shape=(1, 240), dtype=np.float32))
            end_task.result_queue.put(None)
            logger.info(f"speech end {end_task}")
            context.task_queue.append(end_task)

    def destroy_context(self, context: HandlerContext):
        context = cast(TTSContext, context)
        logger.info('destroy context')
        del self.task_queue_map[context.session_id]
        context.task_queue.clear()
        context.task_queue = None
