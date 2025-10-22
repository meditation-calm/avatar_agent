import queue
import threading
import time
from typing import Optional
from threading import Thread

import librosa
import numpy as np
import torch
from loguru import logger

from src.handlers.avatar.audio_input import SpeechAudio
from src.handlers.avatar.musetalk.avatar_handler_algo import Avatar
from src.handlers.avatar.musetalk.avatar_handler_base import AvatarConfig


class AvatarProcessor:
    def __init__(self, avatar: Avatar, config: AvatarConfig, event_out_queue, audio_output_queue, video_output_queue):
        self.avatar = avatar
        self.config = config
        # Output queues
        self.event_out_queue = event_out_queue
        self.audio_output_queue = audio_output_queue
        self.video_output_queue = video_output_queue
        # Internal queues
        self._audio_queue = queue.Queue()  # 输入音频队列
        self._whisper_queue = queue.Queue()  # Whisper特征队列
        self._unet_queue = queue.Queue()  # Unet输出队列
        self._frame_queue = queue.Queue()  # Video桢队列
        self._frame_id_queue = queue.Queue()  # 帧ID分配队列
        self._compose_queue = queue.Queue()  # 帧合成队列
        self._output_queue = queue.Queue()  # 组合后的输出队列
        # Threading and state
        self._stop_event = threading.Event()
        self._feature_thread: Optional[Thread] = None  # 音频特征提取线程
        self._frame_gen_thread: Optional[Thread] = None
        self._frame_gen_unet_thread: Optional[Thread] = None
        self._frame_gen_vae_thread: Optional[Thread] = None
        self._frame_collect_thread: Optional[Thread] = None
        self._compose_thread: Optional[Thread] = None
        self._session_running = False

    def start(self):
        if self._session_running:
            return
        self._session_running = True
        self._stop_event.clear()
        try:
            self._feature_thread = threading.Thread(target=self._feature_extractor_worker)
            logger.info(f"MuseTalk Processor started.")
        except Exception as e:
            logger.opt(exception=True).error(f"Exception during thread start: {e}")

    def add_audio(self, speech_audio: SpeechAudio):
        """ 将音频段添加到处理队列。不进行音频重采样。 """
        audio_data = speech_audio.audio_data
        if isinstance(audio_data, bytes):
            audio_data = np.frombuffer(audio_data, dtype=np.float32)
        elif isinstance(audio_data, np.ndarray):
            audio_data = audio_data.astype(np.float32)
        else:
            logger.error(f"audio_data must be bytes or np.ndarray, got {type(audio_data)}")
            return
        if len(audio_data) == 0:
            logger.error(f"Input audio is empty, speech_id={speech_audio.speech_id}")
            return
        # 检查音频长度
        if len(audio_data) > self.config.output_audio_sample_rate:
            logger.error(f"Audio segment too long: {len(audio_data)} > {self.config.algo_audio_sample_rate}, "
                         f"speech_id={speech_audio.speech_id}")
            return
        # 检查采样率
        if speech_audio.sample_rate != self.config.output_audio_sample_rate:
            logger.error(f"Sample rate mismatch: {speech_audio.sample_rate} vs {self.config.output_audio_sample_rate}, "
                         f"speech_id={speech_audio.speech_id}")
            return

        try:
            speech_audio.audio_data = audio_data
            self._audio_queue.put(speech_audio, timeout=1)
        except queue.Full:
            logger.opt(exception=True).error(f"Audio queue full, dropping audio segment, "
                                             f"speech_id={speech_audio.speech_id}")
            return

    def _feature_extractor_worker(self):
        """ 提取音频特征的工作线程 """
        if torch.cuda.is_available():
            t0 = time.time()
            warmup_sr = 16000
            dummy_audio = np.zeros(warmup_sr, dtype=np.float32)
            self.avatar.extract_whisper_feature(dummy_audio, warmup_sr)
            torch.cuda.synchronize()
            t1 = time.time()
            logger.info(f"[THREAD_WARMUP] _feature_extractor_worker thread id: {threading.get_ident()} "
                        f"whisper feature warmup done, time: {(t1 - t0) * 1000:.1f} ms")
        while not self._stop_event.is_set():
            try:
                t_start = time.time()
                item = self._audio_queue.get(timeout=1)
                speech_id = item['speech_id']
                speech_end = item['speech_end']
                audio_data = item['audio_data']
                fps = self.config.fps if hasattr(self.config, 'fps') else 25
                # 重采样
                segment = librosa.resample(audio_data, orig_sr=self.config.output_audio_sample_rate,
                                           target_sr=self.config.algo_audio_sample_rate)
                target_len = self.config.algo_audio_sample_rate
                if len(segment) > target_len:
                    logger.error(f"Segment too long: {len(segment)} > {target_len}, speech_id={speech_id}")
                    raise ValueError(f"Segment too long: {len(segment)} > {target_len}")
                if len(segment) < target_len:
                    segment = np.pad(segment, (0, target_len - len(segment)), mode='constant')
                # 提取特征
                whisper_chunks = self.avatar.extract_whisper_feature(segment, self.config.algo_audio_sample_rate)
                """
                计算需要生成的视频帧数：
                    获取原始音频数据长度
                    计算每帧对应的音频样本数（采样率/fps）
                    向上取整计算所需帧数
                """
                orig_audio_data_len = len(audio_data)
                orig_samples_per_frame = self.config.output_audio_sample_rate // fps
                actual_audio_len = orig_audio_data_len
                num_frames = int(np.ceil(actual_audio_len / orig_samples_per_frame))
                """
                确保 whisper 特征数量与音频数据长度匹配：
                    裁剪或填充 whisper_chunks 至所需帧数
                    对音频数据进行填充或截断，使其长度与帧数对齐
                """
                whisper_chunks = whisper_chunks[:num_frames]
                target_audio_len = num_frames * orig_samples_per_frame
                if len(audio_data) < target_audio_len:
                    audio_data = np.pad(audio_data, (0, target_audio_len - len(audio_data)), mode='constant')
                else:
                    audio_data = audio_data[:target_audio_len]
                padded_audio_data_len = len(audio_data)

                """ 处理每个帧的数据"""
                num_chunks = len(whisper_chunks)

                for i in range(num_chunks):
                    # 提取单个 whisper 特征块（保持为 [1, 50, 384] 张量）
                    whisper_chunk = whisper_chunks[i:i + 1]

                    # 从对齐后的音频数据中提取对应的音频片段
                    start_sample = i * orig_samples_per_frame
                    end_sample = start_sample + orig_samples_per_frame
                    audio_segment = audio_data[start_sample:end_sample]

                    # 对音频片段进行填充
                    if len(audio_segment) < orig_samples_per_frame:
                        audio_segment = np.pad(audio_segment, (0, orig_samples_per_frame - len(audio_segment)),
                                               mode='constant')

                    # 判断是否为最后一个块（用于标记语音结束）
                    is_last_chunk = (i == num_chunks - 1)

                    # 放入队列
                    self._whisper_queue.put({
                        'speech_id': speech_id,
                        'speech_end': speech_end and is_last_chunk,
                        'audio_data': audio_segment,  # Single frame's audio
                        'whisper_chunks': whisper_chunk,  # Single chunk as [1, 50, 384]
                    }, timeout=1)

                t_end = time.time()
                logger.info(
                    f"[FEATURE_WORKER] speech_id={speech_id}, "
                    f"total_time={(t_end - t_start) * 1000:.1f}ms, "
                    f"whisper_chunks_frames={whisper_chunks.shape[0]}, "
                    f"audio_data_original_length={orig_audio_data_len}, "
                    f"audio_data_padded_length={padded_audio_data_len}, "
                    f"speech_end={speech_end}")
            except queue.Empty:
                continue
            except Exception as e:
                logger.opt(exception=True).error(f"Exception in _feature_extractor_worker: {e}")
                continue
