import os
import queue
import threading
import time
from typing import Optional
from threading import Thread

import av
import librosa
import numpy as np
import soundfile as sf
import torch
from loguru import logger

from src.handlers.avatar.algo_model import AvatarStatus, AudioFrame, VideoFrame, Tts2FaceEvent
from src.handlers.avatar.audio_input import SpeechAudio
from src.handlers.avatar.musetalk.avatar_handler_algo import Avatar
from src.handlers.avatar.musetalk.avatar_handler_base import AvatarConfig


class AvatarProcessor:
    def __init__(self, avatar: Avatar, config: AvatarConfig, event_out_queue, audio_output_queue, video_output_queue):
        self.avatar = avatar
        self.config = config
        self.algo_audio_sample_rate = config.algo_audio_sample_rate
        self.output_audio_sample_rate = config.output_audio_sample_rate
        # Output queues
        self.event_out_queue = event_out_queue
        self.audio_output_queue = audio_output_queue
        self.video_output_queue = video_output_queue
        # Internal queues
        self._audio_queue = queue.Queue()  # 输入音频队列
        self._whisper_queue = queue.Queue()  # Whisper特征队列
        self._unet_queue = queue.Queue()  # Unet输出队列
        self._vae_queue = queue.Queue()  # VAE输出队列
        self._frame_queue = queue.Queue()  # Video桢队列
        self._frame_id_queue = queue.Queue()  # 帧ID分配队列
        self._output_queue = queue.Queue()  # 组合后的输出队列
        # Threading and state
        self._stop_event = threading.Event()
        self._feature_thread: Optional[Thread] = None  # 音频特征提取线程
        self._frame_gen_thread: Optional[Thread] = None  # 单线程处理推理桢生成线程
        self._frame_gen_unet_thread: Optional[Thread] = None  # UNet帧生成线程
        self._frame_gen_vae_thread: Optional[Thread] = None  # VAE帧生成线程
        self._frame_gen_avatar: Optional[Thread] = None  # avatar视频帧生成线程
        self._frame_collect_thread: Optional[Thread] = None  # 输出视频帧和音频帧线程
        self._session_running = False
        # Audio cache for each speech_id
        self._audio_cache = {}
        self._frame_id_lock = threading.Lock()

    def start(self):
        if self._session_running:
            return
        self._session_running = True
        self._stop_event.clear()
        try:
            self._feature_thread = threading.Thread(target=self._feature_extractor_worker)
            if self.config.multi_thread_inference:
                self._frame_gen_unet_thread = threading.Thread(target=self._frame_generator_unet_worker)
                self._frame_gen_vae_thread = threading.Thread(target=self._frame_generator_vae_worker)
            else:
                self._frame_gen_thread = threading.Thread(target=self._frame_generator_worker)
            self._frame_gen_avatar = threading.Thread(target=self._frame_generator_avatar_worker)
            self._frame_collect_thread = threading.Thread(target=self._frame_collector_worker)
            self._feature_thread.start()
            if self.config.multi_thread_inference:
                self._frame_gen_unet_thread.start()
                self._frame_gen_vae_thread.start()
            else:
                self._frame_gen_thread.start()
            self._frame_collect_thread.start()
            logger.info(f"MuseTalk Processor started.")
        except Exception as e:
            logger.opt(exception=True).error(f"Exception during thread start: {e}")

    def stop(self):
        if not self._session_running:
            logger.warning("Processor not running. Skip stop.")
            return
        self._session_running = False
        self._stop_event.set()
        try:
            if self._feature_thread:
                self._feature_thread.join(timeout=5)
                if self._feature_thread.is_alive():
                    logger.warning("Feature thread did not exit in time.")
            if self._frame_gen_thread:
                self._frame_gen_thread.join(timeout=5)
                if self._frame_gen_thread.is_alive():
                    logger.warning("Frame generator thread did not exit in time.")
            if self._frame_gen_unet_thread:
                self._frame_gen_unet_thread.join(timeout=5)
                if self._frame_gen_unet_thread.is_alive():
                    logger.warning("Frame generator unet thread did not exit in time.")
            if self._frame_gen_vae_thread:
                self._frame_gen_vae_thread.join(timeout=5)
                if self._frame_gen_vae_thread.is_alive():
                    logger.warning("Frame generator vae thread did not exit in time.")
            if self._frame_gen_avatar:
                self._frame_gen_avatar.join(timeout=5)
                if self._frame_gen_avatar.is_alive():
                    logger.warning("Frame generator avatar thread did not exit in time.")
            if self._frame_collect_thread:
                self._frame_collect_thread.join(timeout=5)
                if self._frame_collect_thread.is_alive():
                    logger.warning("Frame collector thread did not exit in time.")
            self._clear_queues()
        except Exception as e:
            logger.opt(exception=True).error(f"Exception during thread stop: {e}")
        logger.info(f"MuseTalk Processor stopped.")

    def _clear_queues(self):
        """ 清空队列 """
        with self._frame_id_lock:
            for q in [self._audio_queue, self._whisper_queue, self._unet_queue, self._vae_queue,
                      self._frame_queue, self._frame_id_queue, self._output_queue]:
                while not q.empty():
                    try:
                        q.get_nowait()
                    except Exception as e:
                        logger.opt(exception=True).warning(f"Exception in _clear_queues: {e}")
                        pass

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
        if len(audio_data) > self.output_audio_sample_rate:
            logger.error(f"Audio segment too long: {len(audio_data)} > {self.algo_audio_sample_rate}, "
                         f"speech_id={speech_audio.speech_id}")
            return
        # 检查采样率
        if speech_audio.sample_rate != self.output_audio_sample_rate:
            logger.error(f"Sample rate mismatch: {speech_audio.sample_rate} vs {self.output_audio_sample_rate}, "
                         f"speech_id={speech_audio.speech_id}")
            return

        try:
            self._audio_queue.put({
                "speech_id": speech_audio.speech_id,
                "speech_end": speech_audio.speech_end,
                "audio_data": audio_data
            }, timeout=1)
        except queue.Full:
            logger.opt(exception=True).error(f"Audio queue full, dropping audio segment, "
                                             f"speech_id={speech_audio.speech_id}")
            return

    def _feature_extractor_worker(self):
        """ 提取音频特征的工作线程 """
        if torch.cuda.is_available():
            t0 = time.time()
            warmup_sr = self.algo_audio_sample_rate
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
                segment = librosa.resample(audio_data, orig_sr=self.output_audio_sample_rate,
                                           target_sr=self.algo_audio_sample_rate)
                target_len = self.algo_audio_sample_rate
                if len(segment) > target_len:
                    logger.error(f"Segment too long: {len(segment)} > {target_len}, speech_id={speech_id}")
                    raise ValueError(f"Segment too long: {len(segment)} > {target_len}")
                if len(segment) < target_len:
                    segment = np.pad(segment, (0, target_len - len(segment)), mode='constant')
                # 提取特征
                whisper_chunks = self.avatar.extract_whisper_feature(segment, self.algo_audio_sample_rate)
                """
                计算需要生成的视频帧数：
                    获取原始音频数据长度
                    计算每帧对应的音频样本数（采样率/fps）
                    向上取整计算所需帧数
                """
                orig_audio_data_len = len(audio_data)
                orig_samples_per_frame = self.output_audio_sample_rate // fps
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

    def _frame_generator_worker(self):
        fps = self.config.fps
        orig_samples_per_frame = int(self.output_audio_sample_rate / fps)
        batch_size = self.config.batch_size
        max_speaking_buffer = batch_size * 5
        if torch.cuda.is_available():
            t0 = time.time()
            dummy_whisper = torch.zeros(batch_size, 50, 384, device=self.avatar.device, dtype=self.avatar.weight_dtype)
            self.avatar.generate_frames(dummy_whisper, 0, batch_size)
            # Remainder batch_size warmup (only when there's a remainder)
            # remain = fps % batch_size
            # if remain > 0:
            #     dummy_whisper_remain = torch.zeros(remain, 50, 384, device=self._avatar.device, dtype=self._avatar.weight_dtype)
            #     self._avatar.generate_frames(dummy_whisper_remain, 0, remain)
            torch.cuda.synchronize()
            t1 = time.time()
            logger.info(f"[THREAD_WARMUP] _frame_generator_worker thread id: {threading.get_ident()} "
                        f"self-warmup done, time: {(t1-t0)*1000:.1f} ms")
        batch_speech_id = []  # 语音ID
        batch_speech_end = []  # 语音结束标志
        batch_audio = []  # 音频数据
        batch_chunks = []  # 音频特征块
        while not self._stop_event.is_set():
            # 流控机制，队列满等待
            while self._frame_queue.qsize() > max_speaking_buffer and not self._stop_event.is_set():
                logger.info(f"[FRAME_GEN] speaking frame buffer full, waiting... "
                            f"frame_queue_size={self._frame_queue.qsize()}, "
                            f"max_speaking_buffer={max_speaking_buffer}")
                time.sleep(0.01)
                continue
            try:
                item = self._whisper_queue.get(timeout=1)
                batch_speech_id.append(item['speech_id'])
                batch_speech_end.append(item['speech_end'])
                batch_audio.append(item['audio_data'])
                batch_chunks.append(item['whisper_chunks'])
                if len(batch_chunks) == batch_size or item['speech_end']:
                    valid_num = len(batch_chunks)
                    if valid_num < batch_size:
                        logger.warning(f"[FRAME_GEN] batch_size < valid_num, "
                                       f"batch_size={batch_size}, valid_num={valid_num}")
                        pad_num = batch_size - valid_num
                        pad_shape = list(batch_chunks[0].shape)
                        if isinstance(batch_chunks[0], torch.Tensor):
                            pad_chunks = [torch.zeros(pad_shape, dtype=batch_chunks[0].dtype, device=batch_chunks[0].device) for _ in range(pad_num)]
                        else:
                            pad_chunks = [np.zeros(pad_shape, dtype=batch_chunks[0].dtype) for _ in range(pad_num)]
                        pad_audio = [np.zeros(orig_samples_per_frame, dtype=np.float32) for _ in range(pad_num)]
                        pad_speech_id = [batch_speech_id[-1]] * pad_num
                        pad_speech_end = [False] * pad_num
                        batch_chunks.extend(pad_chunks)
                        batch_audio.extend(pad_audio)
                        batch_speech_id.extend(pad_speech_id)
                        batch_speech_end.extend(pad_speech_end)
                    if isinstance(batch_chunks[0], torch.Tensor):
                        whisper_batch = torch.cat(batch_chunks, dim=0)
                    else:
                        whisper_batch = np.concatenate(batch_chunks, axis=0)
                    batch_start_time = time.time()
                    frame_ids = [self._frame_id_queue.get() for _ in range(batch_size)]
                    try:
                        recon_idx_list = self.avatar.generate_frames(whisper_batch, frame_ids[0], batch_size)
                    except Exception as e:
                        logger.opt(exception=True).error(f"[GEN_FRAME_ERROR] frame_id={frame_ids[0]}, "
                                                         f"speech_id={batch_speech_id[0]}, error: {e}")
                        recon_idx_list = [(np.zeros((256, 256, 3), dtype=np.uint8), frame_ids[0] + i) for i in range(batch_size)]
                    batch_end_time = time.time()
                    logger.info(f"[FRAME_GEN] Generated speaking frame batch: "
                                f"speech_id={batch_speech_id[0]}, "
                                f"batch_size={batch_size}, "
                                f"batch_time={(batch_end_time - batch_start_time)*1000:.1f}ms")
                    # 只处理有效帧
                    for i in range(valid_num):
                        recon, idx = recon_idx_list[i]
                        audio = batch_audio[i]
                        eos = batch_speech_end[i]
                        vae_item = {
                            'speech_id': batch_speech_id[i],
                            'speech_end': eos,
                            'recon': recon,
                            'idx': idx,
                            'frame_id': idx,
                            'avatar_status': AvatarStatus.SPEAKING,
                            'audio_segment': audio,
                            'timestamp': time.time()
                        }
                        self._vae_queue.put(vae_item)
                    batch_chunks = []
                    batch_audio = []
                    batch_speech_id = []
                    batch_speech_end = []
            except queue.Empty:
                time.sleep(0.01)
                continue

    def _frame_generator_unet_worker(self):
        """ UNet部分的帧生成工作线程，使用全局frame_id分配来确保说话帧的唯一和连续的帧编号。 """
        """
        设置帧率、每帧音频样本数等基础参数
        定义批处理大小和最大缓冲区限制
        """
        fps = self.config.fps
        orig_samples_per_frame = int(self.output_audio_sample_rate / fps)
        batch_size = self.config.batch_size
        max_speaking_buffer = batch_size * 5
        if torch.cuda.is_available():
            """ 模型预热，提升后续推理性能，避免首次运行延迟过高 """
            t0 = time.time()
            # 创建虚拟输入数据初始化GPU内存和计算图
            dummy_whisper = torch.zeros(batch_size, 50, 384, device=self.avatar.device, dtype=self.avatar.weight_dtype)
            self.avatar.generate_frames_unet(dummy_whisper, 0, batch_size)
            # Remainder batch_size warmup (only when there's a remainder)
            # remain = fps % batch_size
            # if remain > 0:
            #     dummy_whisper_remain = torch.zeros(remain, 50, 384, device=self._avatar.device, dtype=self._avatar.weight_dtype)
            #     self.avatar.generate_frames_unet(dummy_whisper_remain, 0, remain)
            torch.cuda.synchronize()
            t1 = time.time()
            logger.info(f"[THREAD_WARMUP] _frame_generator_unet_worker thread id: {threading.get_ident()} "
                        f"self-warmup done, time: {(t1 - t0) * 1000:.1f} ms")
        batch_speech_id = []  # 语音ID
        batch_speech_end = []  # 语音结束标志
        batch_audio = []  # 音频数据
        batch_chunks = []  # 音频特征块
        while not self._stop_event.is_set():
            # 流控机制，队列满等待
            while self._frame_queue.qsize() > max_speaking_buffer and not self._stop_event.is_set():
                logger.info(f"[FRAME_GEN] speaking frame buffer full, waiting... "
                            f"frame_queue_size={self._frame_queue.qsize()}, "
                            f"max_speaking_buffer={max_speaking_buffer}")
                time.sleep(0.01)
                continue
            try:
                item = self._whisper_queue.get(timeout=1)
                batch_speech_id.append(item['speech_id'])
                batch_speech_end.append(item['speech_end'])
                batch_audio.append(item['audio_data'])
                batch_chunks.append(item['whisper_chunks'])
                if len(batch_chunks) == batch_size or item['speech_end']:
                    valid_num = len(batch_chunks)
                    if valid_num < batch_size:
                        """
                        当批次不足时，使用零值填充至完整批次大小
                        保证模型输入的一致性
                        """
                        logger.warning(
                            f"[FRAME_GEN] batch_size < valid_num, "
                            f"batch_size={batch_size}, valid_num={valid_num}")
                        pad_num = batch_size - valid_num
                        pad_shape = list(batch_chunks[0].shape)
                        if isinstance(batch_chunks[0], torch.Tensor):
                            pad_chunks = [
                                torch.zeros(pad_shape, dtype=batch_chunks[0].dtype, device=batch_chunks[0].device) for _
                                in range(pad_num)]
                        else:
                            pad_chunks = [np.zeros(pad_shape, dtype=batch_chunks[0].dtype) for _ in range(pad_num)]
                        pad_audio = [np.zeros(orig_samples_per_frame, dtype=np.float32) for _ in range(pad_num)]
                        pad_speech_id = [batch_speech_id[-1]] * pad_num
                        pad_speech_end = [False] * pad_num
                        batch_speech_id.extend(pad_speech_id)
                        batch_speech_end.extend(pad_speech_end)
                        batch_audio.extend(pad_audio)
                        batch_chunks.extend(pad_chunks)
                    # 构造批处理张量
                    if isinstance(batch_chunks[0], torch.Tensor):
                        whisper_batch = torch.cat(batch_chunks, dim=0)
                    else:
                        whisper_batch = np.concatenate(batch_chunks, axis=0)
                    batch_start_time = time.time()
                    frame_ids = [self._frame_id_queue.get() for _ in range(batch_size)]
                    try:
                        pred_latents, idx_list = self.avatar.generate_frames_unet(
                            whisper_batch,
                            frame_ids[0],
                            batch_size
                        )
                    except Exception as e:
                        logger.opt(exception=True).error(f"[GEN_FRAME_ERROR] frame_id={frame_ids[0]}, "
                                                         f"speech_id={batch_speech_id[0]}, error: {e}")
                        pred_latents, idx_list = [
                            torch.zeros((batch_size, 4, 32, 32),
                                        dtype=self.avatar.unet.model.dtype,
                                        device=self.avatar.device),
                            [(frame_ids[0] + i) for i in range(batch_size)]
                        ]
                    batch_end_time = time.time()
                    logger.info(f"[FRAME_GEN] Generated speaking frame batch: "
                                f"speech_id={batch_speech_id[0]}, "
                                f"batch_size={batch_size}, "
                                f"batch_time={(batch_end_time - batch_start_time) * 1000:.1f}ms")
                    unet_item = {
                        'speech_id': batch_speech_id,
                        'speech_end': batch_speech_end,
                        'audio_data': batch_audio,
                        'valid_num': valid_num,
                        'pred_latents': pred_latents,  # torch.Tensor: [B, 4, 32, 32]
                        'idx_list': idx_list,
                        'avatar_status': AvatarStatus.SPEAKING,
                        'timestamp': time.time()
                    }
                    self._unet_queue.put(unet_item)
                    batch_chunks = []
                    batch_audio = []
                    batch_speech_id = []
                    batch_speech_end = []
            except queue.Empty:
                time.sleep(0.01)
                continue

    def _frame_generator_vae_worker(self):
        """ VAE部分的帧生成工作线程，将UNet生成的潜在向量解码为实际的图像帧。 """
        batch_size = self.config.batch_size
        max_speaking_buffer = batch_size * 5
        if torch.cuda.is_available():
            t0 = time.time()
            dummy_latents = torch.zeros(batch_size, 4, 32, 32, device=self.avatar.device,
                                        dtype=self.avatar.weight_dtype)
            idx_list = [0 + i for i in range(batch_size)]
            self.avatar.generate_frames_vae(dummy_latents, idx_list, batch_size)
            # Remainder batch_size warmup (only when there's a remainder)
            # remain = fps % batch_size
            # if remain > 0:
            #     dummy_latents_remain = torch.zeros(remain, 4, 32, 32, device=self._avatar.device, dtype=self._avatar.weight_dtype)
            #     idx_list = [0 + i for i in range(remain)]
            #     self._avatar.generate_frames_vae(dummy_latents_remain, idx_list, remain)
            torch.cuda.synchronize()
            t1 = time.time()
            logger.info(
                f"[THREAD_WARMUP] _frame_generator_vae_worker thread id: {threading.get_ident()} "
                f"self-warmup done, time: {(t1 - t0) * 1000:.1f} ms")
        while not self._stop_event.is_set():
            while self._frame_queue.qsize() > max_speaking_buffer and not self._stop_event.is_set():
                logger.info(f"[FRAME_GEN] speaking frame buffer full, waiting... "
                            f"frame_queue_size={self._frame_queue.qsize()}, "
                            f"max_speaking_buffer={max_speaking_buffer}")
                time.sleep(0.01)
                continue
            while self._unet_queue.qsize() <= 0 and not self._stop_event.is_set():
                time.sleep(0.01)
                continue
            # Batch vae inference for speaking frames
            try:
                item = self._unet_queue.get_nowait()
                batch_speech_id = item['speech_id']
                batch_speech_end = item['speech_end']
                batch_audio = item['audio_data']
                valid_num = item['valid_num']
                pred_latents = item['pred_latents']
                idx_list = item['idx_list']
                cur_batch = pred_latents.shape[0]
                batch_start_time = time.time()
                try:
                    recon_idx_list = self.avatar.generate_frames_vae(pred_latents, idx_list, cur_batch)
                except Exception as e:
                    logger.opt(exception=True).error(f"[GEN_FRAME_ERROR] frame_id={idx_list[0]}, "
                                                     f"speech_id={batch_speech_end[0]}, error: {e}")
                    recon_idx_list = [(np.zeros((256, 256, 3), dtype=np.uint8), idx_list[0] + i)
                                      for i in range(cur_batch)]
                batch_end_time = time.time()

                logger.info(f"[FRAME_GEN] Generated speaking frame batch: "
                            f"speech_id={batch_speech_end[0]}, "
                            f"batch_size={batch_size}, "
                            f"batch_time={(batch_end_time - batch_start_time) * 1000:.1f}ms")
                # just process valid frames
                for i in range(valid_num):
                    recon, idx = recon_idx_list[i]
                    audio = batch_audio[i]
                    eos = batch_speech_end[i]
                    vae_item = {
                        'speech_id': batch_speech_id[i],
                        'speech_end': eos,
                        'recon': recon,
                        'idx': idx,
                        'avatar_status': AvatarStatus.SPEAKING,
                        'audio_segment': audio,
                        'timestamp': time.time()
                    }
                    self._vae_queue.put(vae_item)
            except queue.Empty:
                time.sleep(0.01)
                continue

    def _frame_generator_avatar_worker(self):
        """ 负责执行generate_frames_avatar生成视频帧 """
        while not self._stop_event.is_set():
            try:
                item = self._vae_queue.get(timeout=0.1)
                recon = item['recon']
                idx = item['idx']
                frame = self.avatar.generate_frames_avatar(recon, idx)
                item['frame'] = frame
                self._output_queue.put(item)
            except queue.Empty:
                continue

    def _frame_collector_worker(self):
        """ 指定帧率(fps)收集和输出视频帧和音频帧 """
        fps = self.config.fps
        frame_interval = 1.0 / fps
        start_time = time.perf_counter()
        local_frame_id = 0
        last_active_speech_id = None
        last_speaking = False
        last_speech_end = False
        current_speech_id = None
        while not self._stop_event.is_set():
            # Control fps
            target_time = start_time + local_frame_id * frame_interval
            now = time.perf_counter()
            sleep_time = target_time - now
            if sleep_time > 0.002:
                time.sleep(sleep_time - 0.001)
            while time.perf_counter() < target_time:
                pass
            # Record the start time for profiling
            t_frame_start = time.perf_counter()
            # Allocate frame_id
            self._frame_id_queue.put(local_frame_id)
            try:
                output_item = self._output_queue.get_nowait()
                speech_id = output_item['speech_id']
                speech_end = output_item['speech_end']
                audio_segment = output_item['audio_segment']
                frame = output_item['frame']
                avatar_status = output_item['avatar_status']
                frame_timestamp = output_item.get('timestamp', None)
            except queue.Empty:
                speech_id = last_active_speech_id
                speech_end = False
                audio_segment = None
                frame = self.avatar.generate_idle_frame(local_frame_id)
                avatar_status = AvatarStatus.LISTENING
                frame_timestamp = time.time()
            # Notify video
            video_frame = VideoFrame(
                speech_id=speech_id,
                speech_end=speech_end,
                frame=av.VideoFrame.from_ndarray(frame, format="bgr24"),
                avatar_status=avatar_status,
                bg_frame_id=-1
            )
            self._notify_video(video_frame)
            # Logging logic
            is_idle = (avatar_status == AvatarStatus.LISTENING and speech_id is None)
            is_speaking = (avatar_status == AvatarStatus.SPEAKING)
            is_speech_end = bool(speech_end)
            if is_speaking:
                # First speaking frame
                if speech_id != current_speech_id:
                    logger.info(f"[SPEAKING_FRAME][START] frame_id={local_frame_id}, "
                                f"speech_id={speech_id}, status={avatar_status}, "
                                f"speech_end={speech_end}, video_timestamp={frame_timestamp}")
                    current_speech_id = speech_id
                # Last speaking frame
                if is_speech_end:
                    logger.info(f"[SPEAKING_FRAME][END] frame_id={local_frame_id}, "
                                f"speech_id={speech_id}, status={avatar_status}, "
                                f"speech_end={speech_end}, video_timestamp={frame_timestamp}")
                    current_speech_id = None
                # Middle speaking frame
                if not is_speech_end and (speech_id == current_speech_id):
                    logger.info(f"[SPEAKING_FRAME] frame_id={local_frame_id}, "
                                f"speech_id={speech_id}, status={avatar_status}, "
                                f"speech_end={speech_end}, video_timestamp={frame_timestamp}")
            elif is_idle and last_speaking:
                if last_speech_end:
                    logger.info(f"[IDLE_FRAME] Start after speaking: "
                                f"frame_id={local_frame_id}, status={avatar_status}")
                else:
                    logger.warning(f"[IDLE_FRAME] Inserted idle during speaking: frame_id={local_frame_id}")

            # Audio related
            audio_len = len(audio_segment) if audio_segment is not None else 0
            if audio_segment is not None and audio_len > 0:
                audio_np = np.asarray(audio_segment, dtype=np.float32)
                if audio_np.ndim == 1:
                    audio_np = audio_np[np.newaxis, :]
                audio_frame = av.AudioFrame.from_ndarray(audio_np, format="flt", layout="mono")
                audio_frame.sample_rate = self.output_audio_sample_rate
                audio_frame = AudioFrame(
                    speech_id=speech_id,
                    speech_end=speech_end,
                    frame=audio_frame,
                )
                if speech_id not in self._audio_cache:
                    self._audio_cache[speech_id] = []
                self._audio_cache[speech_id].append(audio_np[0] if audio_np.ndim == 2 else audio_np)
                audio_len_sum = sum([len(seg) for seg in self._audio_cache[speech_id]]) / self.output_audio_sample_rate
                logger.info(f"[AUDIO_FRAME] frame_id={local_frame_id}, "
                            f"speech_id={speech_id}, speech_end={speech_end}, "
                            f"audio_timestamp={frame_timestamp}, Cumulative audio duration={audio_len_sum:.3f}s")
                self._notify_audio(audio_frame)
            # Status switching etc.
            if speech_end:
                logger.info(f"Status change: SPEAKING -> LISTENING, speech_id={speech_id}")
                try:
                    if getattr(self.config, 'debug_save_handler_audio', False):
                        all_audio = np.concatenate(self._audio_cache[speech_id], axis=-1)
                        save_dir = "logs/audio_segments"
                        os.makedirs(save_dir, exist_ok=True)
                        wav_path = os.path.join(save_dir, f"{speech_id}_all.wav")
                        sf.write(wav_path, all_audio, self.output_audio_sample_rate, subtype='PCM_16')
                        logger.info(f"[AUDIO_FRAME] saved full wav: {wav_path}")
                except Exception as e:
                    logger.error(f"[AUDIO_FRAME] save full wav error: {e}")
                del self._audio_cache[speech_id]
                self._notify_status_change(AvatarStatus.LISTENING)
            t_frame_end = time.perf_counter()
            if t_frame_end - t_frame_start > frame_interval:
                logger.warning(f"[PROFILE] frame_id={local_frame_id} "
                               f"total={t_frame_end-t_frame_start:.4f}s (>{frame_interval:.4f}s)")
            local_frame_id += 1
            last_speaking = is_speaking
            last_speech_end = is_speech_end

    def _notify_audio(self, audio_frame: AudioFrame):
        if self.audio_output_queue is not None:
            frame = audio_frame.frame
            audio_data = frame.to_ndarray()
            # Ensure float32 and shape [1, N]
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            if audio_data.ndim == 1:
                audio_data = audio_data[np.newaxis, ...]
            elif audio_data.ndim == 2 and audio_data.shape[0] != 1:
                audio_data = audio_data[:1, ...]
            try:
                self.audio_output_queue.put_nowait(audio_data)
            except Exception as e:
                logger.opt(exception=True).error(f"Exception in _notify_audio: {e}")

    def _notify_video(self, video_frame: VideoFrame):
        if self.video_output_queue is not None:
            video_frame = video_frame.frame
            try:
                data = video_frame.to_ndarray(format="bgr24")
                self.video_output_queue.put_nowait(data)
            except Exception as e:
                logger.opt(exception=True).error(f"Exception in _notify_video: {e}")

    def _notify_status_change(self, status: AvatarStatus):
        if self.event_out_queue is not None and status == AvatarStatus.LISTENING:
            try:
                self.event_out_queue.put_nowait(Tts2FaceEvent.SPEAKING_TO_LISTENING)
            except Exception as e:
                logger.opt(exception=True).error(f"Exception in _notify_status_change: {e}")
