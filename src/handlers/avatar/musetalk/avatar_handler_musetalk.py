import hashlib
import os
import pickle
import queue
import subprocess
from abc import ABC
from typing import Dict, Optional, cast

import numpy as np
import torch
from loguru import logger

from src.chat_engine.common.handler_base import HandlerBase, HandlerBaseInfo, HandlerDetail, HandlerDataInfo, \
    ChatDataConsumeMode
from src.chat_engine.contexts.handler_context import HandlerContext
from src.chat_engine.contexts.session_context import SessionContext
from src.chat_engine.data_models.chat_data.chat_data_model import ChatData
from src.chat_engine.data_models.chat_data_type import ChatDataType
from src.chat_engine.data_models.chat_engine_config_data import ChatEngineConfigModel
from src.chat_engine.data_models.runtime_data.data_bundle import DataBundleDefinition, DataBundleEntry, VariableSize
from src.engine_utils.general_slicer import SliceContext, slice_data
from src.handlers.avatar.audio_input import SpeechAudio
from src.handlers.avatar.musetalk.avatar_handler_algo import Avatar
from src.handlers.avatar.musetalk.avatar_handler_base import AvatarConfig, AvatarContext
from src.handlers.avatar.musetalk.avatar_processor_musetalk import AvatarProcessor


def check_ffmpeg():
    """ 检查安装ffmpeg """
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except Exception:
        logger.warning("Unable to find ffmpeg, please ensure ffmpeg is properly installed")


class HandlerAvatar(HandlerBase, ABC):
    def __init__(self) -> None:
        super().__init__()
        self.avatar: Optional[Avatar] = None
        self.processor: Optional[AvatarProcessor] = None
        self.event_in_queue = queue.Queue()
        self.event_out_queue = queue.Queue()
        self.audio_out_queue = queue.Queue()
        self.video_out_queue = queue.Queue()
        self.output_data_definitions: Dict[ChatDataType, DataBundleDefinition] = {}
        self.shared_states = None
        self._debug_cache = {}

    def get_handler_info(self) -> HandlerBaseInfo:
        return HandlerBaseInfo(
            config_model=AvatarConfig,
            load_priority=-999,
        )

    def load(self, engine_config: ChatEngineConfigModel, handler_config=None):
        if not isinstance(handler_config, AvatarConfig):
            handler_config = AvatarConfig()
        audio_output_definition = DataBundleDefinition()
        audio_output_definition.add_entry(DataBundleEntry.create_audio_entry(
            "avatar_muse_audio", 1, handler_config.output_audio_sample_rate
        ))
        audio_output_definition.lockdown()
        self.output_data_definitions[ChatDataType.AVATAR_AUDIO] = audio_output_definition
        video_output_definition = DataBundleDefinition()
        video_output_definition.add_entry(DataBundleEntry.create_framed_entry(
            "avatar_muse_video",
            [VariableSize(), VariableSize(), VariableSize(), 3],
            0,
            handler_config.fps
        ))
        video_output_definition.lockdown()
        self.output_data_definitions[ChatDataType.AVATAR_VIDEO] = video_output_definition
        # 检查ffmpeg
        check_ffmpeg()
        # auto generate avatar_id
        video_path = handler_config.avatar_video_path
        video_basename = os.path.splitext(os.path.basename(video_path))[0]
        video_hash = hashlib.md5(video_path.encode()).hexdigest()[:8]
        avatar_id = f"avatar_{video_basename}_{video_hash}"
        logger.info(f"Auto generated avatar_id: {avatar_id}")
        if handler_config.version == "v1":
            unet_model_path = os.path.join(engine_config.model_root, "musetalk/pytorch_model.bin")
            unet_config = os.path.join(engine_config.model_root, "musetalk/musetalk.json")
            result_dir = os.path.join(engine_config.model_root, "musetalk/result")
        else:
            unet_model_path = os.path.join(engine_config.model_root, "musetalkV15/unet.pth")
            unet_config = os.path.join(engine_config.model_root, "musetalkV15/musetalk.json")
            result_dir = os.path.join(engine_config.model_root, "musetalkV15/result")
        whisper_dir = os.path.join(engine_config.model_root, "whisper")
        # load avatar
        self.avatar = Avatar(
            avatar_id=avatar_id,
            video_path=video_path,
            bbox_shift=handler_config.bbox_shift,
            batch_size=handler_config.batch_size,
            preparation=handler_config.preparation,
            parsing_mode=handler_config.parsing_mode,
            extra_margin=handler_config.extra_margin,
            fps=handler_config.fps,
            gpu_id=handler_config.gpu_id,
            version=handler_config.version,
            audio_padding_length_left=handler_config.audio_padding_length_left,
            audio_padding_length_right=handler_config.audio_padding_length_right,
            left_cheek_width=handler_config.left_cheek_width,
            right_cheek_width=handler_config.right_cheek_width,
            vae_type=handler_config.vae_type,
            unet_model_path=unet_model_path,
            unet_config=unet_config,
            result_dir=result_dir,
            whisper_dir=whisper_dir
        )
        self.processor = AvatarProcessor(
            self.avatar,
            handler_config,
            self.event_out_queue,
            self.audio_out_queue,
            self.video_out_queue
        )
        logger.info(f"Loaded MuseTalk model from {unet_model_path}")

    def create_context(self, session_context: SessionContext, handler_config=None) -> HandlerContext:
        if not isinstance(handler_config, AvatarConfig):
            handler_config = AvatarConfig()
        self.shared_states = session_context.shared_states
        context = AvatarContext(
            session_context.session_info.session_id,
            self.event_in_queue,
            self.event_out_queue,
            self.audio_out_queue,
            self.video_out_queue,
            self.shared_states
        )
        context.config = handler_config

        frame_audio_len_float = handler_config.output_audio_sample_rate / handler_config.fps
        if not frame_audio_len_float.is_integer():
            logger.error(
                f"output_audio_sample_rate / fps = {handler_config.output_audio_sample_rate} / "
                f"{handler_config.fps} = {frame_audio_len_float}, is not an integer, "
                "there may be cumulative error in audio-video alignment!"
            )
        context.slice_context = SliceContext.create_numpy_slice_context(
            slice_size=handler_config.output_audio_sample_rate,
            slice_axis=0,
        )
        context.output_data_definitions = self.output_data_definitions
        return context

    def start_context(self, session_context: SessionContext, handler_context: HandlerContext):
        self.processor.start()

    def get_handler_detail(self, session_context: SessionContext,
                           context: HandlerContext) -> HandlerDetail:
        inputs = {
            ChatDataType.AVATAR_AUDIO: HandlerDataInfo(
                type=ChatDataType.AVATAR_AUDIO,
                input_consume_mode=ChatDataConsumeMode.ONCE,
            )
        }
        outputs = {
            ChatDataType.AVATAR_AUDIO: HandlerDataInfo(
                type=ChatDataType.AVATAR_AUDIO,
                definition=self.output_data_definitions[ChatDataType.AVATAR_AUDIO],
            ),
            ChatDataType.AVATAR_VIDEO: HandlerDataInfo(
                type=ChatDataType.AVATAR_VIDEO,
                definition=self.output_data_definitions[ChatDataType.AVATAR_VIDEO],
            ),
        }
        return HandlerDetail(inputs=inputs, outputs=outputs)

    def handle(self, context: HandlerContext, inputs: ChatData,
               output_definitions: Dict[ChatDataType, HandlerDataInfo]):
        context = cast(AvatarContext, context)
        if inputs.type != ChatDataType.AVATAR_AUDIO:
            return
        speech_id = inputs.data.get_meta("speech_id", context.session_id)
        avatar_speech_end = inputs.data.get_meta("avatar_speech_end", False)
        audio_entry = inputs.data.get_main_definition_entry()
        audio_array = inputs.data.get_main_data()
        logger.info(f"Avatar Handle Input: speech_id={speech_id}, avatar_speech_end={avatar_speech_end}, "
                    f"audio_array.shape={getattr(audio_array, 'shape', None)}")
        input_sample_rate = audio_entry.sample_rate
        if input_sample_rate != context.config.output_audio_sample_rate:
            logger.error(f"Input sample rate {input_sample_rate} != output sample rate "
                         f"{context.config.output_audio_sample_rate}")
            return
        if audio_array is not None and audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)
        if audio_array is None:
            audio_array = np.zeros([input_sample_rate], dtype=np.float32)
            logger.error(f"Audio data is None, fill with 1s silence, speech_id: {speech_id}")
        for audio_segment in slice_data(context.slice_context, audio_array.squeeze()):
            audio_input = SpeechAudio(
                speech_id=speech_id,
                speech_end=False,
                sample_rate=input_sample_rate,
                audio_data=audio_segment.tobytes(),
            )
            if self.processor:
                self.processor.add_audio(audio_input)
        if avatar_speech_end:
            remainder_audio = context.slice_context.flush()
            if remainder_audio is None:
                logger.warning(f"Last segment is empty: speech_id={speech_id}, speech_end={avatar_speech_end}")
                fps = context.config.fps if hasattr(context.config, "fps") else 25
                frame_len = input_sample_rate // fps
                # 2 frames audio for silence
                zero_frames = np.zeros([2 * frame_len], dtype=np.float32)
                audio_data = zero_frames.tobytes()
            else:
                audio_data = remainder_audio.tobytes()
            audio_input = SpeechAudio(
                speech_id=speech_id,
                speech_end=True,
                sample_rate=input_sample_rate,
                audio_data=audio_data
            )
            if self.processor:
                self.processor.add_audio(audio_input)

    def destroy_context(self, context: HandlerContext):
        if self.processor:
            self.processor.stop()
        if isinstance(context, AvatarContext):
            context.clear()
