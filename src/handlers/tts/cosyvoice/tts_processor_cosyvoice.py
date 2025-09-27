import wave

from torch.multiprocessing import Queue
import torch.multiprocessing as mp

import os
import sys

from loguru import logger

from src.engine_utils.directory_info import DirectoryInfo

""" torch spawn独立进程运行 """
spawn_context = mp.get_context('spawn')


class TTSCosyVoiceProcessor(spawn_context.Process):
    def __init__(self, handler_root: str, config: any, input_queue: Queue, output_queue: Queue):
        super().__init__()
        self.handler_root = handler_root
        self.model = None
        self.model_name = config.model_name
        self.ref_audio_text = config.ref_audio_text
        self.ref_audio_path = config.ref_audio_path
        self.ref_audio_buffer = None
        self.spk_id = config.spk_id
        self.speed = config.speed
        self.sample_rate = config.sample_rate

        self.input_queue = input_queue
        self.output_queue = output_queue
        self.dump_audio = False
        self.audio_dump_file = None

    def run(self):
        logger.remove()
        logger.add(sys.stdout, level='INFO')
        if self.dump_audio:
            dump_file_path = os.path.join(DirectoryInfo.get_project_dir(), "cache", "dump_avatar_audio.pcm")
            self.audio_dump_file = open(dump_file_path, "wb")
        logger.info(f"Start CosyVoice TTS processor with model {self.model_name}")
        # use local model

        """ 加载模型预执行语音合成 """
        if self.model_name is not None:
            sys.path.append(os.path.join(self.handler_root, "CosyVoice"))
            sys.path.append(os.path.join(self.handler_root, 'CosyVoice', 'third_party', 'Matcha-TTS'))
            from src.handlers.tts.cosyvoice.CosyVoice.cosyvoice.utils.file_utils import load_wav
            from src.handlers.tts.cosyvoice.CosyVoice.cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
            try:
                self.model = CosyVoice(model_dir=self.model_name)
            except Exception:
                try:
                    self.model = CosyVoice2(model_dir=self.model_name)
                except Exception:
                    raise TypeError('Cosyvoice no valid model_type!')
            self.model.sample_rate = self.sample_rate
            if self.ref_audio_path:
                self.ref_audio_buffer = load_wav(self.ref_audio_path, self.sample_rate)
                self.ref_audio_text = self.ref_audio_text
            init_text = '欢迎来到中国2025'
            if self.ref_audio_buffer is not None:
                response = self.model.inference_zero_shot(
                    init_text, self.ref_audio_text, self.ref_audio_buffer, True, False, self.speed)
            elif self.spk_id:
                response = self.model.inference_sft(init_text, self.spk_id, False, self.speed)
            else:
                logger.error('Cosyvoice need a ref_audio or spk_id')
                return
            if response is not None:
                for tts_speech in response:
                    self.output_queue.put({
                        'key': '',
                        'tts_speech': tts_speech,
                        'session_id': ''
                    })
                    logger.debug('tts test')
        else:
            raise TypeError('model_name not support yet')

        """ 语音识别任务执行 """
        while True:
            try:
                logger.debug('wait for tts task in')
                input = self.input_queue.get(timeout=5)
                logger.debug(f'get tts task in {input}')
            except Exception:
                continue
            input_text = input['text']
            key = input['key']
            session_id = input['session_id']
            if len(input_text) < 1:
                logger.info('Cosyvoice ignore empty input_text')
                continue
            response = None
            if self.model:
                if self.ref_audio_buffer is not None:
                    response = self.model.inference_zero_shot(
                        input_text, self.ref_audio_text, self.ref_audio_buffer, True, True, self.speed)
                elif self.spk_id:
                    response = self.model.inference_sft(input_text, self.spk_id, True, self.speed)
                else:
                    logger.error('Cosyvoice need a ref_audio or spk_id')
                    return

            for tts_speech in response:
                tts_audio = tts_speech['tts_speech'].numpy()
                logger.debug(f'tts sample rate {self.model.sample_rate}')
                tts_audio = tts_audio
                # librosa.resample(tts_audio, orig_sr=self.model.sample_rate, target_sr=24000)
                # tts_audio = torchaudio.transforms.Resample(orig_freq=22050, new_freq=24000)(tts_audio)
                if self.dump_audio:
                    dump_audio = tts_audio
                    self.audio_dump_file.write(dump_audio.tobytes())
                output = {
                    'key': key,
                    'tts_speech': tts_audio,
                    'session_id': session_id
                }
                self.output_queue.put(output)
            output = {
                'key': key,
                'tts_speech': None,
                'session_id': session_id
            }
            self.output_queue.put(output)
