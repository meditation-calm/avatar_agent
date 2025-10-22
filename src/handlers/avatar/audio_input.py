from typing import Any
from pydantic import BaseModel


class SpeechAudio(BaseModel):
    """
    only support mono audio for now
    """
    def __init__(self, speech_id: Any = "", speech_end: bool = False, sample_rate: int = 16000, audio_data: bytes = bytes()):
        super().__init__()
        self.speech_id = speech_id
        self.speech_end = speech_end
        self.sample_rate = sample_rate
        self.audio_data = audio_data

    def get_audio_duration(self):
        return len(self.audio_data) / self.sample_rate / 2
