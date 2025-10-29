from typing import Any
from pydantic import BaseModel


class SpeechAudio(BaseModel):
    """
    only support mono audio for now
    """
    speech_id: Any = ""
    speech_end: bool = False
    sample_rate: int = 16000
    audio_data: bytes = bytes()

    def get_audio_duration(self):
        return len(self.audio_data) / self.sample_rate / 2
