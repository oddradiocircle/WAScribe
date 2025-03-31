import os
import json
from pathlib import Path
from typing import Optional
import base64

import requests
from pydub import AudioSegment

class TranscriptionError(Exception):
    """Base exception for transcription errors"""
    pass

class Transcriber:
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.pyannote_auth_token = os.getenv("PYANNOTE_AUTH_TOKEN")
        
        if not self.groq_api_key:
            raise TranscriptionError("GROQ_API_KEY not found in environment variables")
        
        if not self.pyannote_auth_token:
            raise TranscriptionError("PYANNOTE_AUTH_TOKEN not found in environment variables")

    def _convert_to_wav(self, audio_path: Path) -> Path:
        """Convert audio file to WAV format if needed"""
        if audio_path.suffix.lower() == '.wav':
            return audio_path
            
        wav_path = audio_path.parent / f"{audio_path.stem}.wav"
        print(f"Converting {audio_path} to {wav_path}")
        audio = AudioSegment.from_file(str(audio_path))
        audio.export(str(wav_path), format='wav')
        return wav_path

    def _transcribe_with_groq(self, audio_path: Path) -> str:
        """Transcribe audio using Groq API"""
        if not audio_path.exists():
            raise TranscriptionError(f"Audio file not found: {audio_path}")
        
        try:
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}"
            }
            
            url = "https://api.groq.com/openai/v1/audio/transcriptions"
            
            with open(audio_path, 'rb') as audio_file:
                files = {
                    'file': (audio_path.name, audio_file, 'audio/wav')
                }
                
                data = {
                    'model': 'whisper-large-v3-turbo',  # Cambiado a whisper-large-v3-turbo
                    'response_format': 'text',
                    'language': 'es'  # Especificando español para mejor precisión
                }
                
                print("Calling Groq API...")
                response = requests.post(
                    url,
                    headers=headers,
                    files=files,
                    data=data
                )
            
            if response.status_code != 200:
                raise TranscriptionError(f"Groq API error: {response.text}")
                
            return response.text.strip()
            
        except requests.exceptions.RequestException as e:
            raise TranscriptionError(f"Error calling Groq API: {str(e)}")
        except Exception as e:
            raise TranscriptionError(f"Error during transcription: {str(e)}")

    def transcribe(self, audio_path: Path, diarize: bool = False) -> str:
        """
        Transcribe an audio file, optionally with speaker diarization
        """
        print(f"Starting transcription of {audio_path}")
        wav_path = self._convert_to_wav(audio_path)
        
        try:
            if diarize:
                # TODO: Implement diarization
                pass
            transcript = self._transcribe_with_groq(wav_path)
            return transcript
            
        finally:
            # Clean up temporary WAV file if it was created
            if wav_path != audio_path and wav_path.exists():
                wav_path.unlink()