from pathlib import Path
from dotenv import load_dotenv
from wascribe.transcriber import Transcriber

def test_transcription():
    try:
        # Cargar variables de entorno
        load_dotenv()
        
        transcriber = Transcriber()
        # Usando tu archivo específico
        audio_path = Path("tests/data/PTT-20241213-WA0021.opus")
        print(f"Transcribiendo {audio_path}...")
        result = transcriber.transcribe(audio_path)
        print(f"\nTranscripción completada:\n{result}")
    except Exception as e:
        print(f"Error durante la transcripción: {str(e)}")

if __name__ == "__main__":
    test_transcription()