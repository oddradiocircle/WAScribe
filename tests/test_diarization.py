import assemblyai as aai
from pathlib import Path
import os
from dotenv import load_dotenv
from pydub import AudioSegment
from tqdm import tqdm
import json
import subprocess

class AssemblyAITranscriber:
    def __init__(self, api_key):
        aai.settings.api_key = api_key
        self.transcriber = aai.Transcriber()
        
    def convert_to_mp3(self, audio_path: Path) -> Path:
        """Convert audio file to MP3 format using ffmpeg"""
        try:
            mp3_path = audio_path.parent / f"{audio_path.stem}.mp3"
            command = [
                'ffmpeg', '-i', str(audio_path),
                '-codec:a', 'libmp3lame',
                '-qscale:a', '2',  # Alta calidad (0-9, donde 0 es la mejor)
                str(mp3_path),
                '-y'
            ]
            
            subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )
            
            return mp3_path
        except Exception as e:
            raise Exception(f"Error converting to MP3: {str(e)}")
            
    def transcribe_with_diarization(self, audio_path: Path) -> dict:
        """Transcribe audio with automatic speaker diarization"""
        try:
            # Convertir a MP3 si no lo es
            if audio_path.suffix.lower() != '.mp3':
                print(f"\nConvirtiendo {audio_path.name} a MP3...")
                mp3_path = self.convert_to_mp3(audio_path)
            else:
                mp3_path = audio_path
                
            print(f"Transcribiendo {mp3_path.name}...")
            
            # Configurar transcripción con diarización y español colombiano
            config = aai.TranscriptionConfig(
                speaker_labels=True,
                language_code="es-CO"
            )
            
            # Realizar transcripción
            transcript = self.transcriber.transcribe(
                str(mp3_path),
                config=config
            )
            
            # Obtener número de hablantes únicos
            speakers = set(utterance.speaker for utterance in transcript.utterances)
            
            # Formatear resultado
            result = {
                "filename": audio_path.name,
                "total_speakers": len(speakers),
                "speakers_detected": list(speakers),
                "utterances": []
            }
            
            for utterance in transcript.utterances:
                result["utterances"].append({
                    "speaker": utterance.speaker,
                    "text": utterance.text,
                    "start": utterance.start,
                    "end": utterance.end,
                    "confidence": utterance.confidence
                })
            
            # Limpiar archivo temporal si se creó
            if mp3_path != audio_path and mp3_path.exists():
                mp3_path.unlink()
                
            return result
            
        except Exception as e:
            print(f"Error transcribiendo {audio_path.name}: {str(e)}")
            return {
                "filename": audio_path.name,
                "error": str(e)
            }

def process_test_files():
    """Procesar archivos de audio en la carpeta de test"""
    # Configurar API key
    ASSEMBLY_AI_KEY = "bd5c7b753e9241edb974baf1c98e5eb5"
    transcriber = AssemblyAITranscriber(ASSEMBLY_AI_KEY)
    
    # Definir carpeta de datos
    data_dir = Path("tests/data")
    
    # Encontrar archivos de audio
    audio_files = [
        f for f in data_dir.glob("**/*")
        if f.suffix.lower() in ['.opus', '.ogg', '.mp3', '.m4a']
    ]
    
    print(f"\nEncontrados {len(audio_files)} archivos de audio:")
    for f in audio_files:
        print(f"- {f.name}")
    
    # Procesar cada archivo
    results = []
    for audio_file in tqdm(audio_files, desc="Procesando archivos", ncols=100):
        result = transcriber.transcribe_with_diarization(audio_file)
        results.append(result)
        
        # Guardar resultado individual
        output_file = data_dir / f"{audio_file.stem}_diarization.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\nTranscripción guardada en: {output_file}")
            print(f"Hablantes detectados: {result.get('total_speakers', 'error')}")
            if 'utterances' in result:
                print("\nPrimeras líneas de la transcripción:")
                for i, utterance in enumerate(result['utterances'][:3]):
                    print(f"Hablante {utterance['speaker']}: {utterance['text']}")
                if len(result['utterances']) > 3:
                    print("...")
    
    # Guardar resultado completo
    complete_output = data_dir / "all_transcriptions_with_diarization.json"
    with open(complete_output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\nResumen final:")
    print(f"✓ Archivos procesados: {len(results)}")
    print(f"✓ Resultados guardados en: {complete_output}")
    print("\nResumen por archivo:")
    for result in results:
        print(f"\n{result['filename']}:")
        print(f"  - Hablantes detectados: {result.get('total_speakers', 'error')}")
        if 'error' in result:
            print(f"  - Error: {result['error']}")

if __name__ == "__main__":
    process_test_files()