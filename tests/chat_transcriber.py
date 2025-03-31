import os
from pathlib import Path
import re
from typing import Dict, Set
from dotenv import load_dotenv
import base64
import requests
import subprocess
from tqdm import tqdm
from PIL import Image
import signal
import sys
import atexit
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

class TranscriptionError(Exception):
    """Base exception for transcription errors"""
    pass

class MediaProcessor:
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.mistral_api_key = os.getenv("MISTRAL_API_KEY")
        if not self.groq_api_key:
            raise TranscriptionError("GROQ_API_KEY not found in environment variables")
        if not self.mistral_api_key:
            raise TranscriptionError("MISTRAL_API_KEY not found in environment variables")

    def transcribe_audio(self, audio_path: Path) -> str:
        """Transcribe audio using Groq API"""
        try:
            if not audio_path.exists():
                raise TranscriptionError(f"Audio file not found: {audio_path}")

            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
            }
            
            url = "https://api.groq.com/openai/v1/audio/transcriptions"
            
            with open(audio_path, 'rb') as audio_file:
                files = {
                    'file': (audio_path.name, audio_file, 'audio/wav'),
                }
                data = {
                    'model': 'whisper-large-v3',
                    'response_format': 'text',
                    'language': 'es'
                }
                
                tqdm.write("Llamando a Groq API...")
                response = requests.post(url, headers=headers, files=files, data=data)
            
            if response.status_code != 200:
                raise TranscriptionError(f"Groq API error: {response.text}")
                
            return response.text.strip()
            
        except Exception as e:
            raise TranscriptionError(f"Error during transcription: {str(e)}")

    def convert_to_jpg(self, image_path: Path) -> Path:
        """Convert image to JPG format maintaining quality"""
        try:
            # Si ya es jpg, retornar la misma ruta
            if image_path.suffix.lower() in ['.jpg', '.jpeg']:
                return image_path
                
            output_path = image_path.parent / f"{image_path.stem}.jpg"
            
            # Abrir y convertir la imagen
            with Image.open(image_path) as img:
                # Convertir a RGB si es necesario (para PNGs con transparencia)
                if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                    bg = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode != 'RGBA':
                        img = img.convert('RGBA')
                    bg.paste(img, mask=img.split()[3])
                    img = bg
                else:
                    img = img.convert('RGB')
                
                # Guardar como JPG con máxima calidad
                img.save(output_path, 'JPEG', quality=95)
            
            return output_path
        except Exception as e:
            raise Exception(f"Error converting image to JPG: {str(e)}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((requests.exceptions.RequestException, Exception))
    )
    def _call_mistral_api(self, headers: dict, data: dict) -> dict:
        """Llamada a Mistral API con reintentos"""
        response = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code != 200:
            raise Exception(f"Mistral Vision API error: {response.text}")
        
        return response.json()

    def process_image(self, image_path: Path) -> str:
        """Process image using Mistral Vision API"""
        try:
            # Verificar tamaño del archivo
            file_size = os.path.getsize(image_path) / (1024 * 1024)  # Size in MB
            if file_size > 10:  # Mistral tiene límite de 10MB
                return f"[Imagen demasiado grande para procesar: {file_size:.1f}MB]"
            
            # Convertir a JPG si es necesario
            jpg_path = None
            try:
                jpg_path = self.convert_to_jpg(image_path)
                
                with open(jpg_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                
                headers = {
                    "Authorization": f"Bearer {self.mistral_api_key}",
                    "Content-Type": "application/json"
                }
                
                # Primera llamada: determinar tipo de imagen
                data = {
                    "model": "pixtral-12b-2409",
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "¿Esta imagen es un documento o texto que debe transcribirse, o es una imagen que debe ser descrita? Responde solo con 'DOCUMENTO' o 'IMAGEN'."},
                            {
                                "type": "image_url",
                                "image_url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        ]
                    }],
                    "max_tokens": 10  # Limitar tokens para respuesta corta
                }
                
                response_json = self._call_mistral_api(headers, data)
                tipo_imagen = response_json["choices"][0]["message"]["content"].strip().upper()
                
                # Segunda llamada: procesar según tipo
                if tipo_imagen == "DOCUMENTO":
                    prompt = "Transcribe todo el texto visible en esta imagen, manteniendo el formato."
                else:  # IMAGEN
                    prompt = "Describe detalladamente esta imagen en español, incluyendo todos los elementos relevantes y su contexto."
                
                data["messages"][0]["content"][0]["text"] = prompt
                data["max_tokens"] = 500  # Aumentar para respuesta completa
                
                response_json = self._call_mistral_api(headers, data)
                result = response_json["choices"][0]["message"]["content"]
                
                return f"[{'Texto' if tipo_imagen == 'DOCUMENTO' else 'Descripción'} en imagen: {result}]"
                
            finally:
                # Limpiar archivo temporal si se creó
                if jpg_path and jpg_path != image_path and jpg_path.exists():
                    try:
                        jpg_path.unlink()
                    except Exception as e:
                        tqdm.write(f"Warning: Error limpiando archivo temporal {jpg_path}: {e}")
                        
        except Exception as e:
            tqdm.write(f"Error procesando imagen {image_path.name}: {str(e)}")
            return f"[Error procesando imagen: {str(e)}]"

    def convert_to_wav(self, input_path: Path) -> Path:
        """Convert any audio/video to WAV format using ffmpeg"""
        wav_path = input_path.parent / f"{input_path.stem}.wav"
        try:
            # Usando subprocess para capturar y suprimir la salida
            command = ['ffmpeg', '-i', str(input_path), '-ar', '44100', '-ac', '2', str(wav_path), '-y']
            process = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            if process.returncode != 0:
                raise Exception(f"Error executing ffmpeg, return code: {process.returncode}")
            
            if not wav_path.exists():
                raise Exception("WAV file was not created")
                
            return wav_path
        except Exception as e:
            raise Exception(f"Error converting to WAV: {str(e)}")

class ChatProcessor:
    def __init__(self, chat_file: Path, data_dir: Path):
        self.chat_file = chat_file
        self.data_dir = data_dir
        self.output_file = chat_file.parent / f"{chat_file.stem}_processed{chat_file.suffix}"
        self.temp_file = chat_file.parent / f"{chat_file.stem}_temp{chat_file.suffix}"
        self.processor = MediaProcessor()
        self.processed_files = set()
        self.new_content_lines = []
        self.progress_bar = None
        
        # Configurar manejador de señales
        signal.signal(signal.SIGINT, self.handle_interrupt)
        atexit.register(self.cleanup)
        
    def handle_interrupt(self, signum, frame):
        """Manejar interrupción (Ctrl+C)"""
        print("\n\n⚠️ Interrupción detectada. Guardando progreso...")
        self.save_progress()
        print("✅ Progreso guardado. Puedes continuar más tarde.")
        sys.exit(0)
        
    def cleanup(self):
        """Limpieza al salir"""
        if self.progress_bar:
            self.progress_bar.close()
        if self.temp_file.exists():
            self.temp_file.unlink()
            
    def save_progress(self):
        """Guardar progreso actual"""
        self.output_file.write_text('\n'.join(self.new_content_lines), encoding='utf-8')
        
    def load_progress(self) -> Dict[str, str]:
        """Cargar progreso anterior"""
        transcriptions = {}
        if not self.output_file.exists():
            return transcriptions
        
        try:
            content = self.output_file.read_text(encoding='utf-8')
            for line in content.splitlines():
                if '[Transcripción' in line or '[Descripción' in line or '[Texto' in line:
                    prev_lines = content.splitlines()[:content.splitlines().index(line)]
                    for prev_line in reversed(prev_lines):
                        if any(ext in prev_line for ext in ['.opus', '.mp4', '.jpg', '.webp']):
                            file_match = re.search(r'((?:PTT|AUD|VID|IMG|STK)-\d{8}-WA\d{4}\.\w+)', prev_line)
                            if file_match:
                                filename = file_match.group(1)
                                transcriptions[filename] = line
                                break
        except Exception as e:
            print(f"Error reading existing transcriptions: {e}")
        
        return transcriptions
        
    def process_chat(self):
        """Procesar chat incrementalmente"""
        print("\n🔍 Analizando chat...")
        
        # Cargar progreso anterior
        existing_transcriptions = self.load_progress()
        self.processed_files = set(existing_transcriptions.keys())
        print(f"✓ Se encontraron {len(existing_transcriptions)} transcripciones existentes")
        
        chat_content = self.chat_file.read_text(encoding='utf-8')
        
        # Contar archivos por procesar
        total_files = len(re.findall(r'(?:PTT|AUD|VID|IMG|STK)-\d{8}-WA\d{4}\.(?:opus|mp4|jpg|jpeg|png|webp)', chat_content))
        remaining_files = total_files - len(existing_transcriptions)
        print(f"✓ Faltan {remaining_files} archivos por procesar de {total_files} totales")
        
        # Patrones de medios
        media_patterns = {
            'audio': r'((?:PTT|AUD)-\d{8}-WA\d{4}\.opus)',
            'video': r'(VID-\d{8}-WA\d{4}\.mp4)',
            'image': r'((?:IMG|STK)-\d{8}-WA\d{4}\.(?:jpg|jpeg|png|webp))'
        }
        
        print("\n🚀 Iniciando procesamiento...")
        self.progress_bar = tqdm(total=total_files, 
                               initial=len(existing_transcriptions),
                               desc="Archivos procesados", 
                               unit="archivo")
        
        # Procesar línea por línea
        for line in chat_content.splitlines():
            self.new_content_lines.append(line)
            timestamp = line.split(' - ')[0] if ' - ' in line else ''
            
            for media_type, pattern in media_patterns.items():
                matches = re.finditer(pattern, line)
                for match in matches:
                    filename = match.group(1)
                    
                    # Usar transcripción existente si está disponible
                    if filename in existing_transcriptions:
                        self.new_content_lines.append(existing_transcriptions[filename])
                        continue
                    
                    if filename in self.processed_files:
                        continue
                    
                    media_path = self.data_dir / filename
                    if media_path.exists():
                        tqdm.write(f"\n📝 Procesando {media_type}: {filename}")
                        try:
                            if media_type in ['audio', 'video']:
                                wav_path = self.processor.convert_to_wav(media_path)
                                try:
                                    transcription = self.processor.transcribe_audio(wav_path)
                                    new_line = f"{timestamp} - [Transcripción {media_type}: {transcription}]"
                                    self.new_content_lines.append(new_line)
                                    self.processed_files.add(filename)
                                finally:
                                    if wav_path.exists():
                                        wav_path.unlink()
                            else:  # image
                                description = self.processor.process_image(media_path)
                                self.new_content_lines.append(f"{timestamp} - {description}")
                                self.processed_files.add(filename)
                            
                            # Guardar progreso después de cada archivo
                            self.save_progress()
                            self.progress_bar.update(1)
                            
                        except Exception as e:
                            tqdm.write(f"❌ Error procesando {filename}: {e}")
                    else:
                        tqdm.write(f"⚠️ Archivo no encontrado: {filename}")
        
        self.progress_bar.close()
        
        print("\n✨ Resumen:")
        print(f"📁 Chat procesado guardado en: {self.output_file}")
        print(f"📊 Archivos procesados: {len(self.processed_files)}")
        print(f"🔄 Nuevos archivos: {len(self.processed_files) - len(existing_transcriptions)}")
        print(f"♻️ Reutilizados: {len(existing_transcriptions)}")

def process_chat_file(chat_file: Path, data_dir: Path):
    """Función principal de procesamiento"""
    load_dotenv()
    processor = ChatProcessor(chat_file, data_dir)
    processor.process_chat()

if __name__ == "__main__":
   chat_file = Path("tests/data/chat.txt")
   data_dir = Path("tests/data")
   process_chat_file(chat_file, data_dir)