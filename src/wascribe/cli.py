import os
import typer
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from .transcriber import Transcriber

app = typer.Typer()
load_dotenv()

@app.command()
def transcribe(
    input_file: Path = typer.Argument(..., help="Audio file to transcribe (.ogg or .mp3)"),
    output_file: Optional[Path] = typer.Option(None, help="Output file path (default: input_file_transcript.txt)"),
    diarize: bool = typer.Option(False, help="Enable speaker diarization"),
):
    """
    Transcribe WhatsApp audio messages using Groq and Pyannote.
    """
    if not input_file.exists():
        typer.echo(f"Error: File {input_file} does not exist")
        raise typer.Exit(1)
    
    if output_file is None:
        output_file = input_file.parent / f"{input_file.stem}_transcript.txt"
    
    try:
        transcriber = Transcriber()
        result = transcriber.transcribe(input_file, diarize=diarize)
        
        output_file.write_text(result)
        typer.echo(f"Transcription saved to {output_file}")
    
    except Exception as e:
        typer.echo(f"Error during transcription: {str(e)}", err=True)
        raise typer.Exit(1)

def main():
    app()

if __name__ == "__main__":
    main()