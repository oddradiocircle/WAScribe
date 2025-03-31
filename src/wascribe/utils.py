from pathlib import Path
from typing import List, Tuple

def format_transcript(segments: List[Tuple[str, str, float, float]]) -> str:
    """
    Format transcription segments with speaker information and timestamps
    
    Args:
        segments: List of tuples (speaker, text, start_time, end_time)
    
    Returns:
        Formatted transcript string
    """
    output = []
    for speaker, text, start, end in segments:
        timestamp = f"[{format_timestamp(start)} -> {format_timestamp(end)}]"
        output.append(f"{timestamp} {speaker}: {text}")
    return "\n".join(output)

def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"