"""
Robust Transcription Backend using Faster-Whisper
"""
import os
from faster_whisper import WhisperModel

def transcribe(video_path):
    print("🎙️ Starting Whisper transcription...")
    try:
        # Use INT8 compute type for compatibility across all CPU/GPU cloud runners
        model = WhisperModel("base", device="cpu", compute_type="int8")
        segments, info = model.transcribe(video_path, beam_size=5, word_timestamps=True)
        
        transcript_segments = []
        for segment in segments:
            words = [{"start": w.start, "end": w.end, "word": w.word} for w in getattr(segment, 'words', [])]
            transcript_segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "words": words
            })
            
        print(f"✅ Transcription complete. Detected Language: {info.language}")
        return {
            "segments": transcript_segments, 
            "duration": info.duration,
            "language": info.language
        }
    except Exception as e:
        print(f"❌ Transcription failed or audio track missing: {e}")
        return {"segments": [], "duration": 60.0}
