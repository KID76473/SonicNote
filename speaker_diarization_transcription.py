import os
import speech_recognition as sr
from pydub import AudioSegment
from dotenv import load_dotenv
from pyannote.audio import Pipeline
import torch
import tempfile

load_dotenv()
my_token = os.getenv('HUGGINGFACE_ACCESS_TOKEN')

def diarize_and_transcribe(audio_path, output_file=None):
    """
    Perform speaker diarization and transcription on an audio file.
    
    Args:
        audio_path: Path to the audio file
        output_file: Optional path to save the transcription results
    
    Returns:
        List of dictionaries containing speaker, start, end, and transcript
    """
    print(f"Processing audio file: {audio_path}")
    
    # Check if file exists
    if not os.path.exists(audio_path):
        print(f"Error: File {audio_path} not found")
        return None
    
    # Initialize pipeline
    print("Loading speaker diarization model...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=my_token
    )
    
    # Optional: send pipeline to GPU if available
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
        print("Using GPU for diarization")
    
    # Perform diarization
    print("Performing speaker diarization...")
    diarization = pipeline(audio_path)
    
    # Initialize speech recognizer
    recognizer = sr.Recognizer()
    
    # Load audio for segmentation
    print("Loading audio for segmentation...")
    if audio_path.lower().endswith('.m4a'):
        audio = AudioSegment.from_file(audio_path, format="m4a")
    else:
        audio = AudioSegment.from_file(audio_path)
    
    # Collect all segments first
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "speaker": speaker,
            "start": turn.start,
            "end": turn.end
        })
    
    # Merge consecutive segments from the same speaker
    merged_segments = []
    if segments:
        current_segment = segments[0].copy()
        
        for i in range(1, len(segments)):
            if segments[i]["speaker"] == current_segment["speaker"]:
                # Same speaker, extend the end time
                current_segment["end"] = segments[i]["end"]
            else:
                # Different speaker, save current and start new
                merged_segments.append(current_segment)
                current_segment = segments[i].copy()
        
        # Don't forget the last segment
        merged_segments.append(current_segment)
    
    print(f"\nMerged {len(segments)} segments into {len(merged_segments)} speaker turns")
    
    # Process merged segments
    results = []
    print("\nProcessing speaker segments...")
    
    for segment in merged_segments:
        start_ms = int(segment["start"] * 1000)
        end_ms = int(segment["end"] * 1000)
        
        # Extract audio segment
        audio_segment = audio[start_ms:end_ms]
        
        # Create temporary file for segment
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            temp_wav_path = temp_wav.name
        
        # Export segment as wav
        audio_segment.export(temp_wav_path, format="wav")
        
        # Transcribe segment
        try:
            with sr.AudioFile(temp_wav_path) as source:
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            text = "[Could not understand audio]"
        except sr.RequestError as e:
            text = f"[Error: {e}]"
        except Exception as e:
            text = f"[Transcription error: {e}]"
        
        # Clean up temporary file
        os.unlink(temp_wav_path)
        
        # Store result
        result = {
            "speaker": segment["speaker"],
            "start": segment["start"],
            "end": segment["end"],
            "transcript": text
        }
        results.append(result)
        
        # Print progress
        duration = segment["end"] - segment["start"]
        print(f"Speaker {segment['speaker']} ({segment['start']:.1f}s - {segment['end']:.1f}s, duration: {duration:.1f}s): {text}")
    
    # Save results if output file specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Speaker Diarization and Transcription Results\n")
            f.write("=" * 50 + "\n\n")
            for r in results:
                f.write(f"Speaker {r['speaker']} ({r['start']:.1f}s - {r['end']:.1f}s):\n")
                f.write(f"{r['transcript']}\n\n")
        print(f"\nResults saved to: {output_file}")
    
    return results

def format_transcript(results):
    """
    Format the transcription results for display.
    
    Args:
        results: List of dictionaries from diarize_and_transcribe
    
    Returns:
        Formatted string
    """
    if not results:
        return "No results to display"
    
    formatted = "Speaker Diarization and Transcription Results\n"
    formatted += "=" * 50 + "\n\n"
    
    for r in results:
        formatted += f"Speaker {r['speaker']} ({r['start']:.1f}s - {r['end']:.1f}s):\n"
        formatted += f"{r['transcript']}\n\n"
    
    return formatted

if __name__ == "__main__":
    # Example usage
    audio_file = "recordings/conversation.wav"
    output_file = "transcription_results.txt"
    
    results = diarize_and_transcribe(audio_file, output_file)
    
    if results:
        print("\n" + "=" * 50)
        print("Transcription complete!")
        print(f"Total segments: {len(results)}")
        unique_speakers = len(set(r['speaker'] for r in results))
        print(f"Number of speakers: {unique_speakers}")