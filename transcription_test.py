import speech_recognition as sr
from pydub import AudioSegment
import os
import tempfile

def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    
    # Check if the file exists
    if not os.path.exists(audio_path):
        print(f"Error: File {audio_path} not found")
        return None
    
    # Convert .m4a to .wav if necessary
    if audio_path.lower().endswith('.m4a'):
        print("Converting .m4a to .wav...")
        try:
            # Load the .m4a file
            audio = AudioSegment.from_file(audio_path, format="m4a")
            
            # Create a temporary wav file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                temp_wav_path = temp_wav.name
                
            # Export as wav
            audio.export(temp_wav_path, format="wav")
            wav_path = temp_wav_path
            
        except Exception as e:
            print(f"Error converting audio file: {e}")
            return None
    else:
        wav_path = audio_path
    
    try:
        with sr.AudioFile(wav_path) as source:
            print("Listening...")
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source)
            audio_data = recognizer.record(source)

        print("Transcribing...")
        # Try Google Speech Recognition
        try:
            text = recognizer.recognize_google(audio_data)
        except AttributeError:
            # Fallback if method name is different
            text = recognizer.recognize_google_cloud(audio_data)
        
        print("Transcription complete:")
        print(text)
        
        # Clean up temporary file if we created one
        if audio_path.lower().endswith('.m4a') and os.path.exists(wav_path):
            os.unlink(wav_path)
            
        return text
        
    except sr.UnknownValueError:
        print("Could not understand audio")
        # Clean up temporary file if we created one
        if audio_path.lower().endswith('.m4a') and os.path.exists(wav_path):
            os.unlink(wav_path)
        return None
        
    except sr.RequestError as e:
        print(f"Request error: {e}")
        # Clean up temporary file if we created one
        if audio_path.lower().endswith('.m4a') and os.path.exists(wav_path):
            os.unlink(wav_path)
        return None
        
    except Exception as e:
        print(f"Error processing audio file: {e}")
        # Clean up temporary file if we created one
        if audio_path.lower().endswith('.m4a') and os.path.exists(wav_path):
            os.unlink(wav_path)
        return None

# Replace with your file path
audio_file_path = "recordings/test1.m4a"
transcribe_audio(audio_file_path)
