from pyannote.audio import Pipeline
from pydub import AudioSegment
import soundfile as sf
import torch
import io
import openai
import os
from dotenv import load_dotenv
import tempfile
 
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
 
# Initialize the pyannote audio pipeline for speaker diarization
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                    use_auth_token=True)
 
# Function to convert MP3 to WAV
def convert_mp3_to_wav(mp3_path, wav_path):
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format="wav")
 
# Convert the MP3 file to WAV format
mp3_file_path = "recording1.mp3"  # Replace with your file path
wav_file_path = "your_audio_file.wav"
convert_mp3_to_wav(mp3_file_path, wav_file_path)
print("MP3 converted to WAV")
 
# Load the WAV file and process for diarization
with sf.SoundFile(wav_file_path) as f:
    frames_to_read = int(120 * f.samplerate)
    audio_data = f.read(frames=frames_to_read, dtype="float32")
    sample_rate = f.samplerate
    if f.channels == 1:
        audio_data = audio_data.reshape(1, -1)
    else:
        audio_data = audio_data.T
 
# Convert to PyTorch tensor
audio_tensor = torch.tensor(audio_data, dtype=torch.float32)
 
# Diarization
diarization = pipeline({"waveform": audio_tensor, "sample_rate": sample_rate}, num_speakers=2)
print("Diarization complete")
 
def transcribe_with_openai(audio_segment):
    if audio_segment.shape[1] / sample_rate >= 0.1:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio_file:
            sf.write(temp_audio_file.name, audio_segment.T, sample_rate, format='wav', subtype='PCM_16')
           
            with open(temp_audio_file.name, 'rb') as file_to_transcribe:
                try:
                    transcript_response = openai_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=file_to_transcribe
                    )
                    # Access the transcription text directly
                    transcript_text = transcript_response.text
                except Exception as e:
                    transcript_text = f"[Error in transcription: {e}]"
 
        os.remove(temp_audio_file.name)
        return transcript_text
    else:
        return "[Audio too short for transcription]"
 
# Transcribe each segment
transcriptions = []
for segment, _, speaker in diarization.itertracks(yield_label=True):
    start = int(segment.start * sample_rate)
    end = int(segment.end * sample_rate)
    speaker_audio = audio_data[:, start:end]
    transcript = transcribe_with_openai(speaker_audio)
    transcriptions.append((speaker, transcript))
 
# Print the annotated conversation
for speaker, transcript in transcriptions:
    print(f"Speaker {speaker}: {transcript}")