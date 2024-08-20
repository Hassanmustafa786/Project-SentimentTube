# https://www.youtube.com/watch?v=SYHPQ0rXzWM

import yt_dlp
import gradio as gr
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load the fine-tuned BERT model and tokenizer
model = BertForSequenceClassification.from_pretrained('./results/checkpoint-150')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model.eval()

import yt_dlp

def download_youtube_video(video_url, output_path='Youtube Videos/Recent.mp4'):
    ydl_opts = {'format': 'bestaudio/best', 'outtmpl': output_path}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

    return output_path

from pydub import AudioSegment

def mp4_to_wav(mp4_path):
    # Convert MP4 to WAV
    sound = AudioSegment.from_file(mp4_path)
    wav_path = "Audios/pydub_output.wav"
    sound.export(wav_path, format="wav")
    return wav_path

import speech_recognition as sr

def transcribe_audio(audio_path):

    # Transcribe audio
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    text = recognizer.recognize_google(audio)
    
    return text

# Function to classify the transcription
def classify_transcription(transcription):
    inputs = tokenizer(transcription, truncation=True, padding=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
    return 'positive' if predicted_class == 1 else 'negative'

# Complete pipeline function
def classify_youtube_video(video_url):
    video_mp4_path = download_youtube_video(video_url)
    audio_wav_path = mp4_to_wav(video_mp4_path)
    transcription = transcribe_audio(audio_wav_path)
    sentiment = classify_transcription(transcription)
    return transcription, sentiment

# Create Gradio Interface
interface = gr.Interface(
    fn=classify_youtube_video,
    inputs=["text"],
    outputs=["text", "text"],
    title="YouTube Video Sentiment Classifier",
    description="Input a YouTube video URL to transcribe the audio and classify the sentiment as positive or negative."
)

# Launch the interface
interface.launch()