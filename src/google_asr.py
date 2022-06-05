from google.cloud import speech
import io
import os

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './credential.json'

def transcribe_file(speech_file):
    """Transcribe the given audio file."""
    client = speech.SpeechClient()

    with io.open(speech_file, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=48000,
        language_code="sw-TZ",  # sw-KE
    )

    response = client.recognize(config=config, audio=audio)

    # Each result is for a consecutive portion of the audio. Iterate through
    # them to get the transcripts for the entire audio file.
    for result in response.results:
        # The first alternative is the most likely one for this portion.
        return result.alternatives[0].transcript


if __name__ == '__main__':
    transcribe_file('data/swa-eng/train/wav/1/0b710f1ba2e5dff867bd617678258c39__1573474526.6732.wav')