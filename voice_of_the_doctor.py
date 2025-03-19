import os
from gtts import gTTS
import subprocess
import platform
import elevenlabs
from elevenlabs.client import ElevenLabs
from pydub import AudioSegment
from pydub.playback import play

def text_to_speech_with_gtts_old(input_text, output_filepath):
    language="en"

    audioobj= gTTS(
        text=input_text,
        lang=language,
        slow=False
    )
    audioobj.save(output_filepath)


input_text="Hi this is Ai with Pritom!"
text_to_speech_with_gtts_old(input_text=input_text, output_filepath="gtts_testing.mp3")

ELEVENLABS_API_KEY=os.environ.get("ELEVENLABS_API_KEY")

def text_to_speech_with_elevenlabs_old(input_text, output_filepath):
    client=ElevenLabs(api_key=ELEVENLABS_API_KEY)
    audio=client.generate(
        text= input_text,
        voice= "Aria",
        output_format= "mp3_22050_32",
        model= "eleven_turbo_v2"
    )
    elevenlabs.save(audio, output_filepath)

def text_to_speech_with_gtts(input_text, output_filepath):
    language="en"

    audioobj= gTTS(
        text=input_text,
        lang=language,
        slow=False
    )
    audioobj.save(output_filepath)
    try:
        audio = AudioSegment.from_mp3(output_filepath)
        play(audio)
    except Exception as e:
        print(f"An error occurred while trying to play the audio: {e}")


input_text="Hi this is Ai with Pritom, autoplay testing!"
# text_to_speech_with_gtts(input_text=input_text, output_filepath="gtts_testing_autoplay.mp3")

def text_to_speech_with_elevenlabs(input_text, output_filepath):
    client=ElevenLabs(api_key=ELEVENLABS_API_KEY)
    audio=client.generate(
        text= input_text,
        voice= "Aria",
        output_format= "mp3_22050_32",
        model= "eleven_turbo_v2"
    )
    elevenlabs.save(audio, output_filepath)
    try:
        audio = AudioSegment.from_mp3(output_filepath)
        play(audio)
    except Exception as e:
        print(f"An error occurred while trying to play the audio: {e}")

# text_to_speech_with_elevenlabs(input_text, output_filepath="elevenlabs_testing_autoplay.mp3")