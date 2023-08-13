''' This is the main application file for the CODER application. It contains the main functions and the support
    functions for the application. '''

#%% ----------------------------- IMPORTS  -----------------------------------
# import libraries
import streamlit as st
import credentials
import os
from audio_recorder_streamlit import audio_recorder
import openai
from gtts import gTTS
import io
from IPython.display import Audio
from pydub import AudioSegment
from audio_recorder_streamlit import audio_recorder
from IPython.display import Audio
from langchain.callbacks import get_openai_callback
# Import the Speech-to-Text client library
from google.cloud import speech

from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech

from elevenlabs import clone, generate, play, set_api_key

set_api_key(credentials.xi_api_key)


#%% ----------------------------- LANGCHAIN FUNCTIONS -----------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
OPENAI_API_KEY = credentials.OPENAI_API_KEY


# %% ----------------------------- AUDIO RECORDING STREAMLIT -----------------------------------
def rec_streamlit():
    """Record audio and return the audio bytes"""
    audio_bytes = audio_recorder(energy_threshold=(-1.0, 1.0), pause_threshold=6.0, text="",
                                 recording_color="#FF0000", neutral_color="#49DE49", icon_name="microphone", icon_size="3x")

    return audio_bytes


def rec_streamlit_long():
    """Record audio and return the audio bytes"""
    audio_bytes = audio_recorder(energy_threshold=(-1.0, 1.0), pause_threshold=50.0, text="",
                                 recording_color="#FF0000", neutral_color="#49DE49", icon_name="microphone", icon_size="3x")

    return audio_bytes

#%% ----------------------------- OPEN AI FUNCTIONS  -----------------------------------
def get_transcript_whisper(file_path):
    '''Get the transcript of the audio file'''
    openai.api_key = OPENAI_API_KEY
    with open(file_path, "rb") as file:
        transcription = openai.Audio.transcribe("whisper-1", file, response_format="json")
    transcribed_text = transcription["text"]

    return transcribed_text


#%% ----------------------------- SPEAK FUNCTIONS -----------------------------------
def speak_answer(answer, tts_enabled):
    if not tts_enabled:
        return

    tts = gTTS(text=answer, lang="en")
    with io.BytesIO() as f:
        tts.write_to_fp(f)
        f.seek(0)
        audio = Audio(f.read(), autoplay=True)
        st.write(audio)

#%% ----------------------------- MAIN APPLICATION FUNCTIONS -----------------------------------
def home():
    # ------------------ SETTINGS ------------------
    st.set_page_config(page_title="Home", layout="wide")
    st.markdown("""<style>.reportview-container .main .block-container {max-width: 95%;}</style>""",
                unsafe_allow_html=True)

    # ------------------ HOME PAGE ------------------
    st.title("MAIN FILE MULTIAGENT 🎙️📖🥷")
    st.write("""Use the power of LLMs with LangChain and OpenAI to scan through your documents. Find information 
    and insight's with lightning speed. 🚀 Create new content with the support of state of the art language models and 
    and voice command your way through your documents. 🎙️""")
    st.write("Choose a file in the sidebar and begin exploring.")

    # ------------------ SIDE BAR SETTINGS ------------------
    st.sidebar.subheader("Settings:")
    tts_enabled = st.sidebar.checkbox("Enable Text-to-Speech", value=False)
    ner_enabled = st.sidebar.checkbox("Enable NER in Response", value=False)

    # ------------------ FILE SELECTION ------------------
    # List all the files in the pages folder
    pages = os.listdir(os.path.join(script_dir, "pages"))

    # Write all file names into the main window to display for user withou selection
    st.write("Choose a file in the sidebar and begin exploring.")
    st.write("Available files:")

    # Sort pages alphabetically
    pages.sort()

    for page in pages:
        st.write(f"- {page}")


# Run home function
if __name__ == "__main__":
    home()
