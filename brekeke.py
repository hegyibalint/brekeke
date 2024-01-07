import io
import os
import datetime
from time import sleep

import numpy as np

import librosa
import pydub
import sounddevice as sd

import torchaudio

import torch
import openai

# import RPi.GPIO

torch.set_num_threads(1)

RESPEAKER_BUTTON = 17

# OpenAI API key
client = openai.OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

print("Loading VAD model...")
vad_model, (
    get_speech_timestamps,
    save_audio,
    read_audio,
    VADIterator,
    collect_chunks,
) = torch.hub.load(
    repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=True
)


def read_audio(wav: np.array, sampling_rate: int = 16000):
    # wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sampling_rate:
        transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sampling_rate)
        wav = transform(wav)
        sr = sampling_rate
    assert sr == sampling_rate
    return wav.squeeze(0)


def record() -> io.BytesIO:
    """
    Record audio from the microphone and return it as a WebM file.
    """
    SAMPLE_RATE = 16000
    FRAME_SIZE = 512
    CHANNELS = 1
    INT16_MAX = np.iinfo(np.int16).max

    vad_iterator = VADIterator(vad_model)
    samples = np.array([], dtype=np.int16)

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=FRAME_SIZE,
        channels=CHANNELS,
    ) as stream:
        print("Recording...")
        while True:
            frame, overflowed = stream.read(FRAME_SIZE)
            if overflowed:
                print("Overflowed!!!")
            frame = frame.flatten()
            # frame = (frame * INT16_MAX).astype(np.int16)

            asd = vad_iterator(frame, return_seconds=True)
            if asd:
                print(asd)

    # # Convert to WebM
    # samples = pydub.AudioSegment(
    #     samples,
    #     frame_rate=SAMPLE_RATE,
    #     sample_width=samples.,
    #     channels=CHANNELS,
    # )
    # # Save to a file
    # audio_file = samples.export(format="webm", codec="libopus").read()
    # buffer = io.BytesIO(audio_file)
    # buffer.name = "audio.webm"
    # return buffer


def handle_conversation():
    ASSISTANT_ID = "asst_zmxagj83hfxswc1MSImQv4A1"

    while True:
        print("Recording...")
        webm_file = record()
        # Upload the file to OpenAI
        print("Uploading...")
        response = client.audio.transcriptions.create(
            file=webm_file, model="whisper-1", response_format="text"
        )
        print(response)

        # If we hear "thank you", stop the conversation
        if response.lower().contains("thank you") or response.lower().contains(
            "thanks"
        ):
            break

        # Create a new thread for this conversation
        thread = client.beta.threads.create()
        # Add a message to this conversation's thread
        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=response,
        )

        # Invoke the assistant
        run = client.beta.threads.runs.create(
            assistant_id=ASSISTANT_ID, thread_id=thread.id
        )

        # Wait for the run to complete
        finished = False
        print("===========================")
        while not finished:
            run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            print(run)
            print("---------------------------")
            messages = client.beta.threads.messages.list(thread_id=thread.id)
            print(messages)
            if run.status == "completed":
                finished = True

        # Print the response
        print("Response:")
        messages = client.beta.threads.messages.list(thread_id=thread.id)
        last_message = messages.data[0].content[0].text.value

        # Synthesize the response
        response = client.audio.speech.create(
            input=last_message,
            model="tts-1",
            voice="onyx",
            response_format="aac",
        )
        # Play the response
        audio_file = io.BytesIO(response.content)
        # Play the response
        samples = pydub.AudioSegment.from_file(audio_file, format="aac")
        samples = samples.set_frame_rate(44100)
        # Play the response
        sd.play(samples.get_array_of_samples())
        sd.wait()


print("Setting up GPIO...")
# Setup GPIO14 as input with pull-down resistor
# RPi.GPIO.setmode(RPi.GPIO.BCM)
# RPi.GPIO.setup(RESPEAKER_BUTTON, RPi.GPIO.IN, pull_up_down=RPi.GPIO.PUD_DOWN)
# Add event handler
# RPi.GPIO.add_event_detect(RESPEAKER_BUTTON, RPi.GPIO.RISING, callback=button_pressed)

# Wait forever
# while True:
# Block until the button is pressed
# print("Waiting for button press...")
# RPi.GPIO.wait_for_edge(RESPEAKER_BUTTON, RPi.GPIO.RISING)
# input("Press enter to continue...")
# handle_conversation()

buffer = record()
# Save to a file
with open("audio.webm", "wb") as f:
    f.write(buffer.read())
