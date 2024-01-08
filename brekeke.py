import io
import os
from time import sleep, time

import numpy as np
import pydub
import sounddevice as sd
import torch
import openai
from openai.types.beta import Thread

from apa102 import APA102

# Conditional import for input handling
#  - If RPi.GPIO is available, use the hardware button
#  - Otherwise, use the keyboard
try:
    import RPi.GPIO

    def setup_input():
        RPi.GPIO.setmode(RPi.GPIO.BCM)
        RPi.GPIO.setup(RESPEAKER_BUTTON, RPi.GPIO.IN, pull_up_down=RPi.GPIO.PUD_UP)

    def wait_for_input():
        print("Waiting for button press...")
        RPi.GPIO.wait_for_edge(RESPEAKER_BUTTON, RPi.GPIO.FALLING)

except ImportError:

    def setup_input():
        print("RPi.GPIO not found, using keyboard input instead.")
        pass

    def wait_for_input():
        return input("Press enter to continue...")


# =============================================================================
# Constants
# =============================================================================

RESPEAKER_BUTTON = 17


# =============================================================================
# Global variables and initialization
# =============================================================================

# OpenAI
client = openai.OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# LED strip
led_strip = APA102(3)

# VAD model
torch.set_num_threads(1)
print("Loading the VAD model...")
vad_model, (
    get_speech_timestamps,
    save_audio,
    read_audio,
    VADIterator,
    collect_chunks,
) = torch.hub.load(
    repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=True
)


# =============================================================================
# Functions
# =============================================================================


def get_samples(audio: io.BytesIO, format="aac", codec=None) -> np.array:
    """
    Get the samples from an audio file, which can be played by sounddevice.
    """
    # Play the response
    samples = pydub.AudioSegment.from_file(audio, format=format, codec=codec)
    samples = samples.set_frame_rate(44100)
    return samples.get_array_of_samples()


def synthetize_response(response: str) -> io.BytesIO:
    """
    Synthetize the response and return an AAC file wrapped in a BytesIO object.
    """
    # Synthesize the response
    response = client.audio.speech.create(
        input=response,
        model="tts-1",
        voice="onyx",
        response_format="aac",
    )
    # Play the response
    audio_file = io.BytesIO(response.content)
    return audio_file


def record() -> io.BytesIO:
    """
    Record audio from the microphone and return it as a WebM file wrapped in a BytesIO object.
    """
    SAMPLE_RATE = 16000
    FRAME_SIZE = 8000
    CHANNELS = 2

    # Number of quiet frames before we stop recording
    # 1024 *
    NUM_MAX_QUITE_SAMPLES = int(16000 * 1.5)

    # Reset the VAD model before recording
    vad_model.reset_states()
    # Frames of audio being recorded
    recorded_frames = None
    # Whether we are recording or not
    is_recording = False
    # Number of quiet frames
    quiet_samples = 0

    # Start streaming
    print("Recording...")
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
    ) as stream:
        while True:
            print("Reading...")
            frame_i16, overflow = stream.read(FRAME_SIZE)
            if overflow:
                print("Overflow!")

            # Convert the stereo audio to mono
            frame_i16 = np.mean(frame_i16, axis=1, dtype="int16")
            # Create a float32 version of the frame (required by the VAD model)
            frame_f32 = (frame_i16 / 32768.0).astype("float32")

            # Run the VAD model
            # model_out = vad_model(torch.from_numpy(frame_f32), sr=SAMPLE_RATE)
            # Get the probability of speech
            # speech_prob = model_out[0][0]
            speech_prob = 0.1

            if not is_recording and speech_prob > 0.5:
                print("Voice detected, start recording...")
                is_recording = True

            if is_recording:
                # We check (with less sensitivity) if there is still speech
                if speech_prob < 0.5:
                    quiet_samples += len(frame_i16)
                else:
                    # If there is speech, we reset the counter
                    quiet_samples = 0
            else:
                # If we are not recording we still store the current frame, as it might contain the start of a sentence
                recorded_frames = frame_i16

            # If there are more than 10 quiet frames we consider the recording complete, and break the loop
            if quiet_samples > NUM_MAX_QUITE_SAMPLES:
                print("Voice stopped, stop recording...")
                break

            # Add the frame to the recording
            recorded_frames = np.append(recorded_frames, frame_i16)

    # Convert to WebM
    segment = pydub.AudioSegment(
        recorded_frames,
        frame_rate=SAMPLE_RATE,
        sample_width=2,
        channels=1,
    )
    audio_file = segment.export(format="webm", codec="libopus").read()
    # Wrap the WebM file in a BytesIO object, so we can use it as a file
    buffer = io.BytesIO(audio_file)
    buffer.name = "recording.webm"
    return buffer


def handle_conversation(thread: Thread, welcome_message_samples: np.array) -> bool:
    """
    Handle a conversation with the user.

    Args:
        thread: The thread to use for the conversation.
        welcome_message_samples: The samples of the welcome message, which will be played at the end of the conversation.

    Returns:
        True if the conversation finished, False otherwise.
    """

    ASSISTANT_ID = "asst_zmxagj83hfxswc1MSImQv4A1"

    webm_file = record()
    # Upload the file to OpenAI
    print("Uploading...")
    response = client.audio.transcriptions.create(
        file=webm_file, model="whisper-1", response_format="text"
    )
    print(response)

    # If we hear "thank you", stop the whole conversation
    lowercase_response = response.lower()
    if "thank you" in lowercase_response or "thanks" in lowercase_response:
        # Play the welcome message
        sd.play(welcome_message_samples)
        sd.wait()
        return True

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
    samples = get_samples(io.BytesIO(response.content), "aac")
    # Play the response
    sd.play(samples)
    sd.wait()

    return False


def get_welcome_message() -> io.BytesIO:
    WELCOME_FILE = "cache/welcome.aac"

    # If welcome.aac doesn't exist, create it
    if not os.path.exists(WELCOME_FILE):
        # Synthetize the welcome message
        print("Synthetizing the welcome message...")
        welcome_response = synthetize_response("You are welcome!")
        # Save the welcome message
        with open(WELCOME_FILE, "wb") as f:
            f.write(welcome_response.read())

    # Return the welcome message
    return io.BytesIO(open(WELCOME_FILE, "rb").read())


# =============================================================================
# Main
# =============================================================================


def main():
    setup_input()

    # Make the cache/ directory if it doesn't exist
    if not os.path.exists("cache"):
        os.mkdir("cache")

    welcome_message_samples = get_samples(get_welcome_message(), "aac")
    while True:
        wait_for_input()
        print("Starting a new thread...")
        thread = client.beta.threads.create()
        while not handle_conversation(thread, welcome_message_samples):
            print("Continuing the conversation...")


if __name__ == "__main__":
    main()
    # while True:
    #     webm = record()
    #     # Play the response
    #     sd.play(get_samples(webm, "webm", "libopus"))
    #     sd.wait()
