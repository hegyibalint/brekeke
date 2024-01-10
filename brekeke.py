import io
import os
from queue import Queue
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

RECORDING_DEVICE = 0
# Sample rate of the audio
RECORDING_SAMPLE_RATE = 16000
# How big each frame should be, which the callback will receive
RECORDING_FRAME_SIZE = 1024
# Number of channels to record
RECORDING_CHANNELS = 2
# Number of quiet samples before we stop recording
NUM_MAX_QUITE_SAMPLES = int(RECORDING_SAMPLE_RATE * 1)  # 1 second

# Siliero VAD model works the best with multiples of 512-sample frames
if RECORDING_FRAME_SIZE % 512 != 0:
    raise ValueError("FRAME_SIZE must be a multiple of 512")

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

# Queue storing the recorded audio frames coming from the recording callback
recording_queue: Queue = None


# =============================================================================
# Functions
# =============================================================================


def get_samples(audio: io.BytesIO, format="aac", codec=None, sr=44100) -> np.array:
    """
    Get the samples from an audio file, which can be played by sounddevice.
    """
    # Play the response
    samples = pydub.AudioSegment.from_file(audio, format=format, codec=codec)
    samples = samples.set_frame_rate(sr)
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


def record_callback(indata, frames, time, status):
    global recording_queue

    if recording_queue:
        recording_queue.put(indata.copy())


def start_recording() -> io.BytesIO:
    """
    Starts recording audio from the microphone and returns an InputStream object.
    """

    # Start streaming
    print("Setting up recording...")
    return sd.InputStream(
        device=RECORDING_DEVICE,
        samplerate=RECORDING_SAMPLE_RATE,
        channels=RECORDING_CHANNELS,
        blocksize=RECORDING_FRAME_SIZE,
        callback=record_callback,
    )


def record() -> io.BytesIO:
    """
    Record audio from the microphone and return it as a WebM file wrapped in a BytesIO object.
    """
    global recording_queue

    # The last number of recorded samples we keep, regardless of whether they is speech or not
    # This is done to avoid cutting the audio when we stop recording
    KEPT_SAMPLES = int(RECORDING_SAMPLE_RATE * 0.5)

    # Reset the VAD model before recording
    vad_model.reset_states()
    # Frames of audio being recorded
    recorded_frames = []
    # Whether we are recording or not
    is_recording = False
    # Number of quiet frames
    quiet_samples = 0

    print("Start receiving audio...")
    # Create the queue for the recording callback
    recording_queue = Queue()

    while True:
        frame = recording_queue.get()
        print("Frame received")

        # Convert the stereo audio to mono
        frame = np.mean(frame, axis=1)

        # Run the VAD model
        model_out = vad_model(torch.from_numpy(frame), sr=RECORDING_SAMPLE_RATE)
        # Get the probability of speech
        speech_prob = model_out[:, 0].max()
        print(f"Speech probability: {speech_prob:.2f}")

        if not is_recording and speech_prob > 0.9:
            print("Voice detected, start recording...")
            is_recording = True

        # We append the frame to the list of recorded frames
        recorded_frames = np.append(recorded_frames, frame)

        if is_recording:
            # We check (with less sensitivity) if there is still speech
            if speech_prob < 0.75:
                quiet_samples += len(frame)
            else:
                # If there is speech, we reset the counter
                quiet_samples = 0
        else:
            # If we are not recording, we still keep the last KEPT_SAMPLES frames
            recorded_frames = recorded_frames[-KEPT_SAMPLES:]

        # If there are more than 10 quiet frames we consider the recording complete, and break the loop
        if quiet_samples > NUM_MAX_QUITE_SAMPLES:
            print("Voice stopped, stop recording...")
            break

    print("Recording finished")
    recording_queue = None

    recorded_frames_i16 = (recorded_frames * 32767).astype(np.int16)
    # Convert to WebM
    segment = pydub.AudioSegment(
        recorded_frames_i16,
        frame_rate=RECORDING_SAMPLE_RATE,
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
        file=webm_file, model="whisper-1", response_format="text", language="en"
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
    # We start "recording", as in we are receiving frames, but drop them until we need them
    # This is done to avoid the delay when starting the recording, which is significant
    with start_recording():
        while True:
            print("Starting a new conversation thread...")
            thread = client.beta.threads.create()
            wait_for_input()
            while not handle_conversation(thread, welcome_message_samples):
                print("Continuing the conversation...")


def test_recording():
    setup_input()

    with start_recording():
        while True:
            print("Recording...")
            wait_for_input()
            webm = record()
            samples = get_samples(webm, "webm", "libopus")
            sd.play(samples)
            sd.wait()


if __name__ == "__main__":
    main()
    # test_recording()
