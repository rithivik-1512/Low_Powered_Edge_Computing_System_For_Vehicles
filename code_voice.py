import os, json, time
import sounddevice as sd
import queue
from vosk import Model, KaldiRecognizer

MODEL_PATH = "C:/Users/chsai/OneDrive/Desktop/review_2_voice_recognition/vosk-model-small-en-us-0.15"
JSON_PATH = "voice_output.json"

# Load model
if not os.path.exists(MODEL_PATH):
    print("Model not found, check MODEL_PATH")
    exit(1)

model = Model(MODEL_PATH)
recognizer = KaldiRecognizer(model, 16000)

q = queue.Queue()

def callback(indata, frames, time_, status):
    if status:
        print(status, flush=True)
    q.put(bytes(indata))

# Reset JSON file at start
with open(JSON_PATH, "w") as f:
    json.dump({"records": []}, f)

with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype="int16",
                       channels=1, callback=callback):

    print("🎤 Speak... (Ctrl+C to stop)")
    try:
        while True:
            data = q.get()

            if recognizer.AcceptWaveform(data):
                # Only final results here
                result = json.loads(recognizer.Result())
                text = result.get("text", "").strip()

                if text:
                    with open(JSON_PATH, "r") as f:
                        all_data = json.load(f)

                    payload = {
                        "ts": time.time(),
                        "voice_text": text
                    }
                    all_data["records"].append(payload)

                    with open(JSON_PATH, "w") as f:
                        json.dump(all_data, f, indent=2)

                    print("Logged:", payload)
    except KeyboardInterrupt:
        print("\n Stopped by user")
