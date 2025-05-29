# Use a pipeline as a high-level helper
from transformers import pipeline
import os

# Set the environment variable for loading the encryption key
current_abs_path = os.path.dirname(os.path.abspath(__file__))
os.environ["CRYPTOTENSOR_KEY_JKU"] = f"file://{current_abs_path}/key.jwk"
print(f"Loading encryption key from {os.environ['CRYPTOTENSOR_KEY_JKU']}")

model = "./Qwen/Qwen3-0.6B-Enc"

pipe = pipeline("text-generation", model=model)
messages = [
    {"role": "user", "content": "Who are you?"},
]
result = pipe(messages, max_new_tokens=10)
print(result)