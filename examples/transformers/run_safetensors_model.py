# Use a pipeline as a high-level helper
from transformers import pipeline

model = "./Qwen/Qwen3-0.6B"

pipe = pipeline("text-generation", model=model)
messages = [
    {"role": "user", "content": "Who are you?"},
]
result = pipe(messages, max_new_tokens=10)
print(result)