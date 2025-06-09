from vllm import LLM, SamplingParams
import os

# Set the environment variable for loading the encryption key
current_abs_path = os.path.dirname(os.path.abspath(__file__))
os.environ["CRYPTOTENSOR_KEY_JKU"] = f"file://{current_abs_path}/key.jwk"
print(f"Loading encryption key from {os.environ['CRYPTOTENSOR_KEY_JKU']}")

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

model = "./Qwen/Qwen3-0.6B-Enc"

llm = LLM(model=model)

outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
