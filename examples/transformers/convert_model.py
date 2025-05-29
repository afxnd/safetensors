## Convert safetensors model to cryptotensors model

from cryptotensors.torch import load_file, save_file
import random
import os

# Need to download the model from huggingface
# https://huggingface.co/Qwen/Qwen3-0.6B
# then copy the folder `cp -r Qwen/Qwen3-0.6B ./Qwen/Qwen3-0.6B-Enc`

model_in = "./Qwen/Qwen3-0.6B/model.safetensors"
model_out = "./Qwen/Qwen3-0.6B-Enc/model.safetensors"

# check if the model exists
if not os.path.exists(model_in):
    raise FileNotFoundError(f"Model file {model_in} not found, please download the model from huggingface")
if not os.path.exists(model_out):
    raise FileNotFoundError(f"Model file {model_out} not found, please copy the model to the output folder")

# Load the model
model = load_file(model_in)

# Serialize the model
num_of_encrypted_tensors = 3
# random select encrypted tensors
tensors = random.sample(sorted(model.keys()), num_of_encrypted_tensors)
print(f"Encrypting {num_of_encrypted_tensors} tensors:")
for tensor in tensors:
    print(f"\t{tensor}")

enc_key = {
    "kty": "oct",
    "alg": "aes256gcm",
    "k": "L+xl38kCEteXk+6Tm1mzu5JvFriVibzAsgpYX2WmAgA=",
    "kid": "test-enc-key",
}
sign_key = {
    "kty": "okp",
    "alg": "ed25519",
    "d": "uTKTjQL6pX1Tqb7Hpor4A1s+TdgHReQEITZWWAf7DIc=",
    "x": "xkqFcGjXCBMk75q0259N1ggRJsqc+FTAiXMuKX72fd8=",
    "kid": "test-sign-key",
}
config = {
    "tensors": tensors,
    "enc_key": enc_key,
    "sign_key": sign_key,
}

metadata = {
    "format": "pt" # transformers requires this field to be present
}

save_file(model, model_out, config=config, metadata=metadata)

print(f"Encrypted model saved to {model_out}")