# 📘 Cryptotensors Examples


## 1. Prepare your environment
### 1.1 Create and Activate a Conda Environment (Recommended)
```bash
conda create -n your_env python=3.10 -y
conda activate your_env
```
### 1.2 Install transformers or vllm
```bash
# If your project uses the transformers library to deploy the model，
pip install "transformers[torch]"

# if uses the vllm
pip install vllm
```
### 1.3 Install Cryptotensors
First, Install Rust and Clone the Repository.
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup update

git clone https://github.com/afxnd/safetensors
```
Second, Uninstall Existing `safetensors`.
```bash
pip uninstall safetensors
```
Finally, Install `cryptotensors`.
```bash
cd safetensors/bindings/python
pip install setuptools_rust
pip install -e .

cd ../compatible
pip install -e.
```

## 2. Download Model and Convert it to Cryptotensors
You can download the model from huggingface.
```bash
# cd to safetensors/examples
cd ../../examples
huggingface-cli download Qwen/Qwen3-0.6B --local-dir ./Qwen/Qwen3-0.6B
```
Run `write_jwk.py`. This will generate a `key.jwk` file in current work directory.
```bash
# Generate Key
python write_jwk.py
```
Convert the model to cryptotensors. Run `convert_model.py` to encrypt several tensors in the model. You can set the value of `num_of_encrypted_tensors` to specify the number of encrypted tensors.
```bash
# copy the folder 
cp -r ./Qwen/Qwen3-0.6B ./Qwen/Qwen3-0.6B-Enc

# convert
python convert_model.py

# print the model's metadata
python print_metadata.py ./Qwen/Qwen3-0.6B-Enc/model.safetensors
```

## 3. Run transformers with cryptotensors
You **don't need to modify any code** in your project to load the cryptotensors model. Cryptotensors is fully compatible with safetensors. You can seamlessly load the encrypted model by simply specifying the path of the `key.jwk` file in the environment variable. You can see examples of deploying models using the `transformers` and `cryptotensors` libraries in the `transformers/` folder.
```bash
# load safetensors model
python transformers/run_safetensors_model.py

# load cryptotensors model
# you need to move your `key.jwk` file to the `transformers/` folder; or delete lines 6 and 7, and set `CRYPTOTENSOR_KEY_JKU=your_key_path` in the CLI.
python transformers/run_cryptotensors_model.py
```

## 4. Use vLLM to deploy model
`vllm/` folder contains examples that use `vllm` and `cryptotensors`. When using Cryptotensors with your vLLM project, you do not need to modify any code of your project.
```bash
# deploy safetensors model
python vllm/run_safetensors_vllm.py

# deploy cryptotensors model
# you need to move your `key.jwk` file to the `vllm/` folder; or delete lines 5 and 6, and set `CRYPTOTENSOR_KEY_JKU=your_key_path` in the CLI.
python vllm/run_cryptotensors_vllm.py
```

## 5. Docker: Install Cryptotensors in your images.
We take the `vllm/vllm-openai` image as an example (you can pull it through `docker pull vllm/vllm-openai`). Run the following command to build a new image with `Cryptotensors`.
```bash
docker build -t myimage:tag .
```
You can modify the `FROM vllm/vllm-openai` in `Dockerfile` to your own images.