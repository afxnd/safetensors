FROM vllm/vllm-openai:latest

# ENV RUSTUP_DIST_SERVER=https://mirrors.tuna.tsinghua.edu.cn/rustup
# ENV RUSTUP_UPDATE_ROOT=https://mirrors.tuna.tsinghua.edu.cn/rustup/rustup
# ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV CRYPTOTENSOR_KEY_JKU=file:///root/safetensors/examples/key.jwk
ENV PATH="/root/.cargo/bin:${PATH}"

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | bash -s -- -y
RUN pip install setuptools_rust
RUN pip install build
RUN pip uninstall -y safetensors
COPY ./safetensors /root/safetensors

SHELL ["/bin/bash", "-c"]
RUN source /root/.cargo/env
RUN rustup update
WORKDIR /root/safetensors/bindings/python/
RUN python3 -m build --wheel
RUN pip install ./target/wheels/*.whl
WORKDIR /root/safetensors/bindings/compatible/
RUN python3 -m build --wheel
RUN pip install ./dist/*.whl
WORKDIR /root/safetensors/examples/
RUN python3 write_jwk.py
