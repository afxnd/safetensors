# Re-export this
from ._safetensors_rust import (  # noqa: F401
    SafetensorError,
    __version__,
    deserialize,
    safe_open,
    serialize,
    serialize_file,
    serialize_crypto,
    serialize_file_crypto,
)
