import os
from typing import Dict, Optional, Union, Any

import numpy as np
import tensorflow as tf

from safetensors import numpy, safe_open


def save(tensors: Dict[str, tf.Tensor], metadata: Optional[Dict[str, str]] = None, config: Optional[Dict[str, Any]] = None) -> bytes:
    """
    Saves a dictionary of tensors into raw bytes in safetensors format.

    Args:
        tensors (`Dict[str, tf.Tensor]`):
            The incoming tensors. Tensors need to be contiguous and dense.
        metadata (`Dict[str, str]`, *optional*, defaults to `None`):
            Optional text only metadata you might want to save in your header.
            For instance it can be useful to specify more about the underlying
            tensors. This is purely informative and does not affect tensor loading.
        config (`Dict[str, Any]`, optional):
            Encryption configuration, structure as follows:
                {
                    "tensors": ["tensor1", "tensor2"],  # List of tensor names to encrypt; if None, encrypt all
                    "enc_key": {  # Encryption key, supports JWK format
                        "alg": "aes256gcm", "kid": "test-enc-key", "key": "..."
                    },
                    "sign_key": {  # Signing key, supports Ed25519, etc.
                        "alg": "ed25519", "kid": "test-sign-key", "private": "...", "public": "..."
                    },
                    "policy": {  # Optional, load policy
                        "local": "...", "remote": "..."
                    }
                }

    Returns:
        `bytes`: The raw bytes representing the format

    Example:

    ```python
    from safetensors.tensorflow import save
    import tensorflow as tf

    tensors = {"embedding": tf.zeros((512, 1024)), "attention": tf.zeros((256, 256))}
    byte_data = save(tensors)
    ```
    """
    np_tensors = _tf2np(tensors)
    return numpy.save(np_tensors, metadata=metadata, config=config)


def save_file(
    tensors: Dict[str, tf.Tensor],
    filename: Union[str, os.PathLike],
    metadata: Optional[Dict[str, str]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Saves a dictionary of tensors into raw bytes in safetensors format.

    Args:
        tensors (`Dict[str, tf.Tensor]`):
            The incoming tensors. Tensors need to be contiguous and dense.
        filename (`str`, or `os.PathLike`)):
            The filename we're saving into.
        metadata (`Dict[str, str]`, *optional*, defaults to `None`):
            Optional text only metadata you might want to save in your header.
            For instance it can be useful to specify more about the underlying
            tensors. This is purely informative and does not affect tensor loading.
        config (`Dict[str, Any]`, optional):
            Encryption configuration, structure as follows:
                {
                    "tensors": ["tensor1", "tensor2"],  # List of tensor names to encrypt; if None, encrypt all
                    "enc_key": {  # Encryption key, supports JWK format
                        "alg": "aes256gcm", "kid": "test-enc-key", "key": "..."
                    },
                    "sign_key": {  # Signing key, supports Ed25519, etc.
                        "alg": "ed25519", "kid": "test-sign-key", "private": "...", "public": "..."
                    },
                    "policy": {  # Optional, load policy
                        "local": "...", "remote": "..."
                    }
                }

    Returns:
        `None`

    Example:

    ```python
    from safetensors.tensorflow import save_file
    import tensorflow as tf

    tensors = {"embedding": tf.zeros((512, 1024)), "attention": tf.zeros((256, 256))}
    save_file(tensors, "model.safetensors")
    ```
    """
    np_tensors = _tf2np(tensors)
    return numpy.save_file(np_tensors, filename, metadata=metadata, config=config)


def load(data: bytes) -> Dict[str, tf.Tensor]:
    """
    Loads a safetensors file into tensorflow format from pure bytes.

    Args:
        data (`bytes`):
            The content of a safetensors file

    Returns:
        `Dict[str, tf.Tensor]`: dictionary that contains name as key, value as `tf.Tensor` on cpu

    Example:

    ```python
    from safetensors.tensorflow import load

    file_path = "./my_folder/bert.safetensors"
    with open(file_path, "rb") as f:
        data = f.read()

    loaded = load(data)
    ```
    """
    flat = numpy.load(data)
    return _np2tf(flat)


def load_file(filename: Union[str, os.PathLike]) -> Dict[str, tf.Tensor]:
    """
    Loads a safetensors file into tensorflow format.

    Args:
        filename (`str`, or `os.PathLike`)):
            The name of the file which contains the tensors

    Returns:
        `Dict[str, tf.Tensor]`: dictionary that contains name as key, value as `tf.Tensor`

    Example:

    ```python
    from safetensors.tensorflow import load_file

    file_path = "./my_folder/bert.safetensors"
    loaded = load_file(file_path)
    ```
    """
    result = {}
    with safe_open(filename, framework="tf") as f:
        for k in f.offset_keys():
            result[k] = f.get_tensor(k)
    return result


def _np2tf(numpy_dict: Dict[str, np.ndarray]) -> Dict[str, tf.Tensor]:
    for k, v in numpy_dict.items():
        numpy_dict[k] = tf.convert_to_tensor(v)
    return numpy_dict


def _tf2np(tf_dict: Dict[str, tf.Tensor]) -> Dict[str, np.array]:
    for k, v in tf_dict.items():
        tf_dict[k] = v.numpy()
    return tf_dict