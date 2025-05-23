import os
from typing import Dict, Optional, Union

import numpy as np

import jax.numpy as jnp
from jax import Array
from safetensors import numpy, safe_open


def save(tensors: Dict[str, Array], metadata: Optional[Dict[str, str]] = None) -> bytes:
    """
    Saves a dictionary of tensors into raw bytes in safetensors format.

    Args:
        tensors (`Dict[str, Array]`):
            The incoming tensors. Tensors need to be contiguous and dense.
        metadata (`Dict[str, str]`, *optional*, defaults to `None`):
            Optional text only metadata you might want to save in your header.
            For instance it can be useful to specify more about the underlying
            tensors. This is purely informative and does not affect tensor loading.

    Returns:
        `bytes`: The raw bytes representing the format

    Example:

    ```python
    from safetensors.flax import save
    from jax import numpy as jnp

    tensors = {"embedding": jnp.zeros((512, 1024)), "attention": jnp.zeros((256, 256))}
    byte_data = save(tensors)
    ```
    """
    np_tensors = _jnp2np(tensors)
    return numpy.save(np_tensors, metadata=metadata)


def save_file(
    tensors: Dict[str, Array],
    filename: Union[str, os.PathLike],
    metadata: Optional[Dict[str, str]] = None,
) -> None:
    """
    Saves a dictionary of tensors into raw bytes in safetensors format.

    Args:
        tensors (`Dict[str, Array]`):
            The incoming tensors. Tensors need to be contiguous and dense.
        filename (`str`, or `os.PathLike`)):
            The filename we're saving into.
        metadata (`Dict[str, str]`, *optional*, defaults to `None`):
            Optional text only metadata you might want to save in your header.
            For instance it can be useful to specify more about the underlying
            tensors. This is purely informative and does not affect tensor loading.

    Returns:
        `None`

    Example:

    ```python
    from safetensors.flax import save_file
    from jax import numpy as jnp

    tensors = {"embedding": jnp.zeros((512, 1024)), "attention": jnp.zeros((256, 256))}
    save_file(tensors, "model.safetensors")
    ```
    """
    np_tensors = _jnp2np(tensors)
    return numpy.save_file(np_tensors, filename, metadata=metadata)


def load(data: bytes) -> Dict[str, Array]:
    """
    Loads a safetensors file into flax format from pure bytes.

    Args:
        data (`bytes`):
            The content of a safetensors file

    Returns:
        `Dict[str, Array]`: dictionary that contains name as key, value as `Array` on cpu

    Example:

    ```python
    from safetensors.flax import load

    file_path = "./my_folder/bert.safetensors"
    with open(file_path, "rb") as f:
        data = f.read()

    loaded = load(data)
    ```
    """
    flat = numpy.load(data)
    return _np2jnp(flat)


def load_file(filename: Union[str, os.PathLike]) -> Dict[str, Array]:
    """
    Loads a safetensors file into flax format.

    Args:
        filename (`str`, or `os.PathLike`)):
            The name of the file which contains the tensors

    Returns:
        `Dict[str, Array]`: dictionary that contains name as key, value as `Array`

    Example:

    ```python
    from safetensors.flax import load_file

    file_path = "./my_folder/bert.safetensors"
    loaded = load_file(file_path)
    ```
    """
    result = {}
    with safe_open(filename, framework="flax") as f:
        for k in f.offset_keys():
            result[k] = f.get_tensor(k)
    return result


def _np2jnp(numpy_dict: Dict[str, np.ndarray]) -> Dict[str, Array]:
    for k, v in numpy_dict.items():
        numpy_dict[k] = jnp.array(v)
    return numpy_dict


def _jnp2np(jnp_dict: Dict[str, Array]) -> Dict[str, np.array]:
    for k, v in jnp_dict.items():
        jnp_dict[k] = np.asarray(v)
    return jnp_dict


def save_crypto(
    tensors: Dict[str, Array],
    metadata: Optional[Dict[str, str]] = None,
    config: Optional[dict] = None
) -> bytes:
    """
    Saves a dictionary of tensors into encrypted raw bytes in safetensors format.

    Args:
        tensors (Dict[str, Array]):
            The input tensors. Tensors must be contiguous and dense.
        metadata (Optional[Dict[str, str]], optional):
            Optional text-only metadata to save in the header.
        config (Optional[dict], optional):
            Encryption configuration, must include encryption/signature keys, etc.

    Returns:
        bytes: The encrypted safetensors format raw bytes.

    Example:
        >>> from safetensors.flax import save_crypto
        >>> from jax import numpy as jnp
        >>> tensors = {"embedding": jnp.zeros((512, 1024)), "attention": jnp.zeros((256, 256))}
        >>> config = {"enc_key": {...}, "sign_key": {...}}
        >>> byte_data = save_crypto(tensors, config=config)
    """
    np_tensors = _jnp2np(tensors)
    from safetensors import numpy as st_numpy
    return st_numpy.save_crypto(np_tensors, metadata=metadata, config=config)


def save_file_crypto(
    tensors: Dict[str, Array],
    filename: Union[str, os.PathLike],
    metadata: Optional[Dict[str, str]] = None,
    config: Optional[dict] = None
) -> None:
    """
    Saves a dictionary of tensors into an encrypted safetensors file.

    Args:
        tensors (Dict[str, Array]):
            The input tensors. Tensors must be contiguous and dense.
        filename (str or os.PathLike):
            The filename to save into.
        metadata (Optional[Dict[str, str]], optional):
            Optional text-only metadata to save in the header.
        config (Optional[dict], optional):
            Encryption configuration, must include encryption/signature keys, etc.

    Returns:
        None

    Example:
        >>> from safetensors.flax import save_file_crypto
        >>> from jax import numpy as jnp
        >>> tensors = {"embedding": jnp.zeros((512, 1024)), "attention": jnp.zeros((256, 256))}
        >>> config = {"enc_key": {...}, "sign_key": {...}}
        >>> save_file_crypto(tensors, "model.safetensors", config=config)
    """
    np_tensors = _jnp2np(tensors)
    from safetensors import numpy as st_numpy
    return st_numpy.save_file_crypto(np_tensors, filename, metadata=metadata, config=config)
