# Generated content DO NOT EDIT
@staticmethod
def deserialize(bytes):
    """
    Opens a safetensors lazily and returns tensors as asked

    Args:
        data (`bytes`):
            The byte content of a file

    Returns:
        (`List[str, Dict[str, Dict[str, any]]]`):
            The deserialized content is like:
                [("tensor_name", {"shape": [2, 3], "dtype": "F32", "data": b"\0\0.." }), (...)]
    """
    pass

@staticmethod
def serialize(tensor_dict, metadata=None, config=None):
    """
    Serializes raw data.

    Args:
        tensor_dict (`Dict[str, Dict[Any]]`):
            The tensor dict is like:
                {"tensor_name": {"dtype": "F32", "shape": [2, 3], "data": b"\0\0"}}
        metadata (`Dict[str, str]`, *optional*):
            The optional purely text annotations
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
        (`bytes`):
            The serialized content.
    """
    pass

@staticmethod
def serialize_file(tensor_dict, filename, metadata=None, config=None):
    """
    Serializes raw data into file.

    Args:
        tensor_dict (`Dict[str, Dict[Any]]`):
            The tensor dict is like:
                {"tensor_name": {"dtype": "F32", "shape": [2, 3], "data": b"\0\0"}}
        filename (`str`, or `os.PathLike`):
            The name of the file to write into.
        metadata (`Dict[str, str]`, *optional*):
            The optional purely text annotations
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
        (`NoneType`):
            On success return None
    """
    pass

class safe_open:
    """
    Opens a safetensors lazily and returns tensors as asked

    Args:
        filename (`str`, or `os.PathLike`):
            The filename to open

        framework (`str`):
            The framework you want you tensors in. Supported values:
            `pt`, `tf`, `flax`, `numpy`.

        device (`str`, defaults to `"cpu"`):
            The device on which you want the tensors.
    """

    def __init__(self, filename, framework, device=...):
        pass

    def __enter__(self):
        """
        Start the context manager
        """
        pass

    def __exit__(self, _exc_type, _exc_value, _traceback):
        """
        Exits the context manager
        """
        pass

    def get_slice(self, name):
        """
        Returns a full slice view object

        Args:
            name (`str`):
                The name of the tensor you want

        Returns:
            (`PySafeSlice`):
                A dummy object you can slice into to get a real tensor
        Example:
        ```python
        from cryptotensors import safe_open

        with safe_open("model.safetensors", framework="pt", device=0) as f:
            tensor_part = f.get_slice("embedding")[:, ::8]

        ```
        """
        pass

    def get_tensor(self, name):
        """
        Returns a full tensor

        Args:
            name (`str`):
                The name of the tensor you want

        Returns:
            (`Tensor`):
                The tensor in the framework you opened the file for.

        Example:
        ```python
        from cryptotensors import safe_open

        with safe_open("model.safetensors", framework="pt", device=0) as f:
            tensor = f.get_tensor("embedding")

        ```
        """
        pass

    def keys(self):
        """
        Returns the names of the tensors in the file.

        Returns:
            (`List[str]`):
                The name of the tensors contained in that file
        """
        pass

    def metadata(self):
        """
        Return the special non tensor information in the header

        Returns:
            (`Dict[str, str]`):
                The freeform metadata.
        """
        pass

    def offset_keys(self):
        """
        Returns the names of the tensors in the file, ordered by offset.

        Returns:
            (`List[str]`):
                The name of the tensors contained in that file
        """
        pass

class SafetensorError(Exception):
    """
    Custom Python Exception for Safetensor errors.
    """
