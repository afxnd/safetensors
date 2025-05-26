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
def serialize(tensor_dict, metadata=None):
    """
    Serializes raw data.

    Args:
        tensor_dict (`Dict[str, Dict[Any]]`):
            The tensor dict is like:
                {"tensor_name": {"dtype": "F32", "shape": [2, 3], "data": b"\0\0"}}
        metadata (`Dict[str, str]`, *optional*):
            The optional purely text annotations

    Returns:
        (`bytes`):
            The serialized content.
    """
    pass

@staticmethod
def serialize_file(tensor_dict, filename, metadata=None):
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

    Returns:
        (`NoneType`):
            On success return None.
    """
    pass

def serialize_encrypted(tensor_dict: dict, metadata: dict = ..., config: dict = ...) -> bytes:
    """
    Serializes raw data as encrypted CryptoTensor.

    Args:
        tensor_dict (Dict[str, Dict[Any]]):
            The tensor dict, e.g. {"tensor_name": {"dtype": "F32", "shape": [2, 3], "data": b"..."}}
        metadata (Optional[Dict[str, str]]):
            Optional text-only annotations.
        config (Optional[dict]):
            Encryption configuration, must include encryption/signature keys, etc.

    Returns:
        bytes: The encrypted safetensors format raw bytes.
    """
    ...

def serialize_file_encrypted(tensor_dict: dict, filename, metadata: dict = ..., config: dict = ...) -> None:
    """
    Serializes raw data as encrypted CryptoTensor and writes to file.

    Args:
        tensor_dict (Dict[str, Dict[Any]]):
            The tensor dict, e.g. {"tensor_name": {"dtype": "F32", "shape": [2, 3], "data": b"..."}}
        filename (str or os.PathLike):
            The name of the file to write into.
        metadata (Optional[Dict[str, str]]):
            Optional text-only annotations.
        config (Optional[dict]):
            Encryption configuration, must include encryption/signature keys, etc.

    Returns:
        None
    """
    ...

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
        from safetensors import safe_open

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
        from safetensors import safe_open

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

class SafetensorError(Exception):
    """
    Custom Python Exception for Safetensor errors.
    """
