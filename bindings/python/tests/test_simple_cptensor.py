import os
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np
import torch
import json

from safetensors import SafetensorError, safe_open, serialize
from safetensors.numpy import load, load_file, save, save_file
from safetensors.torch import _find_shared_tensors
from safetensors.torch import load_file as load_file_pt
from safetensors.torch import save_file as save_file_pt
from safetensors.torch import storage_ptr, storage_size

def write_jwk_tmp():
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
        jwk_tmp = tempfile.NamedTemporaryFile(suffix=".jwk", delete=False, mode="w")
        keys = [enc_key, sign_key]
        jwk = {"keys": keys}
        json.dump(jwk, jwk_tmp, indent=2)
        jwk_tmp.flush()
        jwk_tmp.close()
        enc_key["jku"] = f"file://{jwk_tmp.name}"
        sign_key["jku"] = f"file://{jwk_tmp.name}"
        return jwk_tmp.name, enc_key, sign_key

def delete_jwk_tmp(path):
    os.remove(path)

class TestCase(unittest.TestCase):

    def test_accept_path(self):
        tensors = {
            "a": torch.zeros((2, 2)),
            "b": torch.zeros((2, 3), dtype=torch.uint8),
        }
        jwk_path, enc_key, sign_key = write_jwk_tmp()
        config = {
            "enc_key": enc_key,
            "sign_key": sign_key,
        }
        filename = f"./out_{threading.get_ident()}.safetensors"
        save_file_pt(tensors, Path(filename), config=config)
        load_file_pt(Path(filename))
        os.remove(Path(filename))
        delete_jwk_tmp(jwk_path)


class WindowsTestCase(unittest.TestCase):
    def test_get_correctly_dropped(self):
        tensors = {
            "a": torch.zeros((2, 2)),
            "b": torch.zeros((2, 3), dtype=torch.uint8),
        }
        jwk_path, enc_key, sign_key = write_jwk_tmp()
        config = {
            "enc_key": enc_key,
            "sign_key": sign_key,
        }
        save_file_pt(tensors, "./out_windows.safetensors", config=config)
        with safe_open("./out_windows.safetensors", framework="pt") as f:
            pass

        with self.assertRaises(SafetensorError):
            print(f.keys())

        with open("./out_windows.safetensors", "w") as g:
            g.write("something")

        delete_jwk_tmp(jwk_path)

class ReadmeTestCase(unittest.TestCase):
    def assertTensorEqual(self, tensors1, tensors2, equality_fn):
        self.assertEqual(tensors1.keys(), tensors2.keys(), "tensor keys don't match")

        for k, v1 in tensors1.items():
            v2 = tensors2[k]

            self.assertTrue(equality_fn(v1, v2), f"{k} tensors are different")

    def test_numpy_example(self):
        tensors = {"a": np.zeros((2, 2)), "b": np.zeros((2, 3), dtype=np.uint8)}

        jwk_path, enc_key, sign_key = write_jwk_tmp()
        config = {
            "enc_key": enc_key,
            "sign_key": sign_key,
        }
        save_file(tensors, "./out_np.safetensors", config=config)
        out = save(tensors, config=config)

        # Now loading
        loaded = load_file("./out_np.safetensors")
        self.assertTensorEqual(tensors, loaded, np.allclose)

        loaded = load(out)
        self.assertTensorEqual(tensors, loaded, np.allclose)
        delete_jwk_tmp(jwk_path)

    def test_numpy_bool(self):
        tensors = {"a": np.asarray(False)}
        jwk_path, enc_key, sign_key = write_jwk_tmp()
        config = {
            "enc_key": enc_key,
            "sign_key": sign_key,
        }
        save_file(tensors, "./out_bool.safetensors", config=config)
        out = save(tensors, config=config)

        # Now loading
        loaded = load_file("./out_bool.safetensors")
        self.assertTensorEqual(tensors, loaded, np.allclose)

        loaded = load(out)
        self.assertTensorEqual(tensors, loaded, np.allclose)
        delete_jwk_tmp(jwk_path)

    def test_torch_example(self):
        tensors = {
            "a": torch.zeros((2, 2)),
            "b": torch.zeros((2, 3), dtype=torch.uint8),
        }
        jwk_path, enc_key, sign_key = write_jwk_tmp()
        config = {
            "enc_key": enc_key,
            "sign_key": sign_key,
        }
        # Saving modifies the tensors to type numpy, so we must copy for the
        # test to be correct.
        tensors2 = tensors.copy()
        filename = f"./out_pt_{threading.get_ident()}.safetensors"
        save_file_pt(tensors, filename, config=config)

        # Now loading
        loaded = load_file_pt(filename)
        self.assertTensorEqual(tensors2, loaded, torch.allclose)
        delete_jwk_tmp(jwk_path)

    def test_exception(self):
        flattened = {"test": {"dtype": "float32", "shape": [1]}}

        with self.assertRaises(SafetensorError):
            serialize(flattened)

    def test_torch_slice(self):
        A = torch.randn((10, 5))
        tensors = {
            "a": A,
        }
        ident = threading.get_ident()
        jwk_path, enc_key, sign_key = write_jwk_tmp()
        config = {
            "enc_key": enc_key,
            "sign_key": sign_key,
        }
        save_file_pt(tensors, f"./slice_{ident}.safetensors", config=config)

        # Now loading
        with safe_open(f"./slice_{ident}.safetensors", framework="pt", device="cpu") as f:
            slice_ = f.get_slice("a")
            tensor = slice_[:]
            self.assertEqual(list(tensor.shape), [10, 5])
            torch.testing.assert_close(tensor, A)

            tensor = slice_[tuple()]
            self.assertEqual(list(tensor.shape), [10, 5])
            torch.testing.assert_close(tensor, A)

            tensor = slice_[:2]
            self.assertEqual(list(tensor.shape), [2, 5])
            torch.testing.assert_close(tensor, A[:2])

            tensor = slice_[:, :2]
            self.assertEqual(list(tensor.shape), [10, 2])
            torch.testing.assert_close(tensor, A[:, :2])

            tensor = slice_[0, :2]
            self.assertEqual(list(tensor.shape), [2])
            torch.testing.assert_close(tensor, A[0, :2])

            tensor = slice_[2:, 0]
            self.assertEqual(list(tensor.shape), [8])
            torch.testing.assert_close(tensor, A[2:, 0])

            tensor = slice_[2:, 1]
            self.assertEqual(list(tensor.shape), [8])
            torch.testing.assert_close(tensor, A[2:, 1])

            tensor = slice_[2:, -1]
            self.assertEqual(list(tensor.shape), [8])
            torch.testing.assert_close(tensor, A[2:, -1])

            tensor = slice_[list()]
            self.assertEqual(list(tensor.shape), [0, 5])
            torch.testing.assert_close(tensor, A[list()])

        delete_jwk_tmp(jwk_path)

    def test_numpy_slice(self):
        A = np.random.rand(10, 5)
        tensors = {
            "a": A,
        }
        jwk_path, enc_key, sign_key = write_jwk_tmp()
        config = {
            "enc_key": enc_key,
            "sign_key": sign_key,
        }
        filename = f"./slice_{threading.get_ident()}.safetensors"
        save_file(tensors, filename, config=config)

        # Now loading
        with safe_open(filename, framework="np", device="cpu") as f:
            slice_ = f.get_slice("a")
            tensor = slice_[:]
            self.assertEqual(list(tensor.shape), [10, 5])
            self.assertTrue(np.allclose(tensor, A))

            tensor = slice_[tuple()]
            self.assertEqual(list(tensor.shape), [10, 5])
            self.assertTrue(np.allclose(tensor, A))

            tensor = slice_[:2]
            self.assertEqual(list(tensor.shape), [2, 5])
            self.assertTrue(np.allclose(tensor, A[:2]))

            tensor = slice_[:, :2]
            self.assertEqual(list(tensor.shape), [10, 2])
            self.assertTrue(np.allclose(tensor, A[:, :2]))

            tensor = slice_[0, :2]
            self.assertEqual(list(tensor.shape), [2])
            self.assertTrue(np.allclose(tensor, A[0, :2]))

            tensor = slice_[2:, 0]
            self.assertEqual(list(tensor.shape), [8])
            self.assertTrue(np.allclose(tensor, A[2:, 0]))

            tensor = slice_[2:, 1]
            self.assertEqual(list(tensor.shape), [8])
            self.assertTrue(np.allclose(tensor, A[2:, 1]))

            tensor = slice_[2:, -1]
            self.assertEqual(list(tensor.shape), [8])
            self.assertTrue(np.allclose(tensor, A[2:, -1]))

            tensor = slice_[2:, -5]
            self.assertEqual(list(tensor.shape), [8])
            self.assertTrue(np.allclose(tensor, A[2:, -5]))

            tensor = slice_[list()]
            self.assertEqual(list(tensor.shape), [0, 5])
            self.assertTrue(np.allclose(tensor, A[list()]))

            with self.assertRaises(SafetensorError) as cm:
                tensor = slice_[2:, -6]
            self.assertEqual(str(cm.exception), "Invalid index -6 for dimension 1 of size 5")

            with self.assertRaises(SafetensorError) as cm:
                tensor = slice_[[0, 1]]
            self.assertEqual(str(cm.exception), "Non empty lists are not implemented")

            with self.assertRaises(SafetensorError) as cm:
                tensor = slice_[2:, 20]
            self.assertEqual(
                str(cm.exception),
                "Error during slicing [2:, 20] with shape [10, 5]:  SliceOutOfRange { dim_index: 1, asked: 20, dim_size: 5 }",
            )

            with self.assertRaises(SafetensorError) as cm:
                tensor = slice_[:20]
            self.assertEqual(
                str(cm.exception),
                "Error during slicing [:20] with shape [10, 5]:  SliceOutOfRange { dim_index: 0, asked: 19, dim_size: 10 }",
            )

            with self.assertRaises(SafetensorError) as cm:
                tensor = slice_[:, :20]
            self.assertEqual(
                str(cm.exception),
                "Error during slicing [:, :20] with shape [10, 5]:  SliceOutOfRange { dim_index: 1, asked: 19, dim_size: 5 }",
            )

        delete_jwk_tmp(jwk_path)
