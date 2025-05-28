import tempfile
import json
import unittest
from pathlib import Path
import os

import numpy as np

from safetensors import safe_open
from safetensors.numpy import load, load_file, save, save_file


class TestCase(unittest.TestCase):

    def write_jwk_tmp(self):
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

    def delete_jwk_tmp(self, path):
        os.remove(path)

    def test_serialization_deserialization(self):
        jwk_path, enc_key, sign_key = self.write_jwk_tmp()
        try:
            config = {
                "enc_key": enc_key,
                "sign_key": sign_key,
            }
            data = np.zeros((2, 2), dtype=np.int32)
            serialized = save({"test": data}, config=config)
            print(repr(serialized))
            deserialized = load(serialized)
            self.assertEqual(deserialized["test"].all(), data.all())
        finally:
            self.delete_jwk_tmp(jwk_path)

    def test_serialization_deserialization_file(self):
        jwk_path, enc_key, sign_key = self.write_jwk_tmp()
        try:
            config = {
                "enc_key": enc_key,
                "sign_key": sign_key,
            }
            data = np.zeros((2, 2), dtype=np.int32)
            save_file({"test": data}, "cryptotensor1.safetensors", config=config)
            deserialized = load_file("cryptotensor1.safetensors")
            self.assertEqual(deserialized["test"].all(), data.all())
        finally:
            self.delete_jwk_tmp(jwk_path)

    def test_serialization_deserialization_safe_open(self):
        jwk_path, enc_key, sign_key = self.write_jwk_tmp()
        try:
            config = {
                "tensors": ["test", "test3", "test4"],
                "enc_key": enc_key,
                "sign_key": sign_key,
            }
            data = np.random.randn(2, 2).astype(np.int32)
            data2 = np.random.randn(2, 2).astype(np.float16)
            data3 = np.zeros((2, 2), dtype=">i4")
            data4 = np.array([True, False])
            save_file({"test": data, "test2": data2, "test3": data3, "test4": data4}, "cryptotensor2.safetensors", config=config, metadata={"framework": "pt"})
            with safe_open("cryptotensor2.safetensors", framework="np") as f:
                self.assertIn("framework", f.metadata())
                self.assertEqual(f.metadata()["framework"], "pt")
                self.assertTrue(np.allclose(f.get_tensor("test"), data))
                self.assertTrue(np.allclose(f.get_tensor("test2"), data2))
                self.assertTrue(np.allclose(f.get_tensor("test3"), data3))
                self.assertTrue(np.allclose(f.get_tensor("test4"), data4))
        finally:
            self.delete_jwk_tmp(jwk_path)