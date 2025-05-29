import json
import os

path = "./key.jwk"

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
keys = [enc_key, sign_key]
jwk = {"keys": keys}

with open(path, "w") as f:
    json.dump(jwk, f, indent=2)