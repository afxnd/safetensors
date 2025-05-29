import json
import struct
import sys

def try_decode(val):
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return val
    return val

def print_json(json_obj, indent=4):
    if "__crypto_keys__" in json_obj:
        value = try_decode(json_obj["__crypto_keys__"])
        if isinstance(value, dict):
            print(" " * indent + "__crypto_keys__:")
            for k, v in value.items():
                print(" " * (indent + 4) + f"{k}: {v}")
        else:
            print(" " * indent + f"__crypto_keys__: {value}")
    if "__encryption__" in json_obj:
        value = try_decode(json_obj["__encryption__"])
        if isinstance(value, dict):
            print(" " * indent + "__encryption__:")
            for k, v in value.items():
                v_decoded = try_decode(v)
                print(" " * (indent + 4) + f"{k}:")
                print(" " * (indent + 8) + json.dumps(v_decoded, ensure_ascii=False, indent=4).replace("\n", "\n" + " " * (indent + 8)))
        else:
            print(" " * indent + f"__encryption__: {value}")
    if "__policy__" in json_obj:
        value = try_decode(json_obj["__policy__"])
        if isinstance(value, dict):
            print(" " * indent + "__policy__:")
            for k, v in value.items():
                print(" " * (indent + 4) + f"{k}: {repr(v)}")
        else:
            print(" " * indent + f"__policy__: {value}")
    if "__signature__" in json_obj:
        print(" " * indent + f"__signature__: {json_obj['__signature__']}")
    # other keys
    for k, v in json_obj.items():
        if k not in ["__crypto_keys__", "__encryption__", "__policy__", "__signature__"]:
            print(" " * indent + f"{k}: {v}")

def print_safetensors_header(filename):
    with open(filename, "rb") as f:
        # The first 8 bytes of the safetensors file are the header length (little-endian uint64)
        header_len_bytes = f.read(8)
        header_len = struct.unpack("<Q", header_len_bytes)[0]
        # Read the header content
        header_bytes = f.read(header_len)
        header = json.loads(header_bytes.decode("utf-8"))
        if "__metadata__" in header:
            print("\"__metadata__\": {")
            print_json(header["__metadata__"])
            print("}")
            
        else:
            print("No metadata found")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        print_safetensors_header(filename)
    else:
        print("Usage: python print_header.py <safetensors file name>")
        sys.exit(1)