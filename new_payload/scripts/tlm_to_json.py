#!/usr/bin/env python3

import json
import struct
import sys
import zlib

MAGIC = b"TLM1"


def tlm_to_json(input_path: str, output_path: str) -> None:
    records = []

    with open(input_path, "rb") as f:
        header = f.read(4)
        if header != MAGIC:
            raise ValueError(f"Invalid .tlm file header: {header!r}")

        while True:
            length_bytes = f.read(4)
            if not length_bytes:
                break
            if len(length_bytes) != 4:
                raise ValueError("Corrupt .tlm file: incomplete length field")

            (length,) = struct.unpack(">I", length_bytes)
            compressed = f.read(length)
            if len(compressed) != length:
                raise ValueError("Corrupt .tlm file: incomplete payload")

            raw = zlib.decompress(compressed)
            records.append(json.loads(raw.decode("utf-8")))

    with open(output_path, "w", encoding="utf-8") as out:
        json.dump(records, out, indent=2)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 tlm_to_json.py <input.tlm> <output.json>")
        sys.exit(1)

    tlm_to_json(sys.argv[1], sys.argv[2])
    print(f"Wrote {sys.argv[2]}")
