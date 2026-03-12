#!/usr/bin/env python3

import argparse
import json
import struct
import zlib

MAGIC = b"TLM1"


def iter_records(path: str):
    with open(path, "rb") as f:
        header = f.read(4)
        if header != MAGIC:
            raise ValueError(f"Invalid telemetry file header: expected {MAGIC!r}, got {header!r}")

        while True:
            length_bytes = f.read(4)
            if not length_bytes:
                return
            if len(length_bytes) != 4:
                raise ValueError("Corrupted telemetry file: incomplete record length")

            (length,) = struct.unpack(">I", length_bytes)
            payload = f.read(length)
            if len(payload) != length:
                raise ValueError("Corrupted telemetry file: incomplete payload")

            raw = zlib.decompress(payload)
            yield json.loads(raw.decode("utf-8"))


def main():
    parser = argparse.ArgumentParser(description="Convert payload telemetry binary log to JSON")
    parser.add_argument("input", help="Input .tlm file")
    parser.add_argument("output", help="Output .json file")
    parser.add_argument("--ndjson", action="store_true", help="Write newline-delimited JSON")
    args = parser.parse_args()

    records = iter_records(args.input)

    with open(args.output, "w", encoding="utf-8") as out:
        if args.ndjson:
            for record in records:
                out.write(json.dumps(record, separators=(",", ":")) + "\n")
        else:
            json.dump(list(records), out, indent=2)


if __name__ == "__main__":
    main()
