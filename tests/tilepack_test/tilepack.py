#!/usr/bin/env python3
"""
Tilepack HW → Radio File Builder
--------------------------------
Loads an image from disk, resizes to VGA, tiles it, compresses each tile
as JPEG, and builds a single binary "radio file" that the SAT can
downlink to GS.

Radio file format (image_radio_file.bin):

  record[0] | record[1] | ... | record[N-1]

Each record has fixed size RECORD_BYTES and layout:

  [page_id (u16)][tile_idx (u16)][frag_idx (u8)][frag_cnt (u8)]
  [jpeg_fragment...][0x00 padding...] → RECORD_BYTES bytes total

RECORD_BYTES MUST MATCH the _FILE_PKTSIZE constant on the SATELLITE_RADIO
side (currently 240).
"""

import io
import os
import math
import csv
import struct
import sys
from dataclasses import dataclass
from typing import List, Tuple
from PIL import Image

# Settings
INPUT_IMAGE   = "/home/argus/RidgeRun_test.jpg"  # Path to image on Jetson
OUTPUT_DIR    = "tilepack"                       # Output directory (created if needed)

PAGE_ID       = 1                                # Image/page ID
TARGET_WIDTH  = 640                              # Resize width (VGA)
TARGET_HEIGHT = 480                              # Resize height (VGA)

TILE_W        = 64                               # Tile width   (pixels)
TILE_H        = 32                               # Tile height  (pixels)
JPEG_QUALITY  = 10                               # JPEG quality (1–95)
RECORD_BYTES  = 240                              # MUST MATCH _FILE_PKTSIZE on the SAT mainboard

# ============================================================

# Per-record header (6 bytes)
@dataclass
class PacketHeader:
    page_id: int
    tile_idx: int
    frag_idx: int
    frag_cnt: int

    def to_bytes(self) -> bytes:
        # >HHBB = big-endian: u16, u16, u8, u8
        return struct.pack(">HHBB", self.page_id, self.tile_idx, self.frag_idx, self.frag_cnt)

    @staticmethod
    def size_bytes() -> int:
        return 6


@dataclass
class Packet:
    header: PacketHeader
    payload: bytes

# Helpers
def image_to_tiles(img: Image.Image, tile_w: int, tile_h: int) -> Tuple[List[Image.Image], int, int]:
    """
    Split image into tiles of size (tile_w x tile_h).
    Pads with black if the image is not an exact multiple.
    Returns (tiles_list, tiles_x, tiles_y).
    """
    w, h = img.size
    tiles_x = math.ceil(w / tile_w)
    tiles_y = math.ceil(h / tile_h)

    # Pad if needed
    if (w % tile_w) != 0 or (h % tile_h) != 0:
        pad_w = tiles_x * tile_w
        pad_h = tiles_y * tile_h
        canvas = Image.new("RGB", (pad_w, pad_h), (0, 0, 0))
        canvas.paste(img, (0, 0))
        img = canvas

    tiles: List[Image.Image] = []
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            box = (tx * tile_w, ty * tile_h, (tx + 1) * tile_w, (ty + 1) * tile_h)
            tiles.append(img.crop(box))

    return tiles, tiles_x, tiles_y


def compress_tile_jpeg(tile: Image.Image, jpeg_quality: int) -> bytes:
    """
    Compress a single tile as JPEG and return raw bytes.
    """
    bio = io.BytesIO()
    tile.save(bio, format="JPEG", quality=jpeg_quality, optimize=True)
    return bio.getvalue()


def packetize_tile(page_id: int, tile_idx: int, tile_bytes: bytes) -> List[Packet]:
    """
    Fragment compressed tile bytes into a list of Packets, where each Packet
    will later become one fixed-size record in the radio file.

    payload size per Packet = RECORD_BYTES - header_size (6 bytes).
    """
    hdr_sz = PacketHeader.size_bytes()
    max_payload = max(1, RECORD_BYTES - hdr_sz)

    # Split tile_bytes into chunks that fit into one record's payload space
    frags = [tile_bytes[i : i + max_payload] for i in range(0, len(tile_bytes), max_payload)]
    frag_cnt = len(frags)

    packets: List[Packet] = []
    for i, frag in enumerate(frags):
        header = PacketHeader(page_id=page_id, tile_idx=tile_idx, frag_idx=i, frag_cnt=frag_cnt)
        packets.append(Packet(header=header, payload=frag))

    return packets


# ---------------------------------------------------------------------------------------------
def main():
    # 0) Check input
    if not os.path.exists(INPUT_IMAGE):
        print(f"[ERROR] Input image not found: {INPUT_IMAGE}", file=sys.stderr)
        sys.exit(1)

    # 1) Prepare output dirs
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tiles_dir = os.path.join(OUTPUT_DIR, "tiles_jpeg")
    os.makedirs(tiles_dir, exist_ok=True)

    # 2) Load and resize
    img = Image.open(INPUT_IMAGE).convert("RGB").resize((TARGET_WIDTH, TARGET_HEIGHT))
    src_png = os.path.join(OUTPUT_DIR, "input_resized.png")
    img.save(src_png)

    # 3) Tile the image
    tiles, tiles_x, tiles_y = image_to_tiles(img, TILE_W, TILE_H)

    # 4) JPEG-compress each tile
    comp_tiles: List[bytes] = []
    for idx, t in enumerate(tiles):
        b = compress_tile_jpeg(t, JPEG_QUALITY)
        comp_tiles.append(b)

        # Save tile JPEG for sanity checks
        with open(os.path.join(tiles_dir, f"tile_{idx:05d}.jpg"), "wb") as f:
            f.write(b)

    # 5) Packetize all tiles into Packet objects (NOT yet fixed-size records)
    packets: List[Packet] = []
    for idx, tb in enumerate(comp_tiles):
        packets.extend(packetize_tile(PAGE_ID, idx, tb))

    # 6) Write image-wide metadata (useful but not required by comms)
    meta_path = os.path.join(OUTPUT_DIR, "image_meta.bin")
    meta_blob = struct.pack(
        ">HHHHHHHB",
        PAGE_ID,
        tiles_x, tiles_y,
        TILE_W, TILE_H,
        TARGET_WIDTH, TARGET_HEIGHT,
        max(0, min(255, JPEG_QUALITY)),
    )
    with open(meta_path, "wb") as mf:
        mf.write(meta_blob)

    # 7) Build and write the RADIO FILE with fixed-size records
    radio_file_path = os.path.join(OUTPUT_DIR, "image_radio_file.bin")
    hdr_sz = PacketHeader.size_bytes()

    # Manifest is optional but nice for debugging
    manifest_csv_path = os.path.join(OUTPUT_DIR, "packets_manifest.csv")

    with open(radio_file_path, "wb") as rf, open(manifest_csv_path, "w", newline="") as mf:
        writer = csv.writer(mf)
        writer.writerow(
            [
                "record_idx",
                "record_size_bytes",
                "header_size_bytes",
                "payload_size_bytes",
                "page_id",
                "tile_idx",
                "frag_idx",
                "frag_cnt",
            ]
        )

        for i, p in enumerate(packets):
            header = p.header.to_bytes()
            payload = p.payload

            # Build the record: header + payload
            record = bytearray(header + payload)

            if len(record) > RECORD_BYTES:
                raise RuntimeError(
                    f"Record {i} exceeds RECORD_BYTES={RECORD_BYTES} (got {len(record)} bytes). "
                    "Increase RECORD_BYTES or reduce JPEG_QUALITY / tile size."
                )

            # Pad with zeros up to RECORD_BYTES
            pad_len = RECORD_BYTES - len(record)
            record += b"\x00" * pad_len

            # Write this fixed-size record to the radio file
            rf.write(record)

            # Manifest row for debugging
            writer.writerow(
                [
                    i,
                    RECORD_BYTES,
                    hdr_sz,
                    len(payload),
                    p.header.page_id,
                    p.header.tile_idx,
                    p.header.frag_idx,
                    p.header.frag_cnt,
                ]
            )

    # 8) Print summary
    comp_total = sum(len(c) for c in comp_tiles)
    max_payload = max(1, RECORD_BYTES - hdr_sz)

    print("=== TILEPACK HW → RADIO FILE BUILDER ===")
    print(f"Input:            {INPUT_IMAGE}  resized={TARGET_WIDTH}x{TARGET_HEIGHT}")
    print(f"Tiling:           {tiles_x} x {tiles_y}  (tile {TILE_W}x{TILE_H}), tiles={len(tiles)}")
    print(f"JPEG quality:     {JPEG_QUALITY}")
    print(f"Compressed total: {comp_total/1024:.1f} kB")
    print(f"Header bytes:     {hdr_sz}")
    print(f"Record bytes:     {RECORD_BYTES} (max payload per record ≈ {max_payload})")
    print(f"Total records:    {len(packets)}")
    print(f"Saved resized:    {src_png}")
    print(f"Saved tiles:      {tiles_dir}/tile_00000.jpg ..")
    print(f"Saved meta:       {meta_path}")
    print(f"Saved radio file: {radio_file_path}")
    print(f"Manifest:         {manifest_csv_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)