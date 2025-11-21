#!/usr/bin/env python3
"""
Tilepack HW → Radio File Builder
--------------------------------
Loads an image from disk, resizes to VGA, tiles it, compresses each tile
as JPEG, and builds a single binary "radio file" that the SAT can
downlink to GS.

Output file:
  - image_radio_file.bin: Binary file containing all packets (header + payload)

Packet format:
  Each packet has variable length (≤ 240 bytes) with layout:

  [payload_size_bytes (u16)][page_id (u16)][tile_idx (u16)][frag_idx (u8)]
  [jpeg_fragment...] → 7 bytes header + payload

Constraints:
  - Header: 7 bytes (fixed)
  - Max payload per packet: 233 bytes (to stay within 240 byte limit)
  - Total packet size: header + payload ≤ 240 bytes

The payload_size_bytes field allows proper reconstruction of the JPEG image
by concatenating fragments for each tile based on their tile_idx and frag_idx.
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
INPUT_IMAGE   = "test_image.jpg"  # Path to image on Jetson
OUTPUT_DIR    = "tilepack"                       # Output directory (created if needed)

PAGE_ID       = 1                                # Image/page ID
TARGET_WIDTH  = 640                              # Resize width (VGA)
TARGET_HEIGHT = 480                              # Resize height (VGA)

TILE_W        = 64                               # Tile width   (pixels)
TILE_H        = 32                               # Tile height  (pixels)
JPEG_QUALITY  = 10                               # JPEG quality (1–95)
MAX_PACKET_SIZE = 240                            # Maximum packet size (header + payload)

# ============================================================

# Per-record header (7 bytes)
@dataclass
class PacketHeader:
    payload_size_bytes: int  # u16: actual payload size in this fragment
    page_id: int            # u16: image/page ID
    tile_idx: int           # u16: tile index
    frag_idx: int           # u8: fragment index within tile

    def to_bytes(self) -> bytes:
        # >HHHB = big-endian: u16, u16, u16, u8
        return struct.pack(">HHHB", self.payload_size_bytes, self.page_id, self.tile_idx, self.frag_idx)

    @staticmethod
    def size_bytes() -> int:
        return 7


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
    Fragment compressed tile bytes into a list of Packets.
    Each packet contains header with payload size + metadata for reconstruction.
    
    Each complete packet (header + payload) must not exceed MAX_PACKET_SIZE (240 bytes).
    Header is 7 bytes, so max payload per packet is 233 bytes.
    """
    hdr_sz = PacketHeader.size_bytes()
    max_payload_per_packet = MAX_PACKET_SIZE - hdr_sz  # 240 - 7 = 233 bytes
    
    # Split tile_bytes into chunks that fit within max_payload_per_packet
    frags = [tile_bytes[i : i + max_payload_per_packet] for i in range(0, len(tile_bytes), max_payload_per_packet)]

    packets: List[Packet] = []
    for i, frag in enumerate(frags):
        header = PacketHeader(
            payload_size_bytes=len(frag),
            page_id=page_id,
            tile_idx=tile_idx,
            frag_idx=i
        )
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

    # 7) Build and write the radio file with all packets
    radio_file_path = os.path.join(OUTPUT_DIR, "image_radio_file.bin")
    hdr_sz = PacketHeader.size_bytes()
    max_payload_per_packet = MAX_PACKET_SIZE - hdr_sz

    # Manifest is optional but nice for debugging
    manifest_csv_path = os.path.join(OUTPUT_DIR, "packets_manifest.csv")

    with open(radio_file_path, "wb") as rf, \
         open(manifest_csv_path, "w", newline="") as mf:
        writer = csv.writer(mf)
        writer.writerow(
            [
                "packet_idx",
                "packet_size_bytes",
                "header_size_bytes",
                "payload_size_bytes",
                "page_id",
                "tile_idx",
                "frag_idx",
            ]
        )

        for i, p in enumerate(packets):
            header = p.header.to_bytes()
            payload = p.payload

            # Build the packet: header + payload
            packet = header + payload
            packet_size = len(packet)

            # Verify packet size constraint
            if packet_size > MAX_PACKET_SIZE:
                raise RuntimeError(
                    f"Packet {i} exceeds MAX_PACKET_SIZE={MAX_PACKET_SIZE} (got {packet_size} bytes). "
                    "This should not happen - check packetize_tile logic."
                )

            # Write packet to radio file
            rf.write(packet)

            # Manifest row for debugging
            writer.writerow(
                [
                    i,
                    packet_size,
                    hdr_sz,
                    len(payload),
                    p.header.page_id,
                    p.header.tile_idx,
                    p.header.frag_idx,
                ]
            )

    # 8) Print summary
    comp_total = sum(len(c) for c in comp_tiles)
    total_file_bytes = sum(len(p.header.to_bytes()) + len(p.payload) for p in packets)
    avg_packet_size = total_file_bytes / len(packets) if packets else 0
    max_payload_per_packet = MAX_PACKET_SIZE - hdr_sz

    print("=== TILEPACK HW → RADIO FILE BUILDER ===")
    print(f"Input:            {INPUT_IMAGE}  resized={TARGET_WIDTH}x{TARGET_HEIGHT}")
    print(f"Tiling:           {tiles_x} x {tiles_y}  (tile {TILE_W}x{TILE_H}), tiles={len(tiles)}")
    print(f"JPEG quality:     {JPEG_QUALITY}")
    print(f"Compressed total: {comp_total/1024:.1f} kB")
    print(f"Header bytes:     {hdr_sz}")
    print(f"Max packet size:  {MAX_PACKET_SIZE} bytes (header {hdr_sz} + max payload {max_payload_per_packet})")
    print(f"Avg packet size:  {avg_packet_size:.1f} bytes")
    print(f"Total packets:    {len(packets)}")
    print(f"Total file size:  {total_file_bytes/1024:.1f} kB")
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