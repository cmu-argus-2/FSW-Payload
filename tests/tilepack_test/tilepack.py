"""
Tilepack HW 
-------------------------------------------------------
Loads an image from disk, resizes to VGA, tiles it, compresses each tile
as JPEG, saves them, and packetizes them with a 6-byte header.
"""

import io, os, math, csv, struct, sys
from dataclasses import dataclass
from typing import List, Tuple
from PIL import Image

# SETTINGS
INPUT_IMAGE   = "/home/argus/RidgeRun_test.jpg"  # Path to image on Jetson
OUTPUT_DIR    = "tilepack"                       # Output directory
PAGE_ID       = 1                                # Image/page ID
TARGET_WIDTH  = 640                              # Resize width (VGA)
TARGET_HEIGHT = 480                              # Resize height (VGA)
TILE_W        = 64                               # Tile width
TILE_H        = 32                               # Tile height
JPEG_QUALITY  = 10                               # JPEG quality (1–95)
PACKET_BYTES  = 194                              # Packet size (header + payload)


#Per-packet header (6 bytes)
@dataclass
class PacketHeader:
    page_id: int
    tile_idx: int
    frag_idx: int
    frag_cnt: int

    def to_bytes(self) -> bytes:
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
    w, h = img.size
    tiles_x = math.ceil(w / tile_w)
    tiles_y = math.ceil(h / tile_h)

    if (w % tile_w) != 0 or (h % tile_h) != 0:
        pad_w = tiles_x * tile_w
        pad_h = tiles_y * tile_h
        canvas = Image.new("RGB", (pad_w, pad_h), (0, 0, 0))
        canvas.paste(img, (0, 0))
        img = canvas

    tiles = []
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            box = (tx*tile_w, ty*tile_h, (tx+1)*tile_w, (ty+1)*tile_h)
            tiles.append(img.crop(box))
    return tiles, tiles_x, tiles_y

def compress_tile_jpeg(tile: Image.Image, jpeg_quality: int) -> bytes:
    bio = io.BytesIO()
    tile.save(bio, format="JPEG", quality=jpeg_quality, optimize=True)
    return bio.getvalue()

def packetize_tile(page_id: int, tile_idx: int, tile_bytes: bytes, packet_bytes: int) -> List[Packet]:
    hdr_sz = PacketHeader.size_bytes()
    payload_per_packet = max(1, packet_bytes - hdr_sz)
    frags = [tile_bytes[i:i+payload_per_packet] for i in range(0, len(tile_bytes), payload_per_packet)]
    packets = [Packet(PacketHeader(page_id, tile_idx, i, len(frags)), frag) for i, frag in enumerate(frags)]
    return packets


# ---------------------------------------------------------------------------------------------

def main():
    if not os.path.exists(INPUT_IMAGE):
        print(f"[ERROR] Input image not found: {INPUT_IMAGE}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tiles_dir = os.path.join(OUTPUT_DIR, "tiles_jpeg")
    os.makedirs(tiles_dir, exist_ok=True)

    # 1) Load and resize
    img = Image.open(INPUT_IMAGE).convert("RGB").resize((TARGET_WIDTH, TARGET_HEIGHT))
    src_png = os.path.join(OUTPUT_DIR, "input_resized.png")
    img.save(src_png)

    # 2) Tile
    tiles, tiles_x, tiles_y = image_to_tiles(img, TILE_W, TILE_H)

    # 3) JPEG-compress each tile
    comp_tiles = []
    for idx, t in enumerate(tiles):
        b = compress_tile_jpeg(t, JPEG_QUALITY)
        comp_tiles.append(b)
        with open(os.path.join(tiles_dir, f"tile_{idx:05d}.jpg"), "wb") as f:
            f.write(b)

    # 4) Packetize all tiles
    packets = []
    for idx, tb in enumerate(comp_tiles):
        packets.extend(packetize_tile(PAGE_ID, idx, tb, PACKET_BYTES))

    # 5) Write image-wide metadata once
    meta_path = os.path.join(OUTPUT_DIR, "image_meta.bin")
    meta_blob = struct.pack(
        ">HHHHHHHB",
        PAGE_ID,
        tiles_x, tiles_y,
        TILE_W, TILE_H,
        TARGET_WIDTH, TARGET_HEIGHT,
        max(0, min(255, JPEG_QUALITY))
    )
    with open(meta_path, "wb") as mf:
        mf.write(meta_blob)

    # 6) Write packets.bin and manifest
    packets_bin_path = os.path.join(OUTPUT_DIR, "packets.bin")
    manifest_csv_path = os.path.join(OUTPUT_DIR, "packets_manifest.csv")
    hdr_sz = PacketHeader.size_bytes()

    with open(packets_bin_path, "wb") as pf, open(manifest_csv_path, "w", newline="") as mf:
        writer = csv.writer(mf)
        writer.writerow(["packet_idx", "file_offset", "header_size_bytes",
                         "payload_size_bytes", "page_id", "tile_idx",
                         "frag_idx", "frag_cnt"])
        offset = 0
        for i, p in enumerate(packets):
            hb = p.header.to_bytes()
            pb = p.payload
            pf.write(hb)
            pf.write(pb)
            writer.writerow([i, offset, hdr_sz, len(pb),
                             p.header.page_id, p.header.tile_idx,
                             p.header.frag_idx, p.header.frag_cnt])
            offset += len(hb) + len(pb)

    # 7) Print summary
    payload_pp = max(1, PACKET_BYTES - hdr_sz)
    comp_total = sum(len(c) for c in comp_tiles)

    print("=== TILEPACK HW (fixed config) ===")
    print(f"Input:            {INPUT_IMAGE}  resized={TARGET_WIDTH}x{TARGET_HEIGHT}")
    print(f"Tiling:           {tiles_x} x {tiles_y}  (tile {TILE_W}x{TILE_H}), tiles={len(tiles)}")
    print(f"JPEG quality:     {JPEG_QUALITY}")
    print(f"Compressed total: {comp_total/1024:.1f} kB")
    print(f"Header bytes:     {hdr_sz}")
    print(f"Packet bytes:     {PACKET_BYTES} (payload≈{payload_pp})")
    print(f"Total packets:    {len(packets)}")
    print(f"Saved resized:    {src_png}")
    print(f"Saved tiles:      {tiles_dir}/tile_00000.jpg ..")
    print(f"Saved meta:       {meta_path}")
    print(f"Saved packets:    {packets_bin_path}")
    print(f"Manifest:         {manifest_csv_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


