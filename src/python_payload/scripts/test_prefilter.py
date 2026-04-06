import subprocess
import sys
from pathlib import Path


def run_prefiltering(img_path: str, output_folder_path: str) -> dict:
    run_path = "/home/argus/Documents/FSW-Payload"
    bin_name = "./bin/RUN_PREFILTER"
    parsed = {
        "passed": False,
        "is_significant": False,
        "dominant_type": "unknown",
        "avg_hue": 0.0,
        "avg_saturation": 0.0,
        "avg_value": 0.0,
        "color_std": 0.0,
        "contrast_std": 0.0,
        "cloudiness": 0,
        "avg_rgb": [0.0, 0.0, 0.0],
    }

    result = subprocess.run(
        [bin_name, img_path, output_folder_path],
        cwd=run_path,
        capture_output=True,
        text=True,
    )

    print("[prefilter] raw stdout:")
    print(result.stdout)
    print("[prefilter] raw stderr:")
    print(result.stderr)

    if result.returncode != 0:
        print(f"[prefilter] Binary exited with code {result.returncode} (continuing to parse stdout)")
        print(f"[prefilter] stderr: {result.stderr.strip()}")

    for line in result.stdout.splitlines():
        line = line.strip()  # handles the leading spaces
        if line.startswith("Passed:"):
            parsed["passed"] = line.split(":", 1)[1].strip().lower() == "yes"
        elif line.startswith("Significant:"):
            parsed["is_significant"] = line.split(":", 1)[1].strip().lower() == "yes"
        elif line.startswith("Dominant Type:"):
            parsed["dominant_type"] = line.split(":", 1)[1].strip().lower()
        elif line.startswith("Hue:"):
            parsed["avg_hue"] = float(line.split(":", 1)[1].split("(")[0].strip())
        elif line.startswith("Saturation:"):
            parsed["avg_saturation"] = float(line.split(":", 1)[1].split("(")[0].strip())
        elif line.startswith("Brightness:"):
            parsed["avg_value"] = float(line.split(":", 1)[1].split("(")[0].strip())
        elif line.startswith("Color Std:"):
            parsed["color_std"] = float(line.split(":", 1)[1].split("(")[0].strip())
        elif line.startswith("Contrast Std:"):
            parsed["contrast_std"] = float(line.split(":", 1)[1].split("(")[0].strip())
        elif line.startswith("Cloudiness:"):
            parsed["cloudiness"] = int(line.split(":", 1)[1].split("%")[0].strip())
        elif line.startswith("Avg RGB:"):
            vals = line.split("(")[1].split(")")[0].split(",")
            parsed["avg_rgb"] = [float(v.strip()) for v in vals]

    return parsed


def print_result(img_path: str, r: dict):
    print(f"\n{'='*50}")
    print(f"Image:       {Path(img_path).name}")
    print(f"{'='*50}")
    print(f"Passed:      {r['passed']}")
    print(f"Significant: {r['is_significant']}")
    print(f"Type:        {r['dominant_type']}")
    print(f"---")
    print(f"Hue:         {r['avg_hue']:.1f}")
    print(f"Saturation:  {r['avg_saturation']:.1f}")
    print(f"Brightness:  {r['avg_value']:.1f}")
    print(f"Color Std:   {r['color_std']:.1f}")
    print(f"Contrast Std:{r['contrast_std']:.1f}")
    print(f"Cloudiness:  {r['cloudiness']}%")
    print(f"Avg RGB:     ({r['avg_rgb'][0]:.0f}, {r['avg_rgb'][1]:.0f}, {r['avg_rgb'][2]:.0f})")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_prefilter.py <image_path>")
        sys.exit(1)

    img_path = sys.argv[1]
    output_folder = "/tmp/prefilter_test_output"

    if not Path(img_path).exists():
        print(f"Error: image not found at {img_path}")
        sys.exit(1)

    result = run_prefiltering(img_path, output_folder)
    print_result(img_path, result)