"""
calibration.py
==============
Ground calibration tool for flight cameras on NVIDIA Jetson.

Determines fixed exposure / gain / white-balance settings suitable for
pinning in config/config.toml before a mission.  Cannot be run in orbit —
it requires a scene representative of the intended imaging target
(e.g. a sunlit Earth-analogue target or outdoor scene).

Workflow
--------
1. Convergence  – open camera with AE/AWB active; wait until mean luminance
                  stabilises (AE has settled).
2. Exposure sweep – reopen with each candidate exposure time pinned
                  (min == max) at gain 1×; record luminance, saturation %,
                  and per-channel means.  Pick the candidate whose score is
                  best relative to the target luminance.
3. Gain sweep   – at the chosen exposure, reopen with each candidate gain
                  pinned; record luminance, saturation %, and per-channel
                  means. Pick the gain whose score best matches the target.
4. WB sweep     – at the chosen exposure/gain, open with each white-balance mode
                  and record R/G/B channel means.  Flags the most neutral
                  mode as the recommendation (user should verify visually).
5. Report       – write calibration/cam{N}/calibration_result.toml with a
                  [camera-isp] block ready to paste into config/config.toml,
                  and save a reference JPEG for each sweep step.

Usage
-----
    python3 scripts/calibration.py
    python3 scripts/calibration.py --sensor-id 1 --target-luminance 140
    python3 scripts/calibration.py --help
"""

from __future__ import annotations

import argparse
import datetime
import logging
import sys
import time
from pathlib import Path

import subprocess

import cv2
import numpy as np

# Allow running from the project root without installation
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "python_payload"))
from camera_driver import (  # noqa: E402
    AeAntibandingMode,
    EdgeEnhancementMode,
    JetsonCamera,
    NoiseReductionMode,
    WBMode,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("calibration")

# ---------------------------------------------------------------------------
# Tuneable constants
# ---------------------------------------------------------------------------
TARGET_LUMINANCE    = 128    # 0–255 ideal mean scene brightness
MAX_SATURATION_PCT  = 1.0    # % of pixels > 250 — if exceeded exposure is too long
MAX_DARK_PCT        = 5.0    # % of pixels < 5   — if too high exposure is too short

CONVERGENCE_FRAMES    = 25   # max frames to try before giving up on AE convergence
CONVERGENCE_STABILITY = 5    # number of consecutive frames that must be stable
CONVERGENCE_THRESHOLD = 3.0  # max luminance std-dev (out of 255) to call "stable"

SWEEP_FRAMES  = 8   # frames captured per candidate; median-closest one is kept
WARMUP_FRAMES = 6   # frames discarded after opening the pipeline (sensor settling)

WB_SETTLE_MIN_SECONDS = 2.0    # wait at least this long after a WB config change
WB_SETTLE_TIMEOUT_SECONDS = 8.0 # keep checking until this deadline
WB_SETTLE_STABILITY_FRAMES = 5  # consecutive frames used for the stability test
WB_SETTLE_RATIO_THRESHOLD = 0.015  # max std-dev for R/G and B/G ratios
WB_SETTLE_LUM_THRESHOLD = 3.0      # max luminance std-dev (out of 255)

# Exposure candidates in nanoseconds — covers ~0.5 ms to 32 ms
EXPOSURE_CANDIDATES_NS: list[int] = [
    500_000,
    1_000_000,
    2_000_000,
    4_000_000,
    8_000_000,
    16_000_000,
    32_000_000,
]

# Gain candidates — nvarguscamerasrc supports 1.0 to 16.0 analog gain.
GAIN_CANDIDATES: list[float] = [
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    8.0,
    12.0,
    16.0,
]

# White-balance modes to compare (display name, integer value)
WB_CANDIDATES: list[tuple[str, int]] = [
    ("OFF",             WBMode.OFF),
    ("AUTO",            WBMode.AUTO),
    ("INCANDESCENT",    WBMode.INCANDESCENT),
    ("DAYLIGHT",        WBMode.DAYLIGHT),
    ("CLOUDY_DAYLIGHT", WBMode.CLOUDY_DAYLIGHT),
    ("SHADE",           WBMode.SHADE),
]

# ---------------------------------------------------------------------------
# Frame-analysis helpers
# ---------------------------------------------------------------------------

def _mean_luminance(frame: np.ndarray) -> float:
    return float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())


def _saturation_pct(frame: np.ndarray, hi: int = 250) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float((gray > hi).sum()) / gray.size * 100.0


def _dark_pct(frame: np.ndarray, lo: int = 5) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float((gray < lo).sum()) / gray.size * 100.0


def _channel_means(frame: np.ndarray) -> tuple[float, float, float]:
    b, g, r = cv2.split(frame)
    return float(r.mean()), float(g.mean()), float(b.mean())


def _wb_metrics(frame: np.ndarray) -> dict:
    r, g, b = _channel_means(frame)
    return dict(
        frame=frame,
        lum=_mean_luminance(frame),
        r=r,
        g=g,
        b=b,
        rg_ratio=r / g if g > 0 else 0.0,
        bg_ratio=b / g if g > 0 else 0.0,
    )


def _exposure_score(lum: float, sat_pct: float, dark_pct: float, target: float) -> float:
    """Lower is better.  Heavy penalty for blown highlights; lighter for shadow crush."""
    lum_penalty  = abs(lum - target)
    sat_penalty  = max(0.0, sat_pct  - 0.2) * 50.0  # steep — blown pixels are bad
    dark_penalty = max(0.0, dark_pct - 2.0) *  5.0
    return lum_penalty + sat_penalty + dark_penalty


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def _warmup(cam: JetsonCamera, n: int = WARMUP_FRAMES) -> None:
    for _ in range(n):
        try:
            cam.capture()
        except RuntimeError:
            pass
        time.sleep(0.04)


def _median_frame(cam: JetsonCamera, n: int = SWEEP_FRAMES) -> np.ndarray:
    """Capture n frames; return the one closest to the median luminance."""
    frames, lums = [], []
    for _ in range(n):
        f = cam.capture()
        frames.append(f)
        lums.append(_mean_luminance(f))
    med = float(np.median(lums))
    best = int(np.argmin([abs(l - med) for l in lums]))
    return frames[best]


def _wait_for_wb_settle(
    cam: JetsonCamera,
    wb_name: str,
    *,
    min_seconds: float = WB_SETTLE_MIN_SECONDS,
    timeout_seconds: float = WB_SETTLE_TIMEOUT_SECONDS,
    stability_frames: int = WB_SETTLE_STABILITY_FRAMES,
    ratio_threshold: float = WB_SETTLE_RATIO_THRESHOLD,
    lum_threshold: float = WB_SETTLE_LUM_THRESHOLD,
) -> tuple[np.ndarray, dict]:
    """
    Capture frames until the active WB mode appears stable.

    WB settling is judged from the frame stream itself: R/G, B/G, and mean
    luminance must all have low variation over the most recent frames.  The
    minimum wait avoids accepting the first few buffered/partially-updated
    frames as "settled" when a new nvarguscamerasrc config is applied.
    """
    if stability_frames <= 0:
        raise ValueError("stability_frames must be positive")
    if min_seconds < 0:
        raise ValueError("min_seconds must be non-negative")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if timeout_seconds < min_seconds:
        raise ValueError("timeout_seconds must be >= min_seconds")

    history: list[dict] = []
    start = time.monotonic()
    settled = False
    settle_reason = "timeout"

    while True:
        metric = _wb_metrics(cam.capture())
        metric["elapsed_s"] = time.monotonic() - start
        history.append(metric)

        if len(history) >= stability_frames:
            window = history[-stability_frames:]
            rg_std = float(np.std([m["rg_ratio"] for m in window]))
            bg_std = float(np.std([m["bg_ratio"] for m in window]))
            lum_std = float(np.std([m["lum"] for m in window]))
            waited_long_enough = metric["elapsed_s"] >= min_seconds

            if (
                waited_long_enough
                and rg_std <= ratio_threshold
                and bg_std <= ratio_threshold
                and lum_std <= lum_threshold
            ):
                settled = True
                settle_reason = "stable"
                break

        if metric["elapsed_s"] >= timeout_seconds:
            break

    final_window = history[-min(stability_frames, len(history)):]
    lums = [m["lum"] for m in final_window]
    med = float(np.median(lums))
    best_idx = int(np.argmin([abs(l - med) for l in lums]))
    selected = final_window[best_idx]

    settle_info = dict(
        settled=settled,
        settle_reason=settle_reason,
        settle_s=history[-1]["elapsed_s"],
        settle_frames=len(history),
        rg_std=float(np.std([m["rg_ratio"] for m in final_window])),
        bg_std=float(np.std([m["bg_ratio"] for m in final_window])),
        lum_std=float(np.std([m["lum"] for m in final_window])),
    )

    if settled:
        log.info(
            "    WBMode.%s settled after %.2fs / %d frames "
            "(std R/G=%.4f, B/G=%.4f, lum=%.2f)",
            wb_name,
            settle_info["settle_s"],
            settle_info["settle_frames"],
            settle_info["rg_std"],
            settle_info["bg_std"],
            settle_info["lum_std"],
        )
    else:
        log.warning(
            "    WBMode.%s did not settle by %.2fs / %d frames; using median frame "
            "(std R/G=%.4f, B/G=%.4f, lum=%.2f)",
            wb_name,
            settle_info["settle_s"],
            settle_info["settle_frames"],
            settle_info["rg_std"],
            settle_info["bg_std"],
            settle_info["lum_std"],
        )

    return selected["frame"], settle_info


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

def preflight_check(sensor_id: int) -> bool:
    """
    Run sanity checks before attempting the full calibration pipeline.
    Returns True if everything looks OK; False if a blocking problem was found.
    Logs a diagnostic message for each check so the caller can see what failed.
    """
    ok = True

    # 1. OpenCV GStreamer support
    build_info = cv2.getBuildInformation()
    gst_ok = "GStreamer:                   YES" in build_info
    if gst_ok:
        log.info("  [OK] OpenCV was built with GStreamer support")
    else:
        log.error(
            "  [FAIL] OpenCV does NOT have GStreamer support.\n"
            "         You need an OpenCV build that includes GStreamer.\n"
            "         On Jetson: sudo apt install python3-opencv  or rebuild from source with -DWITH_GSTREAMER=ON"
        )
        ok = False

    # 2. Device node exists
    dev_path = f"/dev/video{sensor_id}"
    if Path(dev_path).exists():
        log.info("  [OK] Device node %s exists", dev_path)
    else:
        # v4l2 devices may be numbered differently from sensor IDs on Jetson
        all_video = sorted(Path("/dev").glob("video*"))
        log.error(
            "  [FAIL] Device node %s not found.\n"
            "         Available video devices: %s\n"
            "         Note: on Jetson, sensor-id maps to the CSI port, not /dev/videoN directly.",
            dev_path,
            [str(p) for p in all_video] or "none",
        )
        ok = False

    # 3. nvargus-daemon running (Jetson-specific)
    try:
        out = subprocess.run(
            ["pgrep", "-x", "nvargus-daemon"],
            capture_output=True, text=True,
        )
        if out.returncode == 0:
            log.info("  [OK] nvargus-daemon is running (pid %s)", out.stdout.strip())
        else:
            log.error(
                "  [FAIL] nvargus-daemon is not running.\n"
                "         Start it with: sudo systemctl start nvargus-daemon\n"
                "         Or: sudo nvargus-daemon &"
            )
            ok = False
    except FileNotFoundError:
        log.warning("  [SKIP] pgrep not available — cannot check nvargus-daemon")

    # 4. No other process has the camera open
    try:
        out = subprocess.run(
            ["fuser", f"/dev/video{sensor_id}"],
            capture_output=True, text=True,
        )
        if out.stdout.strip():
            log.error(
                "  [FAIL] /dev/video%d is already held by process(es): %s\n"
                "         Kill the other process before running calibration.",
                sensor_id, out.stdout.strip(),
            )
            ok = False
        else:
            log.info("  [OK] /dev/video%d is not held by another process", sensor_id)
    except FileNotFoundError:
        log.warning("  [SKIP] fuser not available — cannot check for competing processes")

    # 5. Minimal pipeline smoke-test at low resolution (faster to fail/succeed)
    test_pipeline = (
        f"nvarguscamerasrc sensor-id={sensor_id} num-buffers=1 "
        f"! video/x-raw(memory:NVMM),width=640,height=480,framerate=10/1,format=NV12 "
        f"! nvvidconv ! video/x-raw,format=BGRx ! videoconvert "
        f"! video/x-raw,format=BGR ! appsink drop=true max-buffers=1"
    )
    log.info("  Testing minimal pipeline: %s", test_pipeline)
    cap = cv2.VideoCapture(test_pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        log.error(
            "  [FAIL] Minimal pipeline could not be opened.\n"
            "         This usually means nvarguscamerasrc is unavailable, the sensor\n"
            "         is not detected, or there is a GStreamer element missing.\n"
            "         Run this to debug GStreamer directly:\n"
            "           gst-launch-1.0 nvarguscamerasrc sensor-id=%d num-buffers=1 "
            "! video/x-raw(memory:NVMM),width=640,height=480,framerate=10/1,format=NV12 "
            "! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! fakesink",
            sensor_id,
        )
        ok = False
    else:
        ret, frame = cap.read()
        if ret and frame is not None:
            log.info(
                "  [OK] Minimal pipeline produced a %dx%d frame",
                frame.shape[1], frame.shape[0],
            )
        else:
            log.error(
                "  [FAIL] Pipeline opened but produced no frame.\n"
                "         The sensor may be initialising — try again in a few seconds."
            )
            ok = False
        cap.release()

    return ok


# ---------------------------------------------------------------------------
# Phase 1 – AE/AWB convergence
# ---------------------------------------------------------------------------

def wait_for_convergence(
    sensor_id: int,
    width: int,
    height: int,
    fps: int,
) -> float:
    """Open camera in full-auto mode; return mean luminance once AE stabilises."""
    log.info("Phase 1 — AE convergence  (sensor %d)", sensor_id)

    cam = JetsonCamera(
        sensor_id=sensor_id,
        width=width,
        height=height,
        fps=fps,
        wbmode=WBMode.AUTO,
        aelock=False,
        awblock=False,
        aeantibanding=AeAntibandingMode.AUTO,
    )
    log.info("  Pipeline: %s", cam.build_pipeline())
    try:
        cam.open()
    except RuntimeError as exc:
        raise RuntimeError(
            f"Failed to open sensor {sensor_id} at {width}x{height} for convergence phase.\n"
            f"  Inner error: {exc}\n"
            f"  If the minimal pipeline smoke-test passed but this failed, the sensor may\n"
            f"  not support {width}x{height} — try --width 1920 --height 1080."
        ) from exc
    _warmup(cam)

    history: list[float] = []

    for i in range(CONVERGENCE_FRAMES + CONVERGENCE_STABILITY):
        frame = cam.capture()
        lum   = _mean_luminance(frame)
        history.append(lum)

        if len(history) >= CONVERGENCE_STABILITY:
            window = history[-CONVERGENCE_STABILITY:]
            if float(np.std(window)) < CONVERGENCE_THRESHOLD:
                settled = float(np.mean(window))
                log.info(
                    "  AE converged after %d frames — settled luminance %.1f/255",
                    i + 1, settled,
                )
                cam.close()
                return settled

        log.info("  frame %2d  lum=%.1f", i, lum)

    cam.close()
    settled = float(np.mean(history[-CONVERGENCE_STABILITY:]))
    log.warning(
        "AE did not converge in %d frames; continuing with mean=%.1f",
        CONVERGENCE_FRAMES + CONVERGENCE_STABILITY, settled,
    )
    return settled


# ---------------------------------------------------------------------------
# Phase 2 – Exposure sweep
# ---------------------------------------------------------------------------

def exposure_sweep(
    sensor_id: int,
    width: int,
    height: int,
    fps: int,
    output_dir: Path,
    target_lum: float,
) -> tuple[dict, list[dict]]:
    """Test each candidate exposure time (gain pinned at 1×); return best result."""
    log.info("Phase 2 — Exposure sweep  (sensor %d)", sensor_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []

    for exp_ns in EXPOSURE_CANDIDATES_NS:
        exp_us = exp_ns // 1000
        log.info("  %6d µs …", exp_us)

        cam = JetsonCamera(
            sensor_id=sensor_id,
            width=width,
            height=height,
            fps=fps,
            wbmode=WBMode.AUTO,    # keep AWB free so only exposure changes
            aelock=True,
            awblock=False,
            exposuretimerange=(exp_ns, exp_ns),  # pin: min == max
            gainrange=(1.0, 1.0),                # pin analog gain
            ispdigitalgainrange=(1.0, 1.0),      # pin ISP digital gain
        )

        try:
            cam.open()
        except RuntimeError as exc:
            log.warning("  Skipped %d µs — could not open camera: %s", exp_us, exc)
            continue

        _warmup(cam)
        frame = _median_frame(cam)
        cam.close()

        lum   = _mean_luminance(frame)
        sat   = _saturation_pct(frame)
        dark  = _dark_pct(frame)
        r, g, b = _channel_means(frame)
        score = _exposure_score(lum, sat, dark, target_lum)

        log.info(
            "    lum=%5.1f  sat=%5.2f%%  dark=%5.2f%%  score=%6.1f",
            lum, sat, dark, score,
        )

        cv2.imwrite(str(output_dir / f"exposure_{exp_us:07d}us.jpg"), frame)

        results.append(
            dict(
                exp_ns=exp_ns, lum=lum, sat_pct=sat,
                dark_pct=dark, r=r, g=g, b=b, score=score,
            )
        )

    if not results:
        raise RuntimeError(
            "Exposure sweep produced no results.  "
            "Is the camera connected and nvarguscamerasrc available?"
        )

    best = min(results, key=lambda r: r["score"])
    log.info(
        "Best exposure: %d µs  lum=%.1f  sat=%.2f%%  score=%.1f",
        best["exp_ns"] // 1000, best["lum"], best["sat_pct"], best["score"],
    )
    return best, results


# ---------------------------------------------------------------------------
# Phase 3 – Gain sweep
# ---------------------------------------------------------------------------

def gain_sweep(
    sensor_id: int,
    width: int,
    height: int,
    fps: int,
    exp_ns: int,
    output_dir: Path,
    target_lum: float,
) -> tuple[dict, list[dict]]:
    """Test candidate analog gains at the pinned exposure; return best result."""
    log.info(
        "Phase 3 — Gain sweep  (sensor %d, exposure %d µs)",
        sensor_id, exp_ns // 1000,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []

    for gain in GAIN_CANDIDATES:
        log.info("  gain %.1fx …", gain)

        cam = JetsonCamera(
            sensor_id=sensor_id,
            width=width,
            height=height,
            fps=fps,
            wbmode=WBMode.OFF,     # isolate gain; color balance is evaluated next
            aelock=True,
            awblock=True,
            exposuretimerange=(exp_ns, exp_ns),
            gainrange=(gain, gain),
            ispdigitalgainrange=(1.0, 1.0),
        )

        try:
            cam.open()
        except RuntimeError as exc:
            log.warning("  Skipped gain %.1fx — could not open camera: %s", gain, exc)
            continue

        try:
            _warmup(cam)
            frame = _median_frame(cam)
        finally:
            cam.close()

        lum   = _mean_luminance(frame)
        sat   = _saturation_pct(frame)
        dark  = _dark_pct(frame)
        r, g, b = _channel_means(frame)
        score = _exposure_score(lum, sat, dark, target_lum)

        log.info(
            "    lum=%5.1f  sat=%5.2f%%  dark=%5.2f%%  score=%6.1f",
            lum, sat, dark, score,
        )

        gain_tag = str(gain).replace(".", "p")
        cv2.imwrite(str(output_dir / f"gain_{gain_tag}x.jpg"), frame)

        results.append(
            dict(
                gain=gain, lum=lum, sat_pct=sat,
                dark_pct=dark, r=r, g=g, b=b, score=score,
            )
        )

    if not results:
        raise RuntimeError(
            "Gain sweep produced no results.  "
            "Is the camera connected and nvarguscamerasrc available?"
        )

    best = min(results, key=lambda r: r["score"])
    log.info(
        "Best gain: %.1fx  lum=%.1f  sat=%.2f%%  score=%.1f",
        best["gain"], best["lum"], best["sat_pct"], best["score"],
    )
    return best, results


# ---------------------------------------------------------------------------
# Phase 4 – White-balance mode sweep
# ---------------------------------------------------------------------------

def wb_sweep(
    sensor_id: int,
    width: int,
    height: int,
    fps: int,
    exp_ns: int,
    gain: float,
    output_dir: Path,
    wb_settle_min_seconds: float = WB_SETTLE_MIN_SECONDS,
    wb_settle_timeout_seconds: float = WB_SETTLE_TIMEOUT_SECONDS,
) -> tuple[str, int, list[dict]]:
    """Compare WB modes at the pinned exposure; return (name, val, all_results)."""
    log.info(
        "Phase 4 — WB sweep  (sensor %d, exposure %d µs, gain %.1fx)",
        sensor_id, exp_ns // 1000, gain,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    wb_results: list[dict] = []

    for wb_name, wb_val in WB_CANDIDATES:
        log.info("  WBMode.%s (%d) …", wb_name, wb_val)

        cam = JetsonCamera(
            sensor_id=sensor_id,
            width=width,
            height=height,
            fps=fps,
            wbmode=wb_val,
            aelock=True,
            # For fixed WB presets AWB is not running anyway;
            # for AUTO mode leave AWB free so it settles naturally.
            awblock=(wb_val != WBMode.AUTO),
            exposuretimerange=(exp_ns, exp_ns),
            gainrange=(gain, gain),
            ispdigitalgainrange=(1.0, 1.0),
        )

        try:
            cam.open()
        except RuntimeError as exc:
            log.warning("  Skipped WBMode.%s: %s", wb_name, exc)
            continue

        try:
            frame, settle_info = _wait_for_wb_settle(
                cam,
                wb_name,
                min_seconds=wb_settle_min_seconds,
                timeout_seconds=wb_settle_timeout_seconds,
            )
        finally:
            cam.close()

        r, g, b  = _channel_means(frame)
        rg_ratio = r / g if g > 0 else 0.0
        bg_ratio = b / g if g > 0 else 0.0
        # Neutral balance means R≈G≈B, i.e. both ratios close to 1.
        balance_score = abs(rg_ratio - 1.0) + abs(bg_ratio - 1.0)

        log.info(
            "    R=%.1f  G=%.1f  B=%.1f  R/G=%.3f  B/G=%.3f",
            r, g, b, rg_ratio, bg_ratio,
        )

        cv2.imwrite(str(output_dir / f"wb_{wb_name.lower()}.jpg"), frame)

        wb_results.append(
            dict(
                name=wb_name, val=wb_val,
                r=r, g=g, b=b,
                rg_ratio=rg_ratio, bg_ratio=bg_ratio,
                balance_score=balance_score,
                **settle_info,
            )
        )

    if not wb_results:
        log.warning("WB sweep produced no results — defaulting to DAYLIGHT.")
        return "DAYLIGHT", WBMode.DAYLIGHT, []

    most_neutral = min(wb_results, key=lambda x: x["balance_score"])
    log.info(
        "Most neutral WB mode: %s  (R/G=%.3f, B/G=%.3f)",
        most_neutral["name"], most_neutral["rg_ratio"], most_neutral["bg_ratio"],
    )
    # Note: DAYLIGHT is generally the sensible default for sunlit Earth imagery.
    # The "most neutral" result above is a starting point; inspect the saved
    # images visually to confirm the colour rendering is acceptable.
    return most_neutral["name"], most_neutral["val"], wb_results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

_TOML_TEMPLATE = """\
# [camera-isp] block generated by calibration.py
# Sensor {sensor_id}  {width}x{height} @ {fps} fps  —  {timestamp}
#
# Calibration summary
#   Converged luminance  : {converged_lum:.1f} / 255
#   Selected exposure    : {exp_us} µs  (lum={cal_lum:.1f}, sat={sat:.3f}%, score={score:.1f})
#   Selected gain        : {gain:.1f}x  (lum={gain_lum:.1f}, sat={gain_sat:.3f}%, score={gain_score:.1f})
#   Selected WB mode     : {wb_name} ({wb_val})
#   WB settle check      : {wb_settle_status} after {wb_settle_s:.2f}s / {wb_settle_frames} frames
#
# Review the saved JPEGs in {out_dir} before committing these values.
# Paste this block into config/config.toml, replacing any existing [camera-isp] section.

[camera-isp]
wbmode               = {wb_val}      # {wb_name}
aelock               = true
awblock              = true
ee_mode              = 1             # EdgeEnhance_Fast
ee_strength          = -1.0          # driver default
aeantibanding        = 1             # auto
exposurecompensation = 0.0
tnr_mode             = 1             # TNR_Fast
tnr_strength         = -1.0          # driver default
saturation           = 1.0
fps                  = {fps}
max_buffers          = 2
exposuretimerange    = [{exp_ns}, {exp_ns}]
gainrange            = [{gain:.1f}, {gain:.1f}]
ispdigitalgainrange  = [1.0, 1.0]
"""


def write_report(
    output_dir: Path,
    sensor_id: int,
    width: int,
    height: int,
    fps: int,
    converged_lum: float,
    best_exp: dict,
    best_gain: dict,
    gain_results: list[dict],
    wb_name: str,
    wb_val: int,
    wb_results: list[dict],
) -> None:
    exp_ns = best_exp["exp_ns"]
    exp_us = exp_ns // 1000
    gain = best_gain["gain"]
    selected_wb = next((wb for wb in wb_results if wb["val"] == wb_val), {})

    toml_str = _TOML_TEMPLATE.format(
        sensor_id=sensor_id,
        width=width,
        height=height,
        fps=fps,
        timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        converged_lum=converged_lum,
        exp_us=exp_us,
        cal_lum=best_exp["lum"],
        sat=best_exp["sat_pct"],
        score=best_exp["score"],
        gain=gain,
        gain_lum=best_gain["lum"],
        gain_sat=best_gain["sat_pct"],
        gain_score=best_gain["score"],
        wb_name=wb_name,
        wb_val=wb_val,
        wb_settle_status=(
            "settled"
            if selected_wb.get("settled")
            else selected_wb.get("settle_reason", "unknown")
        ),
        wb_settle_s=float(selected_wb.get("settle_s", 0.0)),
        wb_settle_frames=int(selected_wb.get("settle_frames", 0)),
        exp_ns=exp_ns,
        out_dir=output_dir,
    )

    result_path = output_dir / "calibration_result.toml"
    result_path.write_text(toml_str)
    log.info("Written: %s", result_path)

    # Gain comparison table
    if gain_results:
        print("\n--- Gain comparison ---")
        print(f"  {'Gain':>6}  {'Lum':>6}  {'Sat %':>7}  {'Dark %':>7}  {'Score':>7}")
        for gain_result in gain_results:
            marker = "  <-- selected" if gain_result["gain"] == gain else ""
            print(
                f"  {gain_result['gain']:5.1f}x"
                f"  {gain_result['lum']:6.1f}"
                f"  {gain_result['sat_pct']:7.2f}"
                f"  {gain_result['dark_pct']:7.2f}"
                f"  {gain_result['score']:7.1f}{marker}"
            )

    # WB comparison table
    if wb_results:
        print("\n--- White-balance comparison ---")
        print(
            f"  {'Mode':<16}  {'R':>6}  {'G':>6}  {'B':>6}  {'R/G':>6}  {'B/G':>6}"
            f"  {'settle':>8}  {'frames':>6}"
        )
        for wb in wb_results:
            marker = "  <-- selected" if wb["val"] == wb_val else ""
            settle = f"{wb['settle_s']:.2f}s" if wb["settled"] else "timeout"
            print(
                f"  {wb['name']:<16}  {wb['r']:6.1f}  {wb['g']:6.1f}  {wb['b']:6.1f}"
                f"  {wb['rg_ratio']:6.3f}  {wb['bg_ratio']:6.3f}"
                f"  {settle:>8}  {wb['settle_frames']:6d}{marker}"
            )

    print("\n--- Recommended [camera-isp] block ---")
    print(toml_str)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ground calibration for flight cameras (nvarguscamerasrc).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--sensor-id",        type=int,   default=0,
                   help="Camera sensor index")
    p.add_argument("--width",            type=int,   default=4608,
                   help="Capture width in pixels")
    p.add_argument("--height",           type=int,   default=2592,
                   help="Capture height in pixels")
    p.add_argument("--fps",              type=int,   default=10,
                   help="Pipeline frame rate")
    p.add_argument("--target-luminance", type=float, default=float(TARGET_LUMINANCE),
                   help="Target mean luminance (0–255)")
    p.add_argument("--out",              type=str,   default="calibration",
                   help="Root output directory")
    p.add_argument("--skip-convergence", action="store_true",
                   help="Skip Phase 1 (AE convergence check)")
    p.add_argument("--wb-settle-min-seconds", type=float,
                   default=WB_SETTLE_MIN_SECONDS,
                   help="Minimum time to observe each new WB mode before accepting stability")
    p.add_argument("--wb-settle-timeout", type=float,
                   default=WB_SETTLE_TIMEOUT_SECONDS,
                   help="Maximum time to wait for each WB mode to settle")
    args = p.parse_args()
    if args.wb_settle_min_seconds < 0:
        p.error("--wb-settle-min-seconds must be >= 0")
    if args.wb_settle_timeout <= 0:
        p.error("--wb-settle-timeout must be > 0")
    if args.wb_settle_timeout < args.wb_settle_min_seconds:
        p.error("--wb-settle-timeout must be >= --wb-settle-min-seconds")
    return args


if __name__ == "__main__":
    args = _parse_args()

    output_dir = Path(args.out) / f"cam{args.sensor_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        "=== Calibration start — sensor %d  %dx%d @ %d fps ===",
        args.sensor_id, args.width, args.height, args.fps,
    )

    # Pre-flight
    log.info("--- Pre-flight checks ---")
    if not preflight_check(args.sensor_id):
        log.error("Pre-flight checks failed — aborting.  Fix the issues above and retry.")
        sys.exit(1)
    log.info("--- Pre-flight OK ---")

    # Phase 1
    if args.skip_convergence:
        converged_lum = args.target_luminance
        log.info("Skipping convergence phase — target luminance %.1f", converged_lum)
    else:
        converged_lum = wait_for_convergence(
            args.sensor_id, args.width, args.height, args.fps,
        )

    # Phase 2
    best_exp, _all_exp = exposure_sweep(
        args.sensor_id, args.width, args.height, args.fps,
        output_dir, args.target_luminance,
    )

    # Phase 3
    best_gain, all_gain = gain_sweep(
        args.sensor_id, args.width, args.height, args.fps,
        best_exp["exp_ns"], output_dir, args.target_luminance,
    )

    # Phase 4
    wb_name, wb_val, wb_results = wb_sweep(
        args.sensor_id, args.width, args.height, args.fps,
        best_exp["exp_ns"], best_gain["gain"], output_dir,
        args.wb_settle_min_seconds, args.wb_settle_timeout,
    )

    # Report
    write_report(
        output_dir,
        args.sensor_id, args.width, args.height, args.fps,
        converged_lum, best_exp, best_gain, all_gain, wb_name, wb_val, wb_results,
    )

    log.info("=== Calibration complete ===")
