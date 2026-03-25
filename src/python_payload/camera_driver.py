"""
jetson_camera.py
================
A class-based interface for capturing images from CSI cameras on NVIDIA Jetson
devices using GStreamer and the nvarguscamerasrc element via OpenCV.

Typical usage
-------------
    cam = JetsonCamera(width=1920, height=1080, fps=14, wbmode=0)
    cam.open()
    frame = cam.capture()
    cam.save(frame, "output.jpg")
    cam.close()

    # Or use as a context manager:
    with JetsonCamera(width=1920, height=1080) as cam:
        frame = cam.capture()
        cam.save(frame, "output.jpg")
"""

import logging
import cv2

# ---------------------------------------------------------------------------
# Module-level logger – callers can configure this however they like.
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default camera parameters
# ---------------------------------------------------------------------------

# White-balance mode constants (mirrors nvarguscamerasrc wbmode enum)
class WBMode:
    """Symbolic names for the wbmode parameter of nvarguscamerasrc."""
    OFF             = 0
    AUTO            = 1
    INCANDESCENT    = 2
    FLUORESCENT     = 3
    WARM_FLUORESCENT = 4
    DAYLIGHT        = 5
    CLOUDY_DAYLIGHT = 6
    TWILIGHT        = 7
    SHADE           = 8
    MANUAL          = 9


class EdgeEnhancementMode:
    """Symbolic names for the ee-mode parameter of nvarguscamerasrc."""
    OFF          = 0
    FAST         = 1
    HIGH_QUALITY = 2


class NoiseReductionMode:
    """Symbolic names for the tnr-mode parameter of nvarguscamerasrc."""
    OFF          = 0
    FAST         = 1
    HIGH_QUALITY = 2


class AeAntibandingMode:
    """Symbolic names for the aeantibanding parameter of nvarguscamerasrc."""
    OFF   = 0
    AUTO  = 1
    HZ_50 = 2
    HZ_60 = 3


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _range_str(low, high) -> str:
    """Format a (low, high) pair as the string nvarguscamerasrc expects.

    Args:
        low:  Lower bound of the range.
        high: Upper bound of the range.

    Returns:
        A quoted string like '"1 16"' ready to embed in a GStreamer
        pipeline description.
    """
    return f'"{low} {high}"'


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class JetsonCamera:
    """
    High-level wrapper around an NVIDIA Jetson CSI camera.

    The class builds a GStreamer pipeline that goes through
    nvarguscamerasrc → nvvidconv → videoconvert → appsink
    and exposes the camera via OpenCV's VideoCapture interface.

    Parameters
    ----------
    sensor_id : int
        Index of the camera sensor (default 0).
    width : int
        Capture width in pixels (default 1920).
    height : int
        Capture height in pixels (default 1080).
    fps : int
        Target frame rate (default 14).
    wbmode : int
        White-balance mode – use :class:WBMode constants (default
        WBMode.OFF).
    aelock : bool
        Lock auto-exposure when True (default False).
    awblock : bool
        Lock auto-white-balance when True (default False).
    exposuretimerange : tuple[int, int] or None
        (low_ns, high_ns) exposure time range in nanoseconds.
        Valid range: 500 000 – 65 487 000 ns.  Pass None to omit
        the parameter and let the ISP decide (default None).
    gainrange : tuple[float, float] or None
        (low, high) analogue gain range.
        Valid range: 1.0 – 16.0.  Pass None to omit (default None).
    ispdigitalgainrange : tuple[float, float] or None
        (low, high) ISP digital gain range.  Pass None to omit
        Valid range: 1 - 256
        (default None).
    ee_mode : int
        Edge-enhancement mode – use :class:EdgeEnhancementMode constants
        (default EdgeEnhancementMode.FAST).
    ee_strength : float
        Edge-enhancement strength in the range [-1, 1].  -1 means
        *use default* (default -1).
    aeantibanding : int
        Auto-exposure anti-banding mode – use :class:AeAntibandingMode
        constants (default AeAntibandingMode.AUTO).
    exposurecompensation : float
        Exposure compensation in EV, range [-2, 2] (default 0).
    tnr_mode : int
        Temporal noise-reduction mode – use :class:NoiseReductionMode
        constants (default NoiseReductionMode.FAST).
    tnr_strength : float
        Temporal noise-reduction strength in [-1, 1].  -1 means
        *use default* (default -1).
    saturation : float
        Colour saturation in [0, 2] (default 1).
    max_buffers : int
        Maximum number of frames buffered by the appsink element
        (default 2).

    Examples
    --------
    Basic usage::

        cam = JetsonCamera(width=1920, height=1080, fps=14)
        cam.open()
        frame = cam.capture()
        cam.save(frame, "snap.jpg")
        cam.close()

    Context manager::

        with JetsonCamera() as cam:
            frame = cam.capture()
    """

    # Nanosecond limits documented by nvarguscamerasrc
    _EXPOSURE_NS_MIN = 500_000
    _EXPOSURE_NS_MAX = 65_487_000

    # Gain limits documented by nvarguscamerasrc
    _GAIN_MIN = 1.0
    _GAIN_MAX = 16.0

    def __init__(
        self,
        *,
        sensor_id: int = 0,
        width: int = 1920,
        height: int = 1080,
        fps: int = 14,
        wbmode: int = WBMode.OFF,
        aelock: bool = False,
        awblock: bool = False,
        exposuretimerange: "tuple[int, int] | None" = None,
        gainrange: "tuple[float, float] | None" = None,
        ispdigitalgainrange: "tuple[float, float] | None" = None,
        ee_mode: int = EdgeEnhancementMode.FAST,
        ee_strength: float = -1.0,
        aeantibanding: int = AeAntibandingMode.AUTO,
        exposurecompensation: float = 0.0,
        tnr_mode: int = NoiseReductionMode.FAST,
        tnr_strength: float = -1.0,
        saturation: float = 1.0,
        max_buffers: int = 2,
    ) -> None:
        self._sensor_id = sensor_id
        self._width = width
        self._height = height
        self._fps = fps
        self._wbmode = wbmode
        self._aelock = aelock
        self._awblock = awblock
        self._exposuretimerange = exposuretimerange
        self._gainrange = gainrange
        self._ispdigitalgainrange = ispdigitalgainrange
        self._ee_mode = ee_mode
        self._ee_strength = ee_strength
        self._aeantibanding = aeantibanding
        self._exposurecompensation = exposurecompensation
        self._tnr_mode = tnr_mode
        self._tnr_strength = tnr_strength
        self._saturation = saturation
        self._max_buffers = max_buffers

        self._cap: cv2.VideoCapture | None = None

        self._validate_params()
        logger.debug("JetsonCamera initialised with: %s", self._param_summary())

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_params(self) -> None:
        """Validate constructor parameters and raise on obviously bad values.

        Raises
        ------
        ValueError
            If any parameter is outside its documented valid range.
        """
        if self._width <= 0 or self._height <= 0:
            raise ValueError(
                f"Width and height must be positive integers, "
                f"got {self._width}x{self._height}."
            )
        if self._fps <= 0:
            raise ValueError(f"fps must be a positive integer, got {self._fps}.")

        if self._exposuretimerange is not None:
            lo, hi = self._exposuretimerange
            if lo < self._EXPOSURE_NS_MIN or hi > self._EXPOSURE_NS_MAX or lo > hi:
                raise ValueError(
                    f"exposuretimerange ({lo}, {hi}) is outside the valid range "
                    f"[{self._EXPOSURE_NS_MIN}, {self._EXPOSURE_NS_MAX}] ns, "
                    f"or low > high."
                )

        if self._gainrange is not None:
            lo, hi = self._gainrange
            if lo < self._GAIN_MIN or hi > self._GAIN_MAX or lo > hi:
                raise ValueError(
                    f"gainrange ({lo}, {hi}) is outside the valid range "
                    f"[{self._GAIN_MIN}, {self._GAIN_MAX}], or low > high."
                )

        if not -2.0 <= self._exposurecompensation <= 2.0:
            raise ValueError(
                f"exposurecompensation must be in [-2, 2], "
                f"got {self._exposurecompensation}."
            )

        if not 0.0 <= self._saturation <= 2.0:
            raise ValueError(
                f"saturation must be in [0, 2], got {self._saturation}."
            )

    # ------------------------------------------------------------------
    # Pipeline construction
    # ------------------------------------------------------------------

    def build_pipeline(self) -> str:
        """Construct the GStreamer pipeline string from the current settings.

        The pipeline routes frames through::

            nvarguscamerasrc → nvvidconv → videoconvert → appsink

        Returns
        -------
        str
            A complete GStreamer pipeline description suitable for passing to
            cv2.VideoCapture(..., cv2.CAP_GSTREAMER).
        """
        src_params: list[str] = [
            f"sensor-id={self._sensor_id}",
            f"wbmode={self._wbmode}",
            f"aelock={'true' if self._aelock else 'false'}",
            f"awblock={'true' if self._awblock else 'false'}",
            f"ee-mode={self._ee_mode}",
            f"ee-strength={self._ee_strength}",
            f"aeantibanding={self._aeantibanding}",
            f"exposurecompensation={self._exposurecompensation}",
            f"tnr-mode={self._tnr_mode}",
            f"tnr-strength={self._tnr_strength}",
            f"saturation={self._saturation}",
        ]

        # Optional range parameters – only added when explicitly provided
        if self._exposuretimerange is not None:
            lo, hi = self._exposuretimerange
            src_params.append(f"exposuretimerange={_range_str(lo, hi)}")

        if self._gainrange is not None:
            lo, hi = self._gainrange
            src_params.append(f"gainrange={_range_str(lo, hi)}")

        if self._ispdigitalgainrange is not None:
            lo, hi = self._ispdigitalgainrange
            src_params.append(f"ispdigitalgainrange={_range_str(lo, hi)}")

        param_str = " ".join(src_params)

        pipeline = (
            f"nvarguscamerasrc {param_str} ! "
            f"video/x-raw(memory:NVMM),width={self._width},height={self._height},"
            f"framerate={self._fps}/1,format=NV12 ! "
            "nvvidconv ! "
            "video/x-raw,format=BGRx ! "
            "videoconvert ! "
            "video/x-raw,format=BGR ! "
            f"appsink drop=true max-buffers={self._max_buffers}"
        )

        logger.debug("Built GStreamer pipeline: %s", pipeline)
        return pipeline

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Open the camera by instantiating the GStreamer pipeline.

        This method is idempotent: calling it on an already-open camera logs
        a warning and returns immediately without reopening.

        Raises
        ------
        RuntimeError
            If OpenCV cannot open the GStreamer pipeline (e.g. the sensor is
            not connected, or GStreamer is not built into your OpenCV).
        """
        if self._cap is not None and self._cap.isOpened():
            logger.warning(
                "open() called on an already-open camera (sensor-id=%d). "
                "Ignoring.",
                self._sensor_id,
            )
            return

        pipeline = self.build_pipeline()
        logger.info(
            "Opening camera sensor-id=%d at %dx%d @ %d fps …",
            self._sensor_id, self._width, self._height, self._fps,
        )

        self._cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

        if not self._cap.isOpened():
            self._cap = None
            raise RuntimeError(
                f"Failed to open camera sensor-id={self._sensor_id}. "
                "Check that the sensor is connected, that GStreamer support is "
                "compiled into OpenCV, and that nvarguscamerasrc is available."
            )

        logger.info("Camera sensor-id=%d opened successfully.", self._sensor_id)

    def close(self) -> None:
        """Release the camera and free GStreamer resources.

        Safe to call even if the camera was never opened.
        """
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Camera sensor-id=%d closed.", self._sensor_id)
        else:
            logger.debug("close() called but camera was not open.")

    @property
    def is_open(self) -> bool:
        """True if the camera pipeline is currently running."""
        return self._cap is not None and self._cap.isOpened()

    # ------------------------------------------------------------------
    # Frame capture
    # ------------------------------------------------------------------

    def capture(self):
        """Capture and return a single BGR frame as a NumPy array.

        Returns
        -------
        numpy.ndarray
            A (height, width, 3) uint8 BGR image.

        Raises
        ------
        RuntimeError
            If the camera has not been opened with :meth:open, or if
            reading a frame fails (e.g. the pipeline has stalled).
        """
        if not self.is_open:
            raise RuntimeError(
                "Cannot capture: camera is not open. Call open() first."
            )

        ret, frame = self._cap.read()

        if not ret or frame is None:
            raise RuntimeError(
                f"Failed to read a frame from camera sensor-id={self._sensor_id}. "
                "The pipeline may have stalled or the sensor disconnected."
            )

        logger.debug(
            "Captured frame from sensor-id=%d, shape=%s.",
            self._sensor_id, frame.shape,
        )
        return frame

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @staticmethod
    def save(frame, filename: str) -> None:
        """Save a frame to disk.

        Parameters
        ----------
        frame : numpy.ndarray
            BGR image as returned by :meth:capture.
        filename : str
            Destination file path.  The image format is inferred from the
            file extension (e.g. .jpg, .png).

        Raises
        ------
        ValueError
            If *frame* is None.
        RuntimeError
            If OpenCV cannot write the file (e.g. invalid path or
            unsupported extension).
        """
        if frame is None:
            raise ValueError("Cannot save a None frame.")

        success = cv2.imwrite(filename, frame)
        if not success:
            raise RuntimeError(
                f"cv2.imwrite failed for '{filename}'. "
                "Check the path exists and the extension is supported."
            )

        logger.info("Frame saved to '%s'.", filename)

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "JetsonCamera":
        """Open the camera when entering a with block."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close the camera when leaving a with block (even on error)."""
        self.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _param_summary(self) -> dict:
        """Return a dict of all current parameters (for logging)."""
        return {
            "sensor_id": self._sensor_id,
            "resolution": f"{self._width}x{self._height}",
            "fps": self._fps,
            "wbmode": self._wbmode,
            "aelock": self._aelock,
            "awblock": self._awblock,
            "exposuretimerange": self._exposuretimerange,
            "gainrange": self._gainrange,
            "ispdigitalgainrange": self._ispdigitalgainrange,
            "ee_mode": self._ee_mode,
            "ee_strength": self._ee_strength,
            "aeantibanding": self._aeantibanding,
            "exposurecompensation": self._exposurecompensation,
            "tnr_mode": self._tnr_mode,
            "tnr_strength": self._tnr_strength,
            "saturation": self._saturation,
        }

    def __repr__(self) -> str:
        status = "open" if self.is_open else "closed"
        return (
            f"JetsonCamera(sensor_id={self._sensor_id}, "
            f"{self._width}x{self._height}@{self._fps}fps, "
            f"status={status})"
        )


# ---------------------------------------------------------------------------
# Quick smoke-test / example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    cam = JetsonCamera(
        sensor_id=0,
        width=1920,
        height=1080,
        fps=14,
        # Only pass these if you want to pin exposure / gain:
        # exposuretimerange=(500_000, 65_487_000),
        # gainrange=(16.0, 16.0),
    )

    import time
    
    print(cam)

    with cam:
        time.sleep(5)
        frame = cam.capture()
        cam.save(frame, "test_capture.jpg")
        print(f"Saved frame with shape {frame.shape}")