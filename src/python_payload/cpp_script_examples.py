"""
Example wrappers for the payload C++ dataset and OD scripts.

The goal of this file is to show how Python-side payload code can:
  - configure dataset and OD runs,
  - call the C++ binaries,
  - capture stdout/stderr,
  - decode process return codes and OD stage/error messages.

It is intentionally usable as both a reference module and a small CLI:

    python3 src/python_payload/cpp_script_examples.py od-on-dataset \
        data/datasets/17R_Florida_nadir_test

    python3 src/python_payload/cpp_script_examples.py reprocess \
        data/datasets/17R_Florida_nadir_test --target-stage LDNeted --overwrite

The examples assume they are run from the repository root after the C++ targets
have been built.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
BIN_DIR = REPO_ROOT / "bin"


ERROR_CODES: dict[int, str] = {
    0: "OK",
    1: "PLACEHOLDER",
    2: "INVALID_COMMAND_ID",
    3: "INVALID_COMMAND_ARGUMENTS",
    4: "NO_FILE_READY",
    5: "NO_MORE_PACKET_FOR_FILE",
    6: "FAIL_TO_READ_FILE",
    7: "FILE_NOT_AVAILABLE",
    8: "FILE_DOES_NOT_EXIST",
    9: "FILE_NOT_FOUND",
    10: "START_BYTE_OUT_OF_RANGE",
    11: "FAILED_TO_GRAB_FILE_CHUNK",
    12: "CAMERA_CAPTURE_FAILED",
    13: "CAMERA_INITIALIZATION_FAILED",
    14: "UART_OPEN_FAILED",
    15: "UART_CLOSE_FAILED",
    16: "UART_OPEN_FAILED_AFTER_RETRY",
    17: "UART_CLOSE_FAILED_AFTER_RETRY",
    18: "UART_NOT_OPEN",
    19: "UART_GETATTR_FAILED",
    20: "UART_SETATTR_FAILED",
    21: "UART_FAILED_WRITE",
    22: "UART_WRITE_BUFFER_OVERFLOW",
    23: "UART_INCOMPLETE_READ",
    24: "NN_FAILED_TO_OPEN_ENGINE_FILE",
    25: "NN_FAILED_TO_CREATE_RUNTIME",
    26: "NN_FAILED_TO_CREATE_ENGINE",
    27: "NN_FAILED_TO_CREATE_EXECUTION_CONTEXT",
    28: "NN_FAILED_TO_LOAD_ENGINE",
    29: "NN_ENGINE_NOT_INITIALIZED",
    30: "NN_CUDA_MEMCPY_FAILED",
    31: "NN_POINTER_NULL",
    32: "NN_INFERENCE_FAILED",
    33: "NN_NO_FRAME_AVAILABLE",
    34: "NN_INSUFFICIENT_GPU_MEMORY",
    35: "NN_INVALID_VERSION",
    36: "ODMEAS_NOT_VALID",
    37: "BATCH_OPT_BUILD_FAILED",
    38: "BATCH_OPT_NO_CONVERGENCE",
    39: "BATCH_OPT_SOLVER_FAILED",
    40: "BATCH_OPT_INVALID_OUTPUT",
    41: "UNDEFINED",
}


OD_STAGES: dict[int, str] = {
    0: "DATASET_NOT_AVAILABLE",
    1: "DATASET_NOT_PROCESSED",
    2: "DATASET_PROCESSED",
    3: "MEASUREMENTS_READY",
    4: "INITIAL_GUESS_CREATED",
    5: "OD_COMPLETED",
    6: "FAILED",
}


PROCESSING_STAGES: dict[str, int] = {
    "NotPrefiltered": 0,
    "Prefiltered": 1,
    "RCNeted": 2,
    "LDNeted": 3,
}


CAPTURE_MODES: dict[str, int] = {
    "IDLE": 0,
    "CAPTURE_SINGLE": 1,
    "PERIODIC": 2,
    "PERIODIC_EARTH": 3,
    "PERIODIC_ROI": 4,
    "PERIODIC_LDMK": 5,
}


IMU_COLLECTION_MODES: dict[str, int] = {
    "NONE": 0,
    "GYRO_ONLY": 1,
    "GYRO_TEMP": 2,
    "GYRO_MAG_TEMP": 3,
}


CAMERA_ISP_KEYS = {
    "wbmode",
    "aelock",
    "awblock",
    "ee_mode",
    "ee_strength",
    "aeantibanding",
    "exposurecompensation",
    "tnr_mode",
    "tnr_strength",
    "saturation",
    "fps",
    "max_buffers",
    "exposuretimerange",
    "gainrange",
    "ispdigitalgainrange",
}


CAMERA_CALIBRATION_KEYS = {
    "camera_matrix",
    "dist_coeffs",
}


@dataclass(frozen=True)
class ScriptResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    error_name: str
    od_stage: str | None = None
    od_error: str | None = None
    results_dir: str | None = None
    dataset_folder: str | None = None

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def raise_for_failure(self) -> None:
        if self.ok:
            return
        details = [
            f"command: {' '.join(self.command)}",
            f"return code: {self.returncode} ({self.error_name})",
        ]
        if self.od_stage:
            details.append(f"OD stage: {self.od_stage}")
        if self.od_error:
            details.append(f"OD error: {self.od_error}")
        if self.stderr.strip():
            details.append(f"stderr: {self.stderr.strip()}")
        raise RuntimeError("; ".join(details))


def _repo_relative(path: str | Path) -> str:
    path = Path(path)
    if path.is_absolute():
        try:
            return str(path.relative_to(REPO_ROOT))
        except ValueError:
            return str(path)
    return str(path)


def _decode_returncode(returncode: int) -> str:
    if returncode < 0:
        return f"terminated_by_signal_{-returncode}"
    return ERROR_CODES.get(returncode, f"unknown_error_{returncode}")


def _parse_od_failure(output: str) -> tuple[str | None, str | None]:
    match = re.search(r"failed at stage (\d+) with error code (\d+)", output, re.IGNORECASE)
    if not match:
        return None, None
    stage_num = int(match.group(1))
    error_num = int(match.group(2))
    return OD_STAGES.get(stage_num, f"unknown_stage_{stage_num}"), _decode_returncode(error_num)


def _parse_od_success(output: str) -> tuple[str | None, str | None]:
    dataset_match = re.search(r"OD complete\. Dataset\s+(\S+)\s+results in\s+(\S+)", output)
    if dataset_match:
        return dataset_match.group(1), dataset_match.group(2)
    result_match = re.search(r"OD complete\. Results in\s+(\S+)", output)
    if result_match:
        return None, result_match.group(1)
    return None, None


def run_binary(
    argv: Sequence[str | Path],
    *,
    cwd: Path = REPO_ROOT,
    env: Mapping[str, str] | None = None,
    check: bool = False,
) -> ScriptResult:
    """Run a payload binary and decode common return/error information."""

    command = [str(arg) for arg in argv]
    completed = subprocess.run(
        command,
        cwd=cwd,
        env={**os.environ, **dict(env or {})},
        text=True,
        capture_output=True,
    )
    combined_output = "\n".join([completed.stdout, completed.stderr])
    od_stage, od_error = _parse_od_failure(combined_output)
    dataset_folder, results_dir = _parse_od_success(combined_output)

    result = ScriptResult(
        command=command,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        error_name=_decode_returncode(completed.returncode),
        od_stage=od_stage,
        od_error=od_error,
        results_dir=results_dir,
        dataset_folder=dataset_folder,
    )
    if check:
        result.raise_for_failure()
    return result


def write_dataset_config(
    folder: str | Path,
    *,
    maximum_period: float = 10.0,
    target_frame_nb: int = 4,
    dataset_capture_mode: str = "PERIODIC_LDMK",
    imu_collection_mode: str = "GYRO_ONLY",
    image_capture_rate: int = 1,
    imu_sample_rate_hz: float = 1.0,
    target_processing_stage: str = "LDNeted",
    active_cameras: Sequence[bool] | None = None,
) -> Path:
    """
    Write the dataset_config.toml consumed by run_dataset and run_od_pipeline.

    Enum values are intentionally passed by readable names here, then converted
    to the numeric values expected by the C++ TOML parser.

    active_cameras: sequence of 4 bools selecting which cameras to use (default: all enabled).
    """

    if active_cameras is not None and len(active_cameras) != 4:
        raise ValueError("active_cameras must have exactly 4 elements")

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "dataset_config.toml"
    lines = [
        f"maximum_period = {maximum_period}",
        f"target_frame_nb = {target_frame_nb}",
        f"dataset_capture_mode = {CAPTURE_MODES[dataset_capture_mode]}",
        f"imu_collection_mode = {IMU_COLLECTION_MODES[imu_collection_mode]}",
        f"image_capture_rate = {image_capture_rate}",
        f"imu_sample_rate_hz = {imu_sample_rate_hz}",
        f"target_processing_stage = {PROCESSING_STAGES[target_processing_stage]}",
    ]
    if active_cameras is not None:
        bool_strs = ", ".join("true" if b else "false" for b in active_cameras)
        lines.append(f"active_cameras = [{bool_strs}]")
    lines.append("")
    path.write_text("\n".join(lines))
    return path


def _toml_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, Sequence):
        return "[" + ", ".join(_toml_literal(v) for v in value) + "]"
    raise TypeError(f"unsupported TOML value type: {type(value).__name__}")


def _normalize_updates(updates: Mapping[str, Any], allowed_keys: set[str]) -> dict[str, Any]:
    unknown = sorted(set(updates) - allowed_keys)
    if unknown:
        raise ValueError(f"unknown config key(s): {', '.join(unknown)}")
    return dict(updates)


def _update_toml_section(config_path: str | Path, section: str, updates: Mapping[str, Any]) -> Path:
    """
    Update scalar/array assignments inside one top-level TOML section.

    This intentionally avoids rewriting the whole TOML document. Existing comments
    and spacing are preserved where a key already exists; new keys are appended to
    the section before the next TOML header.
    """

    path = Path(config_path)
    lines = path.read_text().splitlines(keepends=True)

    header = f"[{section}]"
    start = None
    for idx, line in enumerate(lines):
        if line.strip() == header:
            start = idx
            break
    if start is None:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        if lines and lines[-1].strip():
            lines.append("\n")
        lines.append(f"{header}\n")
        start = len(lines) - 1

    end = len(lines)
    for idx in range(start + 1, len(lines)):
        stripped = lines[idx].strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            end = idx
            break

    remaining = dict(updates)
    key_pattern = re.compile(r"^(\s*)([A-Za-z0-9_]+)(\s*=\s*)(.*?)(\s*(#.*)?)$")
    commented_key_pattern = re.compile(r"^(\s*)#\s*([A-Za-z0-9_]+)(\s*=\s*)(.*?)(\s*(#.*)?)$")
    for idx in range(start + 1, end):
        match = key_pattern.match(lines[idx].rstrip("\n"))
        if not match:
            match = commented_key_pattern.match(lines[idx].rstrip("\n"))
            if not match:
                continue
        key = match.group(2)
        if key not in remaining:
            continue
        newline = "\n" if lines[idx].endswith("\n") else ""
        lines[idx] = (
            f"{match.group(1)}{key}{match.group(3)}"
            f"{_toml_literal(remaining.pop(key))}{match.group(5)}{newline}"
        )

    insert_at = end
    while insert_at > start + 1:
        previous = lines[insert_at - 1].strip()
        if previous == "" or previous.startswith("#"):
            insert_at -= 1
            continue
        break
    if remaining and insert_at > start + 1 and lines[insert_at - 1].strip():
        lines.insert(insert_at, "\n")
        insert_at += 1
        end += 1
    for key, value in remaining.items():
        lines.insert(insert_at, f"{key} = {_toml_literal(value)}\n")
        insert_at += 1

    path.write_text("".join(lines))
    return path


def update_camera_isp_config(
    system_config_path: str | Path = "config/config.toml",
    **updates: Any,
) -> Path:
    """
    Update the [camera-isp] section in the system config.

    Example:
        update_camera_isp_config(
            fps=5,
            wbmode=1,
            exposuretimerange=[13000, 683709000],
            gainrange=[1.0, 8.0],
        )
    """

    return _update_toml_section(
        system_config_path,
        "camera-isp",
        _normalize_updates(updates, CAMERA_ISP_KEYS),
    )


def update_camera_calibration_config(
    system_config_path: str | Path = "config/config.toml",
    *,
    camera_matrix: Sequence[float] | None = None,
    dist_coeffs: Sequence[float] | None = None,
) -> Path:
    """
    Update the [camera-calibration] section in the system config.

    camera_matrix must be 9 row-major values:
        [fx, 0, cx, 0, fy, cy, 0, 0, 1]

    dist_coeffs must be 5 values:
        [k1, k2, p1, p2, k3]
    """

    updates: dict[str, Any] = {}
    if camera_matrix is not None:
        if len(camera_matrix) != 9:
            raise ValueError("camera_matrix must contain exactly 9 values")
        updates["camera_matrix"] = [float(v) for v in camera_matrix]
    if dist_coeffs is not None:
        if len(dist_coeffs) != 5:
            raise ValueError("dist_coeffs must contain exactly 5 values")
        updates["dist_coeffs"] = [float(v) for v in dist_coeffs]
    if not updates:
        raise ValueError("provide camera_matrix and/or dist_coeffs")

    return _update_toml_section(system_config_path, "camera-calibration", updates)


def run_dataset(
    *,
    config_path: str | Path | None = None,
    out_path: str | Path | None = None,
    check: bool = False,
) -> ScriptResult:
    """
    Run scripts/run_dataset.cpp via bin/run_dataset.

    config_path: path to a dataset_config.toml (--config flag). Defaults to
                 config/dataset_config.toml when omitted.
    out_path:    file that will receive the generated dataset folder path
                 (--out flag). Defaults to path.out when omitted.
    """

    cmd: list[str | Path] = [BIN_DIR / "run_dataset"]
    if config_path is not None:
        cmd += ["--config", _repo_relative(config_path)]
    if out_path is not None:
        cmd += ["--out", str(out_path)]
    return run_binary(cmd, check=check)


def reprocess_dataset(
    dataset_folder: str | Path,
    *,
    target_stage: str = "LDNeted",
    overwrite: bool = False,
    rc_version: int = 2,
    ld_version: int = 2,
    check: bool = False,
) -> ScriptResult:
    """Run scripts/reprocess_dataset.cpp via bin/reprocess_dataset."""

    return run_binary(
        [
            BIN_DIR / "reprocess_dataset",
            _repo_relative(dataset_folder),
            str(PROCESSING_STAGES[target_stage]),
            "1" if overwrite else "0",
            str(rc_version),
            str(ld_version),
        ],
        check=check,
    )


def run_od_on_dataset(
    dataset_folder: str | Path,
    *,
    od_config_path: str | Path = "config/od.toml",
    system_config_path: str | Path = "config/config.toml",
    out_path: str | Path | None = None,
    check: bool = False,
) -> ScriptResult:
    """Run scripts/run_od_on_dataset.cpp via bin/RUN_OD_ON_DATASET.

    out_path: file that will receive the generated results directory path
              (--out flag). Defaults to path.out when omitted.
    """

    cmd: list[str | Path] = [
        BIN_DIR / "RUN_OD_ON_DATASET",
        _repo_relative(dataset_folder),
        _repo_relative(od_config_path),
        _repo_relative(system_config_path),
    ]
    if out_path is not None:
        cmd += ["--out", str(out_path)]
    return run_binary(cmd, check=check)


def run_od_pipeline(
    dataset_config_folder: str | Path = "config",
    *,
    od_config_path: str | Path = "config/od.toml",
    system_config_path: str | Path = "config/config.toml",
    out_path: str | Path | None = None,
    check: bool = False,
) -> ScriptResult:
    """Run scripts/run_od_pipeline.cpp via bin/RUN_OD_PIPELINE.

    out_path: file that will receive the generated results directory path
              (--out flag). Defaults to path.out when omitted.
    """

    cmd: list[str | Path] = [
        BIN_DIR / "RUN_OD_PIPELINE",
        _repo_relative(dataset_config_folder),
        _repo_relative(od_config_path),
        _repo_relative(system_config_path),
    ]
    if out_path is not None:
        cmd += ["--out", str(out_path)]
    return run_binary(cmd, check=check)


def print_result(result: ScriptResult) -> None:
    print(f"command: {' '.join(result.command)}")
    print(f"return code: {result.returncode} ({result.error_name})")
    if result.od_stage:
        print(f"OD stage: {result.od_stage}")
    if result.od_error:
        print(f"OD error: {result.od_error}")
    if result.dataset_folder:
        print(f"dataset folder: {result.dataset_folder}")
    if result.results_dir:
        print(f"results dir: {result.results_dir}")
    if result.stdout.strip():
        print("\nstdout:")
        print(result.stdout.rstrip())
    if result.stderr.strip():
        print("\nstderr:")
        print(result.stderr.rstrip())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    cfg = sub.add_parser("write-dataset-config")
    cfg.add_argument("--folder", default="config")
    cfg.add_argument("--maximum-period", type=float, default=10.0)
    cfg.add_argument("--target-frame-nb", type=int, default=4)
    cfg.add_argument("--capture-mode", choices=CAPTURE_MODES.keys(), default="PERIODIC_LDMK")
    cfg.add_argument("--imu-mode", choices=IMU_COLLECTION_MODES.keys(), default="GYRO_ONLY")
    cfg.add_argument("--image-capture-rate", type=int, default=1)
    cfg.add_argument("--imu-sample-rate-hz", type=float, default=1.0)
    cfg.add_argument("--target-stage", choices=PROCESSING_STAGES.keys(), default="LDNeted")

    ds = sub.add_parser("dataset")
    ds.add_argument("--config-folder", default="config")

    rep = sub.add_parser("reprocess")
    rep.add_argument("dataset_folder")
    rep.add_argument("--target-stage", choices=PROCESSING_STAGES.keys(), default="LDNeted")
    rep.add_argument("--overwrite", action="store_true")
    rep.add_argument("--rc-version", type=int, default=2)
    rep.add_argument("--ld-version", type=int, default=2)

    od = sub.add_parser("od-on-dataset")
    od.add_argument("dataset_folder")
    od.add_argument("--od-config", default="config/od.toml")
    od.add_argument("--system-config", default="config/config.toml")

    pipe = sub.add_parser("od-pipeline")
    pipe.add_argument("dataset_config_folder", nargs="?", default="config")
    pipe.add_argument("--od-config", default="config/od.toml")
    pipe.add_argument("--system-config", default="config/config.toml")

    isp = sub.add_parser("update-camera-isp")
    isp.add_argument("--system-config", default="config/config.toml")
    isp.add_argument("--wbmode", type=int)
    isp.add_argument("--aelock", type=int, choices=[0, 1])
    isp.add_argument("--awblock", type=int, choices=[0, 1])
    isp.add_argument("--ee-mode", type=int)
    isp.add_argument("--ee-strength", type=float)
    isp.add_argument("--aeantibanding", type=int)
    isp.add_argument("--exposurecompensation", type=float)
    isp.add_argument("--tnr-mode", type=int)
    isp.add_argument("--tnr-strength", type=float)
    isp.add_argument("--saturation", type=float)
    isp.add_argument("--fps", type=int)
    isp.add_argument("--max-buffers", type=int)
    isp.add_argument("--exposuretimerange", type=int, nargs=2, metavar=("LOW_NS", "HIGH_NS"))
    isp.add_argument("--gainrange", type=float, nargs=2, metavar=("LOW", "HIGH"))
    isp.add_argument("--ispdigitalgainrange", type=float, nargs=2, metavar=("LOW", "HIGH"))

    calib = sub.add_parser("update-camera-calibration")
    calib.add_argument("--system-config", default="config/config.toml")
    calib.add_argument("--camera-matrix", type=float, nargs=9, metavar="K")
    calib.add_argument("--dist-coeffs", type=float, nargs=5, metavar="D")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        if args.command == "write-dataset-config":
            path = write_dataset_config(
                args.folder,
                maximum_period=args.maximum_period,
                target_frame_nb=args.target_frame_nb,
                dataset_capture_mode=args.capture_mode,
                imu_collection_mode=args.imu_mode,
                image_capture_rate=args.image_capture_rate,
                imu_sample_rate_hz=args.imu_sample_rate_hz,
                target_processing_stage=args.target_stage,
            )
            print(f"wrote {path}")
            return 0

        if args.command == "update-camera-isp":
            cli_updates = {
                "wbmode": args.wbmode,
                "aelock": None if args.aelock is None else bool(args.aelock),
                "awblock": None if args.awblock is None else bool(args.awblock),
                "ee_mode": args.ee_mode,
                "ee_strength": args.ee_strength,
                "aeantibanding": args.aeantibanding,
                "exposurecompensation": args.exposurecompensation,
                "tnr_mode": args.tnr_mode,
                "tnr_strength": args.tnr_strength,
                "saturation": args.saturation,
                "fps": args.fps,
                "max_buffers": args.max_buffers,
                "exposuretimerange": args.exposuretimerange,
                "gainrange": args.gainrange,
                "ispdigitalgainrange": args.ispdigitalgainrange,
            }
            updates = {k: v for k, v in cli_updates.items() if v is not None}
            if not updates:
                raise ValueError("provide at least one camera ISP field to update")
            path = update_camera_isp_config(args.system_config, **updates)
            print(f"updated [camera-isp] in {path}")
            return 0

        if args.command == "update-camera-calibration":
            path = update_camera_calibration_config(
                args.system_config,
                camera_matrix=args.camera_matrix,
                dist_coeffs=args.dist_coeffs,
            )
            print(f"updated [camera-calibration] in {path}")
            return 0

        if args.command == "dataset":
            result = run_dataset(config_folder=args.config_folder)
        elif args.command == "reprocess":
            result = reprocess_dataset(
                args.dataset_folder,
                target_stage=args.target_stage,
                overwrite=args.overwrite,
                rc_version=args.rc_version,
                ld_version=args.ld_version,
            )
        elif args.command == "od-on-dataset":
            result = run_od_on_dataset(
                args.dataset_folder,
                od_config_path=args.od_config,
                system_config_path=args.system_config,
            )
        elif args.command == "od-pipeline":
            result = run_od_pipeline(
                args.dataset_config_folder,
                od_config_path=args.od_config,
                system_config_path=args.system_config,
            )
        else:
            raise AssertionError(f"unhandled command {args.command}")

        print_result(result)
        return result.returncode
    except Exception as exc:
        print(f"python wrapper error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
