#!/usr/bin/env python3
"""
Integration harness for the C++ reprocess + OD scripts.

This intentionally wraps the Python examples in src/python_payload rather than
calling the C++ binaries directly. The flow mirrors the C++ OD dataset test at a
script level:

  1. reprocess an existing stored dataset to LDNeted,
  2. run OD on that dataset,
  3. validate the OD result artifacts,
  4. run scripts/plot_batch_opt.py on the OD results,
  5. write a JSON and Markdown report.

By default this script does not force overwrite of existing inference artifacts.
Pass --overwrite to intentionally reprocess frames even when prior outputs exist.

CLI example:

    python3 tests/test_cpp_od_scripts.py \
        --dataset data/datasets/17R_Florida_nadir_test \
        --report-dir tests

Pytest example:

    RUN_OD_SCRIPT_INTEGRATION=1 \
    OD_SCRIPT_DATASETS=data/datasets/17R_Florida_nadir_test \
    pytest tests/test_cpp_od_scripts.py -s
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.python_payload.cpp_script_examples import (  # noqa: E402
    ScriptResult,
    reprocess_dataset,
    run_od_on_dataset,
)


@dataclass
class DatasetRunReport:
    dataset: str
    status: str
    reprocess: dict
    od: dict | None = None
    plot: dict | None = None
    od_result_json: dict | None = None
    errors: list[str] | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def _script_result_dict(result: ScriptResult) -> dict:
    data = asdict(result)
    data["ok"] = result.ok
    return data


def _run_plot_batch_opt(
    results_dir: str | Path,
    dataset_dir: str | Path,
    *,
    timeout_sec: float | None,
) -> dict:
    command = [
        sys.executable,
        "scripts/plot_batch_opt.py",
        str(results_dir),
        "--dataset",
        str(dataset_dir),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            timeout=timeout_sec,
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "ok": completed.returncode == 0,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "returncode": None,
            "ok": False,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "timed_out": True,
        }


def _validate_od_artifacts(results_dir: str | Path) -> tuple[dict | None, list[str]]:
    results_dir = Path(results_dir)
    errors: list[str] = []

    od_result_path = results_dir / "od_result.json"
    state_estimates_path = results_dir / "state_estimates.csv"
    initial_trajectory_path = results_dir / "initial_trajectory.csv"

    if not od_result_path.exists():
        errors.append(f"missing {od_result_path}")
        return None, errors
    if not state_estimates_path.exists():
        errors.append(f"missing {state_estimates_path}")
    if not initial_trajectory_path.exists():
        errors.append(f"missing {initial_trajectory_path}")

    try:
        meta = json.loads(od_result_path.read_text())
    except json.JSONDecodeError as exc:
        errors.append(f"invalid JSON in {od_result_path}: {exc}")
        return None, errors

    if meta.get("error_code") != 0:
        errors.append(f"od_result.json error_code is {meta.get('error_code')}, expected 0")
    outputs = meta.get("outputs", {})
    if outputs.get("num_state_estimates", 0) <= 0:
        errors.append("od_result.json outputs.num_state_estimates is missing or <= 0")

    return meta, errors


def _validate_plot_artifacts(results_dir: str | Path) -> list[str]:
    plots_dir = Path(results_dir) / "plots"
    if not plots_dir.exists():
        return [f"missing {plots_dir}"]
    if not any(plots_dir.iterdir()):
        return [f"{plots_dir} is empty"]
    return []


def run_reprocess_od_plot_flow(
    dataset: str | Path,
    *,
    overwrite: bool = False,
    rc_version: int = 2,
    ld_version: int = 2,
    od_config: str | Path = "config/od.toml",
    system_config: str | Path = "config/config.toml",
    run_plot: bool = True,
    timeout_sec: float | None = None,
) -> DatasetRunReport:
    dataset = Path(dataset)
    errors: list[str] = []

    reprocess = reprocess_dataset(
        dataset,
        target_stage="LDNeted",
        overwrite=overwrite,
        rc_version=rc_version,
        ld_version=ld_version,
    )
    if not reprocess.ok:
        return DatasetRunReport(
            dataset=str(dataset),
            status="reprocess_failed",
            reprocess=_script_result_dict(reprocess),
            errors=[f"reprocess failed with {reprocess.returncode} ({reprocess.error_name})"],
        )

    od = run_od_on_dataset(
        dataset,
        od_config_path=od_config,
        system_config_path=system_config,
    )
    if not od.ok:
        return DatasetRunReport(
            dataset=str(dataset),
            status="od_failed",
            reprocess=_script_result_dict(reprocess),
            od=_script_result_dict(od),
            errors=[f"OD failed with {od.returncode} ({od.error_name})"],
        )

    if not od.results_dir:
        errors.append("OD succeeded but no results_dir was parsed from output")
        return DatasetRunReport(
            dataset=str(dataset),
            status="artifact_failed",
            reprocess=_script_result_dict(reprocess),
            od=_script_result_dict(od),
            errors=errors,
        )

    meta, artifact_errors = _validate_od_artifacts(od.results_dir)
    errors.extend(artifact_errors)

    plot_result = None
    if run_plot and not errors:
        plot_result = _run_plot_batch_opt(od.results_dir, dataset, timeout_sec=timeout_sec)
        if not plot_result["ok"]:
            errors.append("plot_batch_opt.py failed")
        else:
            errors.extend(_validate_plot_artifacts(od.results_dir))

    status = "ok" if not errors else ("plot_failed" if plot_result and not plot_result["ok"] else "artifact_failed")
    return DatasetRunReport(
        dataset=str(dataset),
        status=status,
        reprocess=_script_result_dict(reprocess),
        od=_script_result_dict(od),
        plot=plot_result,
        od_result_json=meta,
        errors=errors,
    )


def discover_test_datasets() -> list[Path]:
    base = REPO_ROOT / "data" / "datasets"
    if not base.is_dir():
        return []
    return sorted(path for path in base.iterdir() if path.is_dir() and path.name.endswith("_test"))


def write_reports(reports: Sequence[DatasetRunReport], report_dir: str | Path) -> tuple[Path, Path]:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    json_path = report_dir / "cpp_od_script_test_report.json"
    md_path = report_dir / "cpp_od_script_test_report.md"

    json_path.write_text(json.dumps([asdict(r) for r in reports], indent=2))

    lines = [
        "# C++ OD Script Test Report",
        "",
        "| Dataset | Status | Results Dir | Errors |",
        "| --- | --- | --- | --- |",
    ]
    for report in reports:
        results_dir = ""
        if report.od:
            results_dir = report.od.get("results_dir") or ""
        errors = "<br>".join(report.errors or [])
        lines.append(f"| `{report.dataset}` | `{report.status}` | `{results_dir}` | {errors} |")
    lines.append("")
    md_path.write_text("\n".join(lines))
    return json_path, md_path


def _datasets_from_env() -> list[Path]:
    raw = os.getenv("OD_SCRIPT_DATASETS", "")
    if not raw.strip():
        return []
    return [Path(item.strip()) for item in raw.split(",") if item.strip()]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset folder to test. Repeatable. If omitted, use --discover.",
    )
    parser.add_argument(
        "--discover",
        action="store_true",
        help="Run all data/datasets/*_test datasets.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force reprocessing even when existing outputs are present. This mutates dataset frame JSON.",
    )
    parser.add_argument("--rc-version", type=int, default=2)
    parser.add_argument("--ld-version", type=int, default=2)
    parser.add_argument("--od-config", default="config/od.toml")
    parser.add_argument("--system-config", default="config/config.toml")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--timeout-sec", type=float, default=None)
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--report-dir", default="tests")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    datasets = [Path(p) for p in args.dataset]
    if args.discover:
        datasets.extend(discover_test_datasets())
    if not datasets:
        print("No datasets selected. Pass --dataset or --discover.", file=sys.stderr)
        return 2

    if args.overwrite:
        print("WARNING: --overwrite will mutate stored dataset frame JSON and inference outputs.")

    reports: list[DatasetRunReport] = []
    for dataset in datasets:
        print(f"\n=== {dataset} ===")
        report = run_reprocess_od_plot_flow(
            dataset,
            overwrite=args.overwrite,
            rc_version=args.rc_version,
            ld_version=args.ld_version,
            od_config=args.od_config,
            system_config=args.system_config,
            run_plot=not args.no_plot,
            timeout_sec=args.timeout_sec,
        )
        reports.append(report)
        print(f"status: {report.status}")
        if report.od and report.od.get("results_dir"):
            print(f"results: {report.od['results_dir']}")
        if report.errors:
            for error in report.errors:
                print(f"error: {error}", file=sys.stderr)
        if not report.ok and not args.keep_going:
            break

    json_path, md_path = write_reports(reports, args.report_dir)
    print(f"\nwrote {json_path}")
    print(f"wrote {md_path}")
    return 0 if all(r.ok for r in reports) else 1


def test_reprocess_od_plot_from_env() -> None:
    try:
        import pytest
    except ImportError:  # pragma: no cover - pytest imports this function in normal use
        pytest = None

    if os.getenv("RUN_OD_SCRIPT_INTEGRATION") != "1":
        if pytest is None:
            return
        pytest.skip("Set RUN_OD_SCRIPT_INTEGRATION=1 to run OD script integration tests")

    datasets = _datasets_from_env()
    if not datasets:
        if pytest is None:
            raise RuntimeError("OD_SCRIPT_DATASETS must be set")
        pytest.skip("Set OD_SCRIPT_DATASETS to one or more comma-separated dataset folders")

    overwrite = os.getenv("OD_SCRIPT_OVERWRITE") == "1"
    run_plot = os.getenv("OD_SCRIPT_NO_PLOT") != "1"
    timeout_raw = os.getenv("OD_SCRIPT_TIMEOUT_SEC")
    timeout_sec = float(timeout_raw) if timeout_raw else None

    reports = [
        run_reprocess_od_plot_flow(
            dataset,
            overwrite=overwrite,
            run_plot=run_plot,
            timeout_sec=timeout_sec,
        )
        for dataset in datasets
    ]
    write_reports(reports, REPO_ROOT / "tests")
    failures = [report for report in reports if not report.ok]
    assert not failures, json.dumps([asdict(report) for report in failures], indent=2)


if __name__ == "__main__":
    raise SystemExit(main())
