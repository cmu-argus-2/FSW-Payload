import glob
import json
import os
import shutil
import struct
import subprocess
import threading
import time
import zlib
from datetime import datetime

from thread_shared import log, state_manager


class TelemetryThread(threading.Thread):
    """Continuously collects payload telemetry and keeps the latest snapshot."""

    FILE_MAGIC = b"TLM1"

    def __init__(
        self,
        stop_event: threading.Event,
        interval_s: float = 1.0,
        output_path: str = "telemetry_logs",
        compression_level: int = 3,
    ):
        super().__init__(name="TelemetryThread", daemon=True)
        self.stop_event = stop_event
        self.interval_s = interval_s
        self.output_path = output_path
        self.compression_level = max(0, min(9, compression_level))
        self._lock = threading.Lock()
        self._latest_data: dict = {}
        self._prev_cpu_times: list[tuple[int, int]] = []
        self._writer = None
        self.current_output_path: str | None = None

    def run(self):
        self._open_log_file()
        log.info("Telemetry thread started")
        try:
            while not self.stop_event.is_set():
                try:
                    data = self._collect_system_info()
                    with self._lock:
                        self._latest_data = data
                    self._write_record(data)
                except Exception as exc:
                    log.warning("Telemetry collection failed: %s", exc)

                self.stop_event.wait(self.interval_s)
        finally:
            self._close_log_file()

    def get_latest_data(self) -> dict:
        """Return a copy of the latest telemetry dictionary."""
        with self._lock:
            return dict(self._latest_data)

    def _open_log_file(self):
        try:
            self.current_output_path = self._build_run_filename()
            os.makedirs(os.path.dirname(self.current_output_path) or ".", exist_ok=True)
            self._writer = open(self.current_output_path, "wb")
            self._writer.write(self.FILE_MAGIC)
            self._writer.flush()
            log.info("Telemetry log file: %s", self.current_output_path)
        except OSError as exc:
            log.warning("Failed to open telemetry log file %s: %s", self.output_path, exc)
            self._writer = None

    def _build_run_filename(self) -> str:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # If a directory is given, create a default timestamped file in it.
        if os.path.isdir(self.output_path) or not self.output_path.lower().endswith(".tlm"):
            directory = self.output_path
            return os.path.join(directory, f"payload_tm_{ts}.tlm")

        # If a file path is given, insert timestamp before extension.
        directory = os.path.dirname(self.output_path) or "."
        base_name = os.path.basename(self.output_path)
        stem, ext = os.path.splitext(base_name)
        return os.path.join(directory, f"{stem}_{ts}{ext}")

    def _close_log_file(self):
        if self._writer is None:
            return
        try:
            self._writer.flush()
            self._writer.close()
        except OSError:
            pass
        finally:
            self._writer = None

    def _write_record(self, data: dict):
        if self._writer is None:
            return

        raw = json.dumps(data, separators=(",", ":"), sort_keys=True).encode("utf-8")
        payload = zlib.compress(raw, level=self.compression_level)
        # Record format: [len:4 bytes big-endian][zlib payload]
        frame = struct.pack(">I", len(payload)) + payload

        try:
            self._writer.write(frame)
            self._writer.flush()
        except OSError as exc:
            log.warning("Telemetry log write failed: %s", exc)

    def _collect_system_info(self) -> dict:
        system_time = int(time.time())
        system_uptime = self._read_system_uptime_s()

        disk = shutil.disk_usage("/")
        disk_usage_pct = int((disk.used / disk.total) * 100) if disk.total else 0

        mem_total, mem_avail, swap_total, swap_free = self._read_mem_swap_kb()
        ram_usage = int(((mem_total - mem_avail) / mem_total) * 100) if mem_total else 0
        swap_usage = int(((swap_total - swap_free) / swap_total) * 100) if swap_total else 0

        cpu_loads = self._read_per_core_load_percent(max_cores=6)
        active_cores = sum(1 for load in cpu_loads if load > 0)

        tegra_values = self._read_tegrastats_values()
        cpu_temp = self._read_temp_c("cpu", fallback=tegra_values.get("CPU_TEMP", 0))
        gpu_temp = self._read_temp_c("gpu", fallback=tegra_values.get("GPU_TEMP", 0))
        gpu_freq = self._read_gpu_freq_mhz(fallback=tegra_values.get("GPU_FREQ", 0))

        payload_state = state_manager.get()

        return {
            "SYSTEM_TIME": system_time,
            "SYSTEM_UPTIME": system_uptime,
            "DISK_USAGE": max(0, min(100, disk_usage_pct)),
            "TEGRASTATS_PROCESS_STATUS": self._is_tegrastats_running(),
            "RAM_USAGE": max(0, min(100, ram_usage)),
            "SWAP_USAGE": max(0, min(100, swap_usage)),
            "ACTIVE_CORES": active_cores,
            "CPU_LOAD_0": cpu_loads[0],
            "CPU_LOAD_1": cpu_loads[1],
            "CPU_LOAD_2": cpu_loads[2],
            "CPU_LOAD_3": cpu_loads[3],
            "CPU_LOAD_4": cpu_loads[4],
            "CPU_LOAD_5": cpu_loads[5],
            "GPU_FREQ": gpu_freq,
            "CPU_TEMP": cpu_temp,
            "GPU_TEMP": gpu_temp,
            "VDD_IN": tegra_values.get("VDD_IN", 0),
            "VDD_CPU_GPU_CV": tegra_values.get("VDD_CPU_GPU_CV", 0),
            "VDD_SOC": tegra_values.get("VDD_SOC", 0),
            "PD_STATE_JETSON": payload_state,
        }

    def _read_system_uptime_s(self) -> int:
        try:
            with open("/proc/uptime", "r", encoding="utf-8") as f:
                return int(float(f.read().split()[0]))
        except (OSError, ValueError, IndexError):
            return 0

    def _read_mem_swap_kb(self) -> tuple[int, int, int, int]:
        meminfo: dict[str, int] = {}
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if ":" not in line:
                        continue
                    key, raw = line.split(":", 1)
                    meminfo[key] = int(raw.strip().split()[0])
        except (OSError, ValueError, IndexError):
            return (0, 0, 0, 0)

        return (
            meminfo.get("MemTotal", 0),
            meminfo.get("MemAvailable", 0),
            meminfo.get("SwapTotal", 0),
            meminfo.get("SwapFree", 0),
        )

    def _read_per_core_load_percent(self, max_cores: int = 6) -> list[int]:
        cpu_lines: list[list[str]] = []
        try:
            with open("/proc/stat", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("cpu") and len(line) > 3 and line[3].isdigit():
                        cpu_lines.append(line.split())
        except OSError:
            return [0] * max_cores

        loads: list[int] = []
        new_prev: list[tuple[int, int]] = []

        for idx in range(min(max_cores, len(cpu_lines))):
            parts = cpu_lines[idx]
            values = [int(v) for v in parts[1:]]
            idle = values[3] + (values[4] if len(values) > 4 else 0)
            total = sum(values)
            new_prev.append((idle, total))

            if idx >= len(self._prev_cpu_times):
                loads.append(0)
                continue

            prev_idle, prev_total = self._prev_cpu_times[idx]
            delta_total = total - prev_total
            delta_idle = idle - prev_idle
            if delta_total <= 0:
                loads.append(0)
                continue

            usage = int((1.0 - (delta_idle / delta_total)) * 100)
            loads.append(max(0, min(100, usage)))

        self._prev_cpu_times = new_prev

        while len(loads) < max_cores:
            loads.append(0)

        return loads

    def _read_temp_c(self, needle: str, fallback: int = 0) -> int:
        zone_paths = glob.glob("/sys/class/thermal/thermal_zone*/type")
        for type_path in zone_paths:
            try:
                with open(type_path, "r", encoding="utf-8") as f:
                    zone_type = f.read().strip().lower()
                if needle not in zone_type:
                    continue

                temp_path = type_path.rsplit("/", 1)[0] + "/temp"
                with open(temp_path, "r", encoding="utf-8") as f:
                    raw = int(f.read().strip())
                if raw > 1000:
                    return int(raw / 1000)
                return int(raw)
            except (OSError, ValueError):
                continue

        return int(fallback)

    def _read_gpu_freq_mhz(self, fallback: int = 0) -> int:
        candidates = [
            "/sys/devices/17000000.gv11b/devfreq/17000000.gv11b/cur_freq",
            "/sys/class/devfreq/17000000.gv11b/cur_freq",
        ]
        candidates.extend(glob.glob("/sys/class/devfreq/*gpu*/cur_freq"))
        candidates.extend(glob.glob("/sys/class/devfreq/*gv11b*/cur_freq"))

        for path in candidates:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    raw_hz = int(f.read().strip())
                return int(raw_hz / 1_000_000)
            except (OSError, ValueError):
                continue

        return int(fallback)

    def _is_tegrastats_running(self) -> int:
        try:
            result = subprocess.run(
                ["pgrep", "-x", "tegrastats"],
                capture_output=True,
                text=True,
                timeout=0.3,
                check=False,
            )
            return 1 if result.returncode == 0 else 0
        except (OSError, subprocess.SubprocessError):
            return 0

    def _read_tegrastats_values(self) -> dict:
        default = {
            "GPU_FREQ": 0,
            "CPU_TEMP": 0,
            "GPU_TEMP": 0,
            "VDD_IN": 0,
            "VDD_CPU_GPU_CV": 0,
            "VDD_SOC": 0,
        }

        try:
            proc = subprocess.Popen(["tegrastats"], stdout=subprocess.PIPE, text=True)
            output = proc.stdout.readline()
            proc.kill()
        except (OSError, subprocess.SubprocessError):
            print("Error occurred while running tegrastats")
            return default
        
        # Result: 03-09-2026 14:12:52 RAM 1905/7620MB (lfb 36x4MB) SWAP 320/12002MB (cached 20MB) CPU [0%@729,0%@729,1%@729,0%@729,0%@729,0%@729] GR3D_FREQ 0% cpu@46.125C soc2@45.843C soc0@44.156C gpu@45.062C tj@46.125C soc1@46C VDD_IN 4742mW/4742mW VDD_CPU_GPU_CV 482mW/482mW VDD_SOC 1446mW/1446mW


        
        if not output.strip():
            return default

        values = dict(default)
        output_lower = output.lower()
        values["GPU_FREQ"] = self._extract_number_before(output_lower, "gr3d_freq", "%")
        values["CPU_TEMP"] = self._extract_number_before(output_lower, "cpu@", "c")
        values["GPU_TEMP"] = self._extract_number_before(output_lower, "gpu@", "c")

        values["VDD_IN"] = self._extract_power_mw(output_lower, "vdd_in")
        values["VDD_CPU_GPU_CV"] = self._extract_power_mw(output_lower, "vdd_cpu_gpu_cv")
        values["VDD_SOC"] = self._extract_power_mw(output_lower, "vdd_soc")

        return values

    def _extract_power_mw(self, text: str, key: str) -> int:
        idx = text.find(key)
        if idx < 0:
            return 0
        snippet = text[idx : idx + 80]
        marker = "mw"
        marker_idx = snippet.find(marker)
        if marker_idx < 0:
            return 0

        j = marker_idx - 1
        digits = []
        while j >= 0 and snippet[j].isdigit():
            digits.append(snippet[j])
            j -= 1

        if not digits:
            return 0

        return int("".join(reversed(digits)))

    def _extract_number_before(self, text: str, key: str, stop_char: str) -> int:
        idx = text.find(key)
        if idx < 0:
            return 0

        start = idx + len(key)
        while start < len(text) and (text[start] in " :=["):
            start += 1

        end = start
        while end < len(text) and text[end] != stop_char:
            end += 1

        token = text[start:end]

        num_chars = []
        for ch in token:
            if ch.isdigit() or ch == ".":
                num_chars.append(ch)
        if not num_chars:
            return 0

        try:
            return int(float("".join(num_chars)))
        except ValueError:
            return 0
