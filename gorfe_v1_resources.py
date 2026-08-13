"""Small resource monitor for preregistered GoRFE-V1 validity limits."""

import resource
import subprocess
import sys
import threading


class ResourceMonitorError(RuntimeError):
    """Platform monitoring failed or the locked GPU ceased to be exclusive."""


def host_peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS reports bytes.
    return int(value if sys.platform == "darwin" else value * 1024)


def query_gpu_memory(physical_gpu):
    output = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={physical_gpu}",
            "--query-gpu=memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()
    rows = [row.strip() for row in output.splitlines() if row.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"unexpected nvidia-smi output: {output!r}")
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 2 or any(not field.isdigit() for field in fields):
        raise RuntimeError(f"malformed nvidia-smi memory row: {rows[0]!r}")
    return {"used_mib": int(fields[0]), "free_mib": int(fields[1])}


def query_gpu_compute_processes(physical_gpu):
    output = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={physical_gpu}",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()
    processes = []
    for row in output.splitlines():
        row = row.strip()
        if not row:
            continue
        fields = [field.strip() for field in row.split(",", 2)]
        if len(fields) != 3 or not fields[0].isdigit() or not fields[2].isdigit():
            raise RuntimeError(f"malformed nvidia-smi process row: {row!r}")
        processes.append(
            {"pid": int(fields[0]), "name": fields[1], "used_mib": int(fields[2])}
        )
    return tuple(processes)


class PhysicalGpuMonitor:
    def __init__(self, physical_gpu, *, interval_seconds=1.0, exclusive_pid=None):
        if not isinstance(physical_gpu, int) or physical_gpu < 0:
            raise ValueError("physical_gpu must be a nonnegative integer")
        self.physical_gpu = physical_gpu
        self.interval_seconds = float(interval_seconds)
        if self.interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        if exclusive_pid is not None and (
            not isinstance(exclusive_pid, int) or exclusive_pid < 1
        ):
            raise ValueError("exclusive_pid must be a positive integer")
        self.exclusive_pid = exclusive_pid
        self.peak_used_mib = 0
        self.minimum_free_mib = None
        self.samples = 0
        self.error = None
        self.observed_compute_pids = set()
        self._stop = threading.Event()
        self._thread = None

    def _sample(self):
        value = query_gpu_memory(self.physical_gpu)
        if self.exclusive_pid is not None:
            processes = query_gpu_compute_processes(self.physical_gpu)
            pids = {process["pid"] for process in processes}
            self.observed_compute_pids.update(pids)
            unexpected = sorted(pids - {self.exclusive_pid})
            if unexpected:
                raise ResourceMonitorError(
                    f"physical GPU {self.physical_gpu} was shared by PIDs {unexpected}"
                )
        self.peak_used_mib = max(self.peak_used_mib, value["used_mib"])
        self.minimum_free_mib = (
            value["free_mib"]
            if self.minimum_free_mib is None
            else min(self.minimum_free_mib, value["free_mib"])
        )
        self.samples += 1

    def _run(self):
        while not self._stop.is_set():
            try:
                self._sample()
            except Exception as error:  # a missing sample invalidates provenance
                self.error = f"{type(error).__name__}: {error}"
                self._stop.set()
                return
            self._stop.wait(self.interval_seconds)

    def __enter__(self):
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._stop.set()
        self._thread.join()
        if self.error is None:
            self._sample()

    def record(self):
        if self.error is not None:
            raise ResourceMonitorError(f"physical GPU monitor failed: {self.error}")
        if self.samples < 2:
            raise ResourceMonitorError("physical GPU monitor produced too few samples")
        return {
            "physical_gpu": self.physical_gpu,
            "poll_interval_seconds": self.interval_seconds,
            "samples": self.samples,
            "peak_used_mib": self.peak_used_mib,
            "minimum_free_mib": self.minimum_free_mib,
            "exclusive_pid": self.exclusive_pid,
            "observed_compute_pids": sorted(self.observed_compute_pids),
        }
