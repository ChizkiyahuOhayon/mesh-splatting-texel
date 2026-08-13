import unittest
from unittest import mock

import gorfe_v1_resources as resources


class ResourceMonitorTest(unittest.TestCase):
    def test_gpu_row_parser_is_strict(self):
        with mock.patch.object(
            resources.subprocess, "check_output", return_value="123, 456\n"
        ):
            self.assertEqual(
                resources.query_gpu_memory(3), {"used_mib": 123, "free_mib": 456}
            )
        with mock.patch.object(
            resources.subprocess, "check_output", return_value="not,a,number\n"
        ):
            with self.assertRaises(RuntimeError):
                resources.query_gpu_memory(3)

    def test_compute_process_parser_is_strict(self):
        with mock.patch.object(
            resources.subprocess,
            "check_output",
            return_value="17, python, 1482\n23, worker, 512\n",
        ):
            self.assertEqual(
                resources.query_gpu_compute_processes(3),
                (
                    {"pid": 17, "name": "python", "used_mib": 1482},
                    {"pid": 23, "name": "worker", "used_mib": 512},
                ),
            )
        with mock.patch.object(resources.subprocess, "check_output", return_value="\n"):
            self.assertEqual(resources.query_gpu_compute_processes(3), ())
        with mock.patch.object(
            resources.subprocess, "check_output", return_value="python, 1482\n"
        ):
            with self.assertRaises(RuntimeError):
                resources.query_gpu_compute_processes(3)

    def test_monitor_records_peak_and_minimum(self):
        values = iter(
            [
                {"used_mib": 5, "free_mib": 95},
                {"used_mib": 8, "free_mib": 92},
                {"used_mib": 6, "free_mib": 94},
            ]
        )
        monitor = resources.PhysicalGpuMonitor(3, interval_seconds=100)
        with mock.patch.object(resources, "query_gpu_memory", side_effect=lambda _: next(values)):
            monitor._sample()
            monitor._sample()
            monitor._sample()
        self.assertEqual(monitor.peak_used_mib, 8)
        self.assertEqual(monitor.minimum_free_mib, 92)
        self.assertEqual(monitor.samples, 3)

    def test_monitor_refuses_a_foreign_compute_process(self):
        monitor = resources.PhysicalGpuMonitor(3, exclusive_pid=17)
        with mock.patch.object(
            resources, "query_gpu_memory", return_value={"used_mib": 8, "free_mib": 92}
        ), mock.patch.object(
            resources,
            "query_gpu_compute_processes",
            return_value=({"pid": 23, "name": "python", "used_mib": 1},),
        ):
            with self.assertRaisesRegex(resources.ResourceMonitorError, "PIDs.*23"):
                monitor._sample()


if __name__ == "__main__":
    unittest.main()
