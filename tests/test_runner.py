"""Tests for scripts/runner.py — stdlib only, Python toolchain only.

Java, Kotlin and TypeScript paths are skipped when their toolchains are absent so
the suite stays runnable on any machine.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "runner.py"

SOLUTION = "def allocate(total, parts):\n    return [total // parts] * parts\n"
PASSING_TEST = 'from solution import allocate\n\nassert allocate(10, 2) == [5, 5]\nprint("PASS")\n'
FAILING_TEST = "from solution import allocate\n\nassert allocate(10, 2) == [4, 6]\n"


class PythonRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = tempfile.TemporaryDirectory()
        self.addCleanup(self.workspace.cleanup)
        self.root = Path(self.workspace.name)
        (self.root / "solution.py").write_text(SOLUTION, encoding="utf-8")

    def run_challenge(self, test_source: str, *extra: str) -> subprocess.CompletedProcess:
        test_file = self.root / "challenge_test.py"
        test_file.write_text(test_source, encoding="utf-8")
        return subprocess.run(
            [
                sys.executable,
                str(RUNNER),
                "run",
                "--language",
                "python",
                "--solution",
                str(self.root / "solution.py"),
                "--tests",
                str(test_file),
                "--timeout",
                "5",
                *extra,
            ],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_passing_tests_exit_zero(self) -> None:
        result = self.run_challenge(PASSING_TEST)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("status=passed", result.stdout)

    def test_failing_tests_exit_nonzero(self) -> None:
        result = self.run_challenge(FAILING_TEST)
        self.assertEqual(result.returncode, 1)
        self.assertIn("status=failed", result.stdout)

    def test_json_output_is_machine_readable(self) -> None:
        result = self.run_challenge(PASSING_TEST, "--json")
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "passed")
        self.assertEqual(payload["phase"], "run")

    def test_cpu_bound_hang_is_stopped(self) -> None:
        result = self.run_challenge("while True:\n    pass\n")
        self.assertEqual(result.returncode, 1)
        self.assertIn("status=timeout", result.stdout)

    def test_workspace_path_is_not_leaked(self) -> None:
        result = self.run_challenge(FAILING_TEST)
        self.assertNotIn("coding-reasoning-", result.stdout)

    def test_ansi_escapes_are_stripped(self) -> None:
        ansi_test = 'print("\\033[31mRed Text\\033[0m")\nassert True\n'
        result = self.run_challenge(ansi_test, "--json")
        self.assertEqual(result.returncode, 0)
        payload = json.loads(result.stdout)
        self.assertNotIn("\033[", payload["stdout"])
        self.assertIn("Red Text", payload["stdout"])

    def test_sleep_timeout_is_stopped(self) -> None:
        result = self.run_challenge("import time\ntime.sleep(30)\n", "--timeout", "1")
        self.assertEqual(result.returncode, 1)
        self.assertIn("status=timeout", result.stdout)

    def test_child_process_timeout_is_stopped(self) -> None:
        child_test = (
            "import subprocess, sys\n"
            "subprocess.run([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        )
        result = self.run_challenge(child_test, "--timeout", "1")
        self.assertEqual(result.returncode, 1)
        self.assertIn("status=timeout", result.stdout)

    def spawn_background_child(self, detach_output: bool) -> str:
        pid_file = self.root / "child.pid"
        redirect = ", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL" if detach_output else ""
        return (
            "import pathlib, subprocess, sys\n"
            "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']"
            f"{redirect})\n"
            f"pathlib.Path({str(pid_file)!r}).write_text(str(child.pid))\n"
            'print("PASS")\n'
        )

    def assert_child_is_gone(self) -> None:
        pid = int((self.root / "child.pid").read_text())
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except (ProcessLookupError, PermissionError):
                # PermissionError: the pid now belongs to another user's process.
                return
            time.sleep(0.05)
        os.kill(pid, signal.SIGKILL)
        self.fail(f"background child {pid} outlived the runner")

    @unittest.skipUnless(os.name == "posix", "process groups are POSIX-only")
    def test_background_child_holding_output_is_killed_on_timeout(self) -> None:
        result = self.run_challenge(self.spawn_background_child(False), "--timeout", "1")
        # Check the child first so a failed status assertion cannot leak it.
        self.assert_child_is_gone()
        self.assertIn("status=timeout", result.stdout)

    @unittest.skipUnless(os.name == "posix", "process groups are POSIX-only")
    def test_detached_background_child_is_killed_after_run(self) -> None:
        result = self.run_challenge(self.spawn_background_child(True), "--timeout", "1")
        # Check the child first so a failed status assertion cannot leak it.
        self.assert_child_is_gone()
        self.assertIn("status=passed", result.stdout)

    def test_support_files_are_copied_and_accessible(self) -> None:
        support_file = self.root / "helper.py"
        support_file.write_text("HELPER_VALUE = 42\n", encoding="utf-8")
        test_source = (
            "from solution import allocate\n"
            "from helper import HELPER_VALUE\n"
            "assert HELPER_VALUE == 42\n"
            "assert allocate(10, 2) == [5, 5]\n"
        )
        result = self.run_challenge(test_source, "--support", str(support_file))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("status=passed", result.stdout)


class ArgumentValidationTest(unittest.TestCase):
    def run_runner(self, *argv: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(RUNNER), *argv], capture_output=True, text=True, check=False
        )

    def test_unsupported_language_is_rejected(self) -> None:
        result = self.run_runner(
            "run", "--language", "cobol", "--solution", "s.cob", "--tests", "t.cob"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unsupported language", result.stderr)

    def test_timeout_outside_bounds_is_rejected(self) -> None:
        result = self.run_runner(
            "run",
            "--language",
            "python",
            "--solution",
            "s.py",
            "--tests",
            "t.py",
            "--timeout",
            "600",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--timeout must be between 1 and 60", result.stderr)

    def test_memory_outside_bounds_is_rejected(self) -> None:
        result = self.run_runner(
            "run",
            "--language",
            "python",
            "--solution",
            "s.py",
            "--tests",
            "t.py",
            "--memory-mb",
            "16",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--memory-mb must be between 64 and 4096", result.stderr)

    def test_missing_source_file_is_rejected(self) -> None:
        result = self.run_runner(
            "run",
            "--language",
            "python",
            "--solution",
            "nonexistent_solution.py",
            "--tests",
            "t.py",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Source file not found", result.stderr)

    def test_doctor_reports_every_language(self) -> None:
        result = self.run_runner("doctor")
        self.assertEqual(result.returncode, 0)
        for language in ("python", "typescript", "java", "kotlin"):
            self.assertIn(language, result.stdout)


if __name__ == "__main__":
    unittest.main()
