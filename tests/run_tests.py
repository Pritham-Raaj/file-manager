"""
Convenience test runner.

Runs the full mock-based unit test suite and prints a compact summary.

Usage
-----
    # From the file-manager/ directory:
    python tests/run_tests.py

    # With verbose output:
    python tests/run_tests.py -v

    # Run only a specific module:
    python tests/run_tests.py --module indexer

    # Run only a specific class:
    python tests/run_tests.py --module indexer --class TestChangeDetection
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

TESTS_DIR    = Path(__file__).parent
PROJECT_ROOT = TESTS_DIR.parent

MODULES = [
    "test_indexer",
    "test_file_manager",
    "test_writer",
    "test_summarizer",
    "test_orchestrator",
    "test_sync",
]


def build_pytest_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [sys.executable, "-m", "pytest", "--tb=short", "-q"]

    if args.verbose:
        cmd.append("-v")
    if args.coverage:
        cmd += ["--cov=.", "--cov-report=term-missing", "--cov-config=.coveragerc"]

    if args.module:
        target = TESTS_DIR / f"test_{args.module}.py"
        if args.cls:
            cmd.append(f"{target}::{args.cls}")
        else:
            cmd.append(str(target))
    else:
        for mod in MODULES:
            cmd.append(str(TESTS_DIR / f"{mod}.py"))

    return cmd


def main():
    parser = argparse.ArgumentParser(description="Run file-manager unit tests")
    parser.add_argument("-v", "--verbose",  action="store_true")
    parser.add_argument("--module",  metavar="NAME",
                        help="Module to test (e.g. indexer, writer, sync)")
    parser.add_argument("--class",   dest="cls", metavar="CLASS",
                        help="Specific test class within --module")
    parser.add_argument("--coverage", action="store_true",
                        help="Generate coverage report (requires pytest-cov)")
    args = parser.parse_args()

    cmd = build_pytest_cmd(args)
    print("Running:", " ".join(cmd), "\n")

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
