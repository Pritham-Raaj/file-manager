"""
Benchmark runner for the file-manager.

Measures four dimensions against a real Ollama backend:
  - Latency   : wall-clock time for index, search, first-token, full summarize
  - Memory    : RSS before/after each operation (psutil)
  - CPU       : average CPU% sampled during each operation (threading)
  - Indexing efficiency : re-index skip rate (unchanged files must return 0)

Usage
-----
    # From the file-manager/ directory:
    python tests/benchmarks/benchmark_runner.py --root /tmp/bench_files

    # Generate synthetic files first:
    python tests/benchmarks/benchmark_runner.py --generate --root /tmp/bench_files

Output
------
    benchmark_results.json  (machine-readable)
    benchmark_report.txt    (human-readable summary)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Any

# ── make project root importable ────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import psutil
except ImportError:
    print("psutil not installed. Run: pip install psutil --break-system-packages")
    sys.exit(1)

import config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _current_rss_mb() -> float:
    """Return RSS memory of this process in megabytes."""
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss / (1024 ** 2)


class _CpuSampler:
    """Background thread that averages CPU% over the duration of an operation."""

    def __init__(self, interval: float = 0.1):
        self._interval = interval
        self._samples:  list[float] = []
        self._stop      = threading.Event()
        self._proc      = psutil.Process(os.getpid())

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> float:
        self._stop.set()
        self._thread.join()
        return sum(self._samples) / len(self._samples) if self._samples else 0.0

    def _run(self):
        while not self._stop.is_set():
            try:
                self._samples.append(self._proc.cpu_percent(interval=None))
            except Exception:
                pass
            time.sleep(self._interval)


def measure(label: str, fn: Callable[[], Any]) -> dict:
    """
    Run fn() and return a metrics dict:
      { label, duration_s, mem_before_mb, mem_after_mb, mem_delta_mb, avg_cpu_pct, result }
    """
    mem_before = _current_rss_mb()
    cpu        = _CpuSampler()
    cpu.start()
    t0         = time.perf_counter()

    result     = fn()

    duration   = time.perf_counter() - t0
    avg_cpu    = cpu.stop()
    mem_after  = _current_rss_mb()

    metrics = {
        "label":        label,
        "duration_s":   round(duration, 4),
        "mem_before_mb": round(mem_before, 2),
        "mem_after_mb":  round(mem_after, 2),
        "mem_delta_mb":  round(mem_after - mem_before, 2),
        "avg_cpu_pct":   round(avg_cpu, 2),
        "result":        str(result)[:200],   # truncate for readability
    }
    print(f"  [{label}] {duration:.3f}s | ΔRAM {metrics['mem_delta_mb']:+.1f} MB "
          f"| CPU {avg_cpu:.1f}%")
    return metrics


# ---------------------------------------------------------------------------
# Synthetic file generation
# ---------------------------------------------------------------------------

SHORT_CONTENT = (
    "The patient was diagnosed with hypertension in January 2024. "
    "Blood pressure readings averaged 145/92 mmHg. "
    "Treatment includes beta-blockers and lifestyle modifications.\n"
)

LONG_CONTENT = SHORT_CONTENT * 200   # ~25 KB

CODE_CONTENT = '''\
def compute_bmi(weight_kg, height_m):
    return round(weight_kg / height_m ** 2, 2)

def classify_bmi(bmi):
    if bmi < 18.5: return "Underweight"
    elif bmi < 25: return "Normal"
    elif bmi < 30: return "Overweight"
    return "Obese"
''' * 10


def generate_files(root: Path) -> list[Path]:
    """Create a diverse set of synthetic benchmark files in root."""
    root.mkdir(parents=True, exist_ok=True)
    files = []

    specs = [
        ("short_note.txt",  SHORT_CONTENT),
        ("long_report.txt", LONG_CONTENT),
        ("analysis.md",     f"# Analysis\n\n{SHORT_CONTENT * 5}"),
        ("patient_data.csv","patient_id,age,bp\n" + "P001,45,145\n" * 50),
        ("utils.py",        CODE_CONTENT),
    ]

    for name, content in specs:
        p = root / name
        p.write_text(content, encoding="utf-8")
        files.append(p)
        print(f"  Generated {p.name} ({p.stat().st_size:,} bytes)")

    return files


# ---------------------------------------------------------------------------
# Benchmark suites
# ---------------------------------------------------------------------------

def bench_indexing(indexer, files: list[Path]) -> list[dict]:
    results = []
    print("\n=== Indexing benchmarks ===")
    for f in files:
        m = measure(f"index:{f.name}", lambda p=f: indexer.index_file(p))
        results.append(m)
    return results


def bench_reindex_efficiency(indexer, files: list[Path]) -> dict:
    """Measure skip rate: unchanged files must return 0 chunks (no reprocessing)."""
    print("\n=== Re-index efficiency ===")
    skipped = 0
    for f in files:
        count = indexer.index_file(f)   # second call -- should skip
        if count == 0:
            skipped += 1
    skip_rate = skipped / len(files) if files else 0
    print(f"  Skip rate: {skipped}/{len(files)} = {skip_rate:.0%}")
    return {"skip_rate": skip_rate, "skipped": skipped, "total": len(files)}


def bench_search(indexer) -> list[dict]:
    queries = [
        "hypertension blood pressure treatment",
        "BMI calculation body mass index",
        "patient diagnosis age",
        "beta blockers lifestyle",
        "CSV patient data",
    ]
    results = []
    print("\n=== Search benchmarks ===")
    for q in queries:
        m = measure(f"search:{q[:30]}", lambda q=q: indexer.search(q, n_results=5))
        results.append(m)
    return results


def bench_summarize(summarizer, files: list[Path]) -> list[dict]:
    results = []
    print("\n=== Summarization benchmarks ===")
    for f in files:
        text = f.read_text(encoding="utf-8")
        m = measure(f"summarize:{f.name}", lambda t=text: list(summarizer.summarize(t)))
        results.append(m)
    return results


def bench_first_token(llm) -> dict:
    """Time from generate() call to first yielded token — perceived latency."""
    print("\n=== First-token latency ===")
    prompt = "Summarize: " + SHORT_CONTENT

    t0 = time.perf_counter()
    gen = llm.generate(prompt)
    _ = next(gen)   # first token
    ttft = time.perf_counter() - t0

    # Drain remaining tokens
    for _ in gen:
        pass

    print(f"  Time-to-first-token: {ttft:.3f}s")
    return {"label": "first_token_latency", "duration_s": round(ttft, 4)}


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_report(all_results: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "benchmark_results.json"
    json_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    lines = ["=" * 60, "FILE-MANAGER BENCHMARK REPORT", "=" * 60, ""]

    def section(title: str, rows: list[dict], key_fields: list[str]):
        lines.append(f"## {title}")
        for r in rows:
            parts = [f"  {r.get('label', '?')}"]
            for k in key_fields:
                if k in r:
                    parts.append(f"{k}={r[k]}")
            lines.append("  " + " | ".join(parts))
        lines.append("")

    section("Indexing",     all_results.get("indexing",    []),
            ["duration_s", "mem_delta_mb", "avg_cpu_pct"])
    section("Search",       all_results.get("search",      []),
            ["duration_s", "mem_delta_mb", "avg_cpu_pct"])
    section("Summarization",all_results.get("summarize",   []),
            ["duration_s", "mem_delta_mb", "avg_cpu_pct"])

    eff = all_results.get("reindex_efficiency", {})
    lines.append("## Re-index Efficiency")
    lines.append(f"  Skip rate: {eff.get('skip_rate', 0):.0%} "
                 f"({eff.get('skipped')}/{eff.get('total')} files skipped)")
    lines.append("")

    ttft = all_results.get("first_token", {})
    lines.append("## First-Token Latency")
    lines.append(f"  {ttft.get('duration_s', 'N/A')} seconds")
    lines.append("")

    report_path = out_dir / "benchmark_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nResults saved to:\n  {json_path}\n  {report_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="File-manager benchmark suite")
    parser.add_argument("--root",     default="/tmp/bench_files",
                        help="Directory with (or for) benchmark files")
    parser.add_argument("--generate", action="store_true",
                        help="Generate synthetic test files before benchmarking")
    parser.add_argument("--out",      default="benchmark_output",
                        help="Output directory for results")
    args = parser.parse_args()

    root = Path(args.root)

    # Redirect config so the real ChromaDB is used in a temp location
    config.ROOT_DIR        = root
    config.CHROMA_DIR      = root / ".chroma"
    config.INDEX_MANIFEST  = config.CHROMA_DIR / "manifest.json"

    from llm_client  import LLMClient
    from indexer     import Indexer
    from summarizer  import Summarizer

    llm       = LLMClient()
    indexer   = Indexer(llm=llm)
    summarizer = Summarizer(llm=llm)

    if args.generate:
        print("Generating synthetic benchmark files...")
        files = generate_files(root)
    else:
        files = [p for p in root.rglob("*") if p.is_file() and not p.name.startswith(".")]
        if not files:
            print(f"No files found in {root}. Use --generate to create them.")
            sys.exit(1)

    print(f"\nBenchmarking {len(files)} file(s) in {root}\n")

    all_results = {}

    idx_results             = bench_indexing(indexer, files)
    all_results["indexing"] = idx_results

    all_results["reindex_efficiency"] = bench_reindex_efficiency(indexer, files)

    all_results["search"]   = bench_search(indexer)
    all_results["summarize"] = bench_summarize(summarizer, files)

    try:
        all_results["first_token"] = bench_first_token(llm)
    except Exception as exc:
        print(f"  first_token benchmark skipped: {exc}")

    write_report(all_results, Path(args.out))


if __name__ == "__main__":
    main()
