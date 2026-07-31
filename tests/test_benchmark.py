# -*- coding: utf-8 -*-
"""
Benchmark tests for SarcAsM analysis speed.

Measures wall-clock time (and RSS memory delta, when `psutil` is available)
for the structure, motion, and domain-motion pipelines. Each run is saved as
a timestamped JSON file in ``tests/benchmark_results/`` and can be summarized
with ``TestBenchmarkSummary``.

Usage
-----
uv run pytest tests/test_benchmark.py -v                        # all
uv run pytest tests/test_benchmark.py::TestStructureBenchmark -v
uv run pytest tests/test_benchmark.py::TestBenchmarkSummary -v  # report only
uv run pytest tests/test_benchmark.py -m "not slow" -v          # skip slow
"""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from sarcasm import Motion, SarcAsM
from sarcasm._version import __version__ as sarcasm_version
from sarcasm.utils import Utils

logger = logging.getLogger(__name__)


try:
    import psutil
    _PROC: Optional["psutil.Process"] = psutil.Process()
except ImportError:  # pragma: no cover
    _PROC = None


def _current_rss_mb() -> Optional[float]:
    if _PROC is None:
        return None
    return _PROC.memory_info().rss / (1024 * 1024)


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True, capture_output=True, text=True, timeout=2,
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def _gpu_name() -> Optional[str]:
    try:
        import torch
    except ImportError:
        return None
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "Apple MPS"
    return None


def _system_metadata(test_file: str) -> Dict[str, Any]:
    """Standard system metadata attached to every benchmark run."""
    meta: Dict[str, Any] = {
        "sarcasm_version": sarcasm_version,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "device": str(Utils.get_device()),
        "gpu_name": _gpu_name(),
        "git_commit": _git_commit(),
        "test_file_size_mb": round(os.path.getsize(test_file) / (1024 * 1024), 2),
    }
    return {k: v for k, v in meta.items() if v is not None}


@dataclass
class BenchmarkResult:
    """Timing + memory data for a single benchmark run."""

    name: str
    test_file: str
    substeps: Dict[str, float] = field(default_factory=dict)
    mem_delta_mb: Dict[str, float] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def for_test(cls, name: str, test_file: str) -> "BenchmarkResult":
        """Build a result with standard system metadata pre-populated."""
        result = cls(name=name, test_file=str(test_file))
        result.metadata.update(_system_metadata(str(test_file)))
        return result

    def record_step(self, name: str, duration: float, mem_delta_mb: Optional[float] = None) -> None:
        self.substeps[name] = duration
        if mem_delta_mb is not None:
            self.mem_delta_mb[name] = mem_delta_mb

    def add_metadata(self, key: str, value: Any) -> None:
        self.metadata[key] = value

    @property
    def total_time(self) -> float:
        return sum(self.substeps.values())

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "test_file": self.test_file,
            "timestamp": self.start_time.isoformat(),
            "substeps": self.substeps,
            "mem_delta_mb": self.mem_delta_mb,
            "total_time": self.total_time,
            "metadata": self.metadata,
        }

    def save_json(self, output_dir: Path, prefix: str = "") -> Path:
        stamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        filepath = output_dir / f"{prefix}{self.name}_{stamp}.json"
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return filepath


@contextmanager
def timer(result: BenchmarkResult, step_name: str, log: bool = True):
    """Time a block and record duration + RSS delta (if psutil available)."""
    rss_before = _current_rss_mb()
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        rss_after = _current_rss_mb()
        mem_delta = None
        if rss_before is not None and rss_after is not None:
            mem_delta = rss_after - rss_before
        result.record_step(step_name, duration, mem_delta)
        if log:
            mem_str = f"  (Δrss {mem_delta:+.1f} MB)" if mem_delta is not None else ""
            print(f"  {step_name}: {duration:.4f}s{mem_str}")


# ---------------------------------------------------------------------------
# SarcAsM
# ---------------------------------------------------------------------------


class TestStructureBenchmark:
    """Benchmark the structure-analysis pipeline."""

    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_full_structure_pipeline(self, structure_single_file_path, benchmark_output_dir):
        result = BenchmarkResult.for_test("full_structure_pipeline", structure_single_file_path)
        print(f"\n[bench] {result.name} on {os.path.basename(structure_single_file_path)}")

        with timer(result, "full_pipeline"):
            sarc = SarcAsM(structure_single_file_path, restart=True)
            sarc.detect_sarcomeres(frames=0, max_patch_size=(1024, 1024))
            sarc.full_analysis_structure()

        filepath = result.save_json(benchmark_output_dir, prefix="structure_")
        print(f"[bench] saved {filepath.name}  | total {result.total_time:.3f}s")

    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_structure_substeps(self, structure_single_file_path, benchmark_output_dir):
        result = BenchmarkResult.for_test("structure_substeps", structure_single_file_path)
        print(f"\n[bench] {result.name} on {os.path.basename(structure_single_file_path)}")

        sarc = SarcAsM(structure_single_file_path, restart=True)

        with timer(result, "detect_sarcomeres"):
            sarc.detect_sarcomeres(frames=0, max_patch_size=(1024, 1024))
        with timer(result, "analyze_cell_mask"):
            sarc.analyze_cell_mask()
        with timer(result, "analyze_z_bands"):
            sarc.analyze_z_bands(frames=0)
        with timer(result, "analyze_sarcomere_vectors"):
            sarc.analyze_sarcomere_vectors(frames=0)
        with timer(result, "analyze_myofibrils"):
            sarc.analyze_myofibrils(frames=0)
        with timer(result, "analyze_sarcomere_domains"):
            sarc.analyze_sarcomere_domains(frames=0)

        filepath = result.save_json(benchmark_output_dir, prefix="structure_")
        print(f"[bench] saved {filepath.name}  | total {result.total_time:.3f}s")

    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_structure_timelapse_pipeline(self, structure_timelapse_file_path, benchmark_output_dir):
        result = BenchmarkResult.for_test("structure_timelapse_pipeline", structure_timelapse_file_path)
        print(f"\n[bench] {result.name} on {os.path.basename(structure_timelapse_file_path)}")

        with timer(result, "full_pipeline_timelapse"):
            sarc = SarcAsM(structure_timelapse_file_path, restart=True)
            sarc.detect_sarcomeres(frames=0, max_patch_size=(512, 512))
            sarc.full_analysis_structure()

        filepath = result.save_json(benchmark_output_dir, prefix="structure_")
        print(f"[bench] saved {filepath.name}  | total {result.total_time:.3f}s")


# ---------------------------------------------------------------------------
# Motion (LOI)
# ---------------------------------------------------------------------------




class TestDomainMotionBenchmark:
    """Benchmark the domain-motion pipeline."""

    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_domain_motion_pipeline(self, motion_30kPa_file_path, benchmark_output_dir):
        result = BenchmarkResult.for_test("domain_motion_pipeline", motion_30kPa_file_path)
        print(f"\n[bench] {result.name} on {os.path.basename(motion_30kPa_file_path)}")

        # Detection runs on a 50-frame window, so every downstream step must be
        # given the same window: frames="all" would claim all 500 frames of the
        # movie while only 50 have masks, and the tracker rejects the gap.
        bench_frames = list(range(50))

        with timer(result, "full_domain_motion_pipeline"):
            sarc = SarcAsM(motion_30kPa_file_path, restart=True)
            sarc.detect_sarcomeres(frames=bench_frames, max_patch_size=(256, 1024))
            sarc.analyze_sarcomere_vectors(frames=bench_frames, interpolation_method="akima")
            sarc.analyze_sarcomere_domains(frames=0, leiden_resolution=1, store_mask=True)
            sarc.track_sarcomere_vectors(frames=bench_frames)
            sarc.analyze_track_motion(by='domain', reference_frame=0, threshold=0.3, contr_time_min=0.2)

        filepath = result.save_json(benchmark_output_dir, prefix="domain_motion_")
        print(f"[bench] saved {filepath.name}  | total {result.total_time:.3f}s")

    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_domain_motion_substeps(self, motion_30kPa_file_path, benchmark_output_dir):
        result = BenchmarkResult.for_test("domain_motion_substeps", motion_30kPa_file_path)
        print(f"\n[bench] {result.name} on {os.path.basename(motion_30kPa_file_path)}")

        sarc = SarcAsM(motion_30kPa_file_path, restart=True)

        # Same window for every step — see test_domain_motion_pipeline.
        bench_frames = list(range(50))

        with timer(result, "detect_sarcomeres_multi_frame"):
            sarc.detect_sarcomeres(frames=bench_frames, max_patch_size=(256, 1024))
        with timer(result, "analyze_sarcomere_vectors"):
            sarc.analyze_sarcomere_vectors(frames=bench_frames, interpolation_method="akima")
        with timer(result, "analyze_sarcomere_domains"):
            sarc.analyze_sarcomere_domains(frames=0, leiden_resolution=1, store_mask=True)
        with timer(result, "track_sarcomere_vectors"):
            sarc.track_sarcomere_vectors(frames=bench_frames)
        with timer(result, "analyze_track_motion_domain"):
            sarc.analyze_track_motion(by='domain', reference_frame=0, threshold=0.3, contr_time_min=0.2)

        filepath = result.save_json(benchmark_output_dir, prefix="domain_motion_")
        print(f"[bench] saved {filepath.name}  | total {result.total_time:.3f}s")


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------


def _format_entry(data: dict) -> str:
    """Format a single benchmark result as a readable block."""
    meta = data.get("metadata", {})
    substeps = data.get("substeps", {}) or {}
    mem = data.get("mem_delta_mb", {}) or {}
    total = data.get("total_time") or sum(substeps.values())

    lines = []
    lines.append(f"■ {data['name']}   [{data.get('timestamp', '')[:19]}]")
    lines.append(
        f"  file    : {os.path.basename(data.get('test_file', ''))} "
        f"({meta.get('test_file_size_mb', '?')} MB)"
    )
    gpu = f" / {meta['gpu_name']}" if meta.get("gpu_name") else ""
    lines.append(f"  device  : {meta.get('device', '?')}{gpu}")
    lines.append(
        f"  version : sarcasm={meta.get('sarcasm_version', '?')} "
        f"py={meta.get('python_version', '?')} "
        f"git={meta.get('git_commit', '?')}"
    )
    lines.append(f"  total   : {total:.3f}s")

    if substeps:
        name_w = max(len(k) for k in substeps) + 2
        header = (
            f"  {'step'.ljust(name_w)} {'time (s)':>10} "
            f"{'% total':>8} {'Δrss MB':>10}"
        )
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for step, dur in substeps.items():
            pct = (dur / total * 100) if total > 0 else 0.0
            m = mem.get(step)
            m_str = f"{m:>+10.1f}" if m is not None else " " * 10
            lines.append(
                f"  {step.ljust(name_w)} {dur:>10.3f} {pct:>7.1f}% {m_str}"
            )
    return "\n".join(lines)


class TestBenchmarkSummary:
    """Print a summary report of all benchmark JSON files on disk."""

    @pytest.mark.benchmark
    def test_generate_summary(self, benchmark_output_dir):
        json_files = sorted(
            benchmark_output_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        print("\n" + "=" * 72)
        print("BENCHMARK SUMMARY".center(72))
        print("=" * 72)

        if not json_files:
            print("\nNo benchmark results found in", benchmark_output_dir)
            print("=" * 72)
            return

        print(f"\nFound {len(json_files)} result file(s) in {benchmark_output_dir}\n")

        groups: Dict[str, list] = {"structure": [], "motion": [], "domain_motion": []}
        for jf in json_files:
            if jf.name.startswith("structure_"):
                groups["structure"].append(jf)
            elif jf.name.startswith("domain_motion_"):
                groups["domain_motion"].append(jf)
            elif jf.name.startswith("motion_"):
                groups["motion"].append(jf)

        for group, files in groups.items():
            if not files:
                continue
            print(f"\n── {group.upper()} ──")
            for jf in files:
                with open(jf) as f:
                    data = json.load(f)
                print()
                print(_format_entry(data))

        print("\n" + "=" * 72)
