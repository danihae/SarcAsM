# SarcAsM Benchmark Results

This directory contains JSON files with benchmark timing results for SarcAsM analysis pipelines.

## File Format

Each benchmark result is stored as a JSON file with the following structure:

```json
{
    "name": "structure_substeps",
    "test_file": "/path/to/test_data/file.tif",
    "timestamp": "2026-04-23T10:00:00.000000",
    "substeps": {
        "detect_sarcomeres": 12.3456,
        "analyze_cell_mask": 2.1234,
        "analyze_z_bands": 5.6789,
        "analyze_sarcomere_vectors": 3.4567,
        "analyze_myofibrils": 8.9012,
        "analyze_sarcomere_domains": 4.5678
    },
    "total_time": 37.0736,
    "metadata": {
        "device": "cuda:0",
        "sarcasm_version": "0.5.0",
        "test_file_size_mb": 150.5
    }
}
```

## Running Benchmarks

```bash
# Run all benchmarks
uv run pytest tests/test_benchmark.py -v

# Run only structure benchmarks
uv run pytest tests/test_benchmark.py::TestStructureBenchmark -v

# Run only motion benchmarks
uv run pytest tests/test_benchmark.py::TestMotionBenchmark -v

# Run only domain motion benchmarks
uv run pytest tests/test_benchmark.py::TestDomainMotionBenchmark -v

# Generate summary report
uv run pytest tests/test_benchmark.py::TestBenchmarkSummary -v

# Run without slow markers (quick test - summary only)
uv run pytest tests/test_benchmark.py -m "not slow" -v
```

## Benchmark Categories

### Structure Benchmarks
- `full_structure_pipeline`: Complete structure analysis (detect_sarcomeres + full_analysis_structure)
- `structure_substeps`: Individual substep timing
- `structure_timelapse_pipeline`: Structure analysis on time-lapse data

### Motion Benchmarks
- `full_motion_pipeline`: Complete motion analysis workflow
- `motion_substeps`: Individual motion substep timing
- `loi_detection_only`: LOI detection pipeline only

### Domain Motion Benchmarks
- `domain_motion_pipeline`: Complete domain motion analysis
- `domain_motion_substeps`: Individual domain motion substep timing

## Analyzing Results

To compare benchmark runs, you can use the summary report or load the JSON files directly:

```python
import json
from pathlib import Path

# Load all benchmark results
results_dir = Path('tests/benchmark_results')
for json_file in results_dir.glob('*.json'):
    with open(json_file) as f:
        data = json.load(f)
    print(f"{data['name']}: {data['total_time']:.2f}s")
```
