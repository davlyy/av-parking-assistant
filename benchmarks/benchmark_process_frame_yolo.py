"""Benchmark the complete local-YOLO ``process_frame`` pipeline.

Run from the repository root:
    python benchmarks/benchmark_process_frame_yolo.py
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import frame_analyze as analyzer

# Benchmark configuration
ITERATIONS = 100
BENCHMARK_MAX_WIDTH = 1_280
INPUT_IMAGE = ROOT / "example_frame" / "toy_1.jpg"
MODEL_PATH = ROOT / "model.pt"
RESULT_PATH = ROOT / "benchmark_results" / "process_frame_yolo_benchmark.json"


def main() -> None:
    frame = cv2.imread(str(INPUT_IMAGE))
    if frame is None:
        raise FileNotFoundError(f"Cannot read benchmark image: {INPUT_IMAGE}")
    original_height, original_width = frame.shape[:2]
    if original_width > BENCHMARK_MAX_WIDTH:
        benchmark_height = round(original_height * BENCHMARK_MAX_WIDTH / original_width)
        frame = cv2.resize(frame, (BENCHMARK_MAX_WIDTH, benchmark_height))

    # Keep this stream open for the full process. Ultralytics caches its logger
    # during warm-up and reuses the configured output stream for later frames.
    devnull = open(os.devnull, "w")
    try:
        # Load and warm the model outside the timed section.
        with contextlib.redirect_stdout(devnull):
            analyzer.process_frame(
                frame.copy(),
                debug_dir=None,
                frame_id=-1,
                detector="yolo",
                yolo_model_path=str(MODEL_PATH),
                confidence=0.1,
            )

        samples_ns = []
        with contextlib.redirect_stdout(devnull):
            for frame_id in range(ITERATIONS):
                started_ns = time.perf_counter_ns()
                _, payload = analyzer.process_frame(
                    frame.copy(),
                    debug_dir=None,
                    frame_id=frame_id,
                    detector="yolo",
                    yolo_model_path=str(MODEL_PATH),
                    confidence=0.1,
                )
                elapsed_ns = time.perf_counter_ns() - started_ns
                assert payload is not None
                samples_ns.append(elapsed_ns)
    finally:
        devnull.close()

    samples_ms = [sample / 1_000_000 for sample in samples_ns]
    sorted_samples = sorted(samples_ms)
    summary = {
        "mean_ms": sum(samples_ms) / len(samples_ms),
        "median_ms": sorted_samples[len(sorted_samples) // 2],
        "p95_ms": sorted_samples[int(0.95 * (len(sorted_samples) - 1))],
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "throughput_fps": 1_000 / (sum(samples_ms) / len(samples_ms)),
    }
    result = {
        "iterations": ITERATIONS,
        "input_image": str(INPUT_IMAGE.relative_to(ROOT)),
        "input_resolution": {"width": frame.shape[1], "height": frame.shape[0]},
        "original_resolution": {"width": original_width, "height": original_height},
        "detector": "yolo",
        "model": str(MODEL_PATH.relative_to(ROOT)),
        "confidence": 0.1,
        "unit": "milliseconds",
        "samples_ms": samples_ms,
        "summary": summary,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Saved benchmark data to: {RESULT_PATH}")


if __name__ == "__main__":
    main()
