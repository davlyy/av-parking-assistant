from __future__ import annotations
import argparse
import dataclasses
import os
from pathlib import Path

import cv2
from input import load_config, CameraSource, CarlaSource, SourceConfig

try:
    from frame_analyze import process_frame, YOLO_MODEL_PATH
except Exception:
    process_frame = None
    YOLO_MODEL_PATH = "besttoy.pt"

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_model_path(model_name: str = YOLO_MODEL_PATH) -> str:
    candidates = [
        Path(os.environ.get("BESTTOY_MODEL_PATH", "")) if os.environ.get("BESTTOY_MODEL_PATH") else None,
        PROJECT_ROOT / model_name,
        PROJECT_ROOT / "src" / model_name,
        Path.cwd() / model_name,
        Path(model_name),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        if candidate.is_file():
            return str(candidate)
    return str(PROJECT_ROOT / model_name)


def resolve_source_config(config: SourceConfig, source_override: str | None) -> SourceConfig:
    if source_override is None:
        return config
    return dataclasses.replace(config, source_type=source_override)


def create_source(config: SourceConfig):
    if config.source_type == "carla":
        if config.carla is None:
            raise ValueError("CARLA config missing")
        source = CarlaSource(config.carla)
    else:
        if config.camera is None:
            raise ValueError("Camera config missing")
        source = CameraSource(config.camera)
    return source


def _draw_coords(frame, source) -> None:
    if isinstance(source, CarlaSource):
        x, y, z, yaw = source.vehicle_pose()
        lines = [
            f"X: {x:.1f}",
            f"Y: {y:.1f}",
            f"Z: {z:.1f}",
            f"Yaw: {yaw:.1f}",
        ]
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (10, 30 + i * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def run(source, config: SourceConfig) -> None:
    source.open()
    try:
        if config.source_type == "carla" and config.carla and config.carla.manual:
            for frame in source:
                if frame is None:
                    break
        else:
            for frame in source:
                if frame is None:
                    break

                if process_frame is not None and config.source_type == "camera":
                    try:
                        result, _payload = process_frame(
                            frame,
                            detector="yolo",
                            yolo_model_path=resolve_model_path(),
                            confidence=0.2,
                        )
                        display = result
                    except Exception:
                        display = frame.copy()
                else:
                    display = frame.copy()

                _draw_coords(display, source)
                cv2.imshow(f"Input - {config.source_type}", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        source.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="AV Parking Assistant")
    parser.add_argument(
        "--source",
        choices=["camera", "carla"],
        default=None,
        help="Override the input source configured in the JSON file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./config/config.json",
        help="Path to JSON config file",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="scenario_1",
        help="Named scenario preset from config",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    config = resolve_source_config(config, args.source)

    if args.scenario and config.carla and args.scenario in config.carla.scenarios:
        preset = config.carla.scenarios[args.scenario]
        config = dataclasses.replace(
            config,
            carla=dataclasses.replace(config.carla, scenario=preset, scenario_name=args.scenario),
        )

    source = create_source(config)
    run(source, config)


if __name__ == "__main__":
    main()
