from __future__ import annotations
import argparse
import cv2
from input import load_config, CameraSource, CarlaSource, SourceConfig


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
                _draw_coords(frame, source)
                cv2.imshow(f"Input - {config.source_type}", frame)
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
        default="camera",
        help="Input source type",
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

    if args.scenario and config.carla and args.scenario in config.carla.scenarios:
        import dataclasses
        preset = config.carla.scenarios[args.scenario]
        config = dataclasses.replace(
            config,
            carla=dataclasses.replace(config.carla, scenario=preset, scenario_name=args.scenario),
        )

    source = create_source(config)
    run(source, config)


if __name__ == "__main__":
    main()
