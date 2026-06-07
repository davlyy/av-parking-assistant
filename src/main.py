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
    args = parser.parse_args()

    config = load_config(args.config)
    source = create_source(config)
    run(source, config)


if __name__ == "__main__":
    main()
