from __future__ import annotations
import argparse
import cv2
from input import load_config, CameraSource, CarlaSource, SourceConfig
import json
from pathlib import Path
from mod.path_planning import *
from frame_analyze import *


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

def draw_path_in_carla(source, path, z: float = 0.3, life_time: float = 0.1, stride: int = 1) -> None:
    import carla

    world = source._client.get_world()
    debug = world.debug

    if not path or len(path) < 2:
        return

    sparse_path = path[::stride]

    if sparse_path[-1] is not path[-1]:
        sparse_path.append(path[-1])

    for a, b in zip(sparse_path, sparse_path[1:]):
        debug.draw_line(
            carla.Location(x=float(a.x), y=float(a.y), z=z),
            carla.Location(x=float(b.x), y=float(b.y), z=z),
            thickness=0.05,
            color=carla.Color(255, 0, 0),
            life_time=life_time,
        )

    debug.draw_point(
        carla.Location(x=float(path[0].x), y=float(path[0].y), z=z),
        size=0.15,
        color=carla.Color(0, 255, 0),
        life_time=life_time,
    )

    debug.draw_point(
        carla.Location(x=float(path[-1].x), y=float(path[-1].y), z=z),
        size=0.15,
        color=carla.Color(0, 0, 255),
        life_time=life_time,
    )

def draw_test_marker_in_carla(source, z: float = 1.0, life_time: float = 0.1) -> None:
    import carla

    x, y, vehicle_z, yaw = source.vehicle_pose()
    world = source._client.get_world()

    loc = carla.Location(
        x=float(x),
        y=float(y),
        z=float(vehicle_z) + z
    )

    world.debug.draw_point(
        loc,
        size=0.3,
        color=carla.Color(255, 0, 0),
        life_time=life_time
    )

    world.debug.draw_string(
        loc,
        "EGO",
        draw_shadow=False,
        color=carla.Color(255, 0, 0),
        life_time=life_time
    )

def carla_world_to_pixel(x: float, y: float, frame, config: SourceConfig) -> tuple[int, int]:
    boundaries = config.carla.scenario.boundaries

    cx = boundaries.center.x
    cy = boundaries.center.y
    ex = boundaries.extent.x
    ey = boundaries.extent.y

    height, width = frame.shape[:2]

    min_x = cx - ex
    max_x = cx + ex
    min_y = cy - ey
    max_y = cy + ey

    u = int((x - min_x) / (max_x - min_x) * width)

    # Bild-y geht nach unten, Welt-y meistens nach oben
    v = int((max_y - y) / (max_y - min_y) * height)

    return u, v

def run(source, config: SourceConfig, payload: dict) -> None:
    source.open()

    path = None
    carla_payload = payload
    frame_counter = 0

    try:
        print("Planning path from payload.json...")

        print("Payload start before conversion:", payload["start_pose"])
        print("Payload goal before conversion:", payload["goal_pose"])
        print("Obstacles:", len(payload.get("obstacles", [])))

        if isinstance(source, CarlaSource):
            x, y, z, yaw = source.vehicle_pose()

            ego_world_pose = {
                "x": x,
                "y": y,
                "yaw": yaw,
            }

            print("CARLA ego pose:", ego_world_pose)

            # Payload genau einmal konvertieren
            carla_payload = convert_payload_to_carla_world(
                payload,
                ego_world_pose,
            )

            print("Payload start after conversion:", carla_payload["start_pose"])
            print("Payload goal after conversion:", carla_payload["goal_pose"])

            for i, obs in enumerate(carla_payload.get("obstacles", [])):
                print(f"obs {i} after conversion:", obs)

            # Wichtig: hier keine ego_world_pose mehr übergeben
            path = plan_path(payload=carla_payload)

        else:
            path = plan_path(payload=payload)

        print("Path length:", len(path) if path else 0)
        print("First:", path[0] if path else None)
        print("Last:", path[-1] if path else None)

        for frame in source:
            if frame is None:
                break

            frame_counter += 1

            if isinstance(source, CarlaSource):
                # leichter Marker, jedes Frame
                draw_test_marker_in_carla(
                    source,
                    z=1.0,
                    life_time=0.2,
                )

                # schwerere Debug-Zeichnungen nur jedes 5. Frame
                if frame_counter % 5 == 0:
                    draw_obstacles_in_carla(
                        source,
                        carla_payload,
                        z=1.2,
                        life_time=1.0,
                    )

                    if path:
                        draw_path_in_carla(
                            source,
                            path,
                            z=1.0,
                            life_time=1.0,
                            stride=4,
                        )

            else:
                cv2.imshow("AV Parking Assistant", frame)

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
        default=str(Path(__file__).resolve().parent.parent / "config" / "config.json"),
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

    #sample payload to test
    payload = json.load(open(Path(__file__).resolve().parent.parent / "config" / "payload.json", encoding="utf-8"))


    source = create_source(config)
    run(source, config, payload)


if __name__ == "__main__":
    main()
