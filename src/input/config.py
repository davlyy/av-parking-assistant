from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import Literal, get_type_hints


@dataclass(frozen=True)
class CameraCalibration:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class CameraSetup:
    device_id: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    calibration: CameraCalibration | None = None


@dataclass(frozen=True)
class Vector3D:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class Rotation3D:
    pitch: float
    yaw: float
    roll: float


@dataclass(frozen=True)
class EgoVehicleConfig:
    model: str = "vehicle.lincoln.mkz_2017"
    spawn_point_index: int = 0


@dataclass(frozen=True)
class CarlaScenarioConfig:
    map: str = "Town01"
    weather: str = "ClearNoon"
    ego_vehicle: EgoVehicleConfig = field(default_factory=EgoVehicleConfig)


@dataclass(frozen=True)
class CarlaCameraConfig:
    position: Vector3D
    rotation: Rotation3D
    width: int = 1280
    height: int = 720
    fov: int = 90


@dataclass(frozen=True)
class CarlaConfig:
    host: str = "localhost"
    port: int = 2000
    timeout: float = 10.0
    manual: bool = False
    synchronous: bool = True
    autopilot: bool = False
    scenario: CarlaScenarioConfig = field(default_factory=CarlaScenarioConfig)
    camera: CarlaCameraConfig | None = None


SourceType = Literal["camera", "carla"]


@dataclass(frozen=True)
class SourceConfig:
    source_type: SourceType
    camera: CameraSetup | None = None
    carla: CarlaConfig | None = None


def _dict_to_dataclass(d: dict, cls):
    if d is None:
        return None
    hints = get_type_hints(cls)
    from dataclasses import fields
    field_names = {f.name for f in fields(cls)}
    kwargs = {}
    for k, v in d.items():
        if k in field_names and k in hints:
            ft = hints[k]
            origin = getattr(ft, "__origin__", None)
            if origin is not None:
                kwargs[k] = v
            elif hasattr(ft, "__dataclass_fields__"):
                kwargs[k] = _dict_to_dataclass(v, ft) if isinstance(v, dict) else v
            else:
                kwargs[k] = v
    return cls(**kwargs)


def _pick(raw: dict, *keys: str) -> dict | None:
    for k in keys:
        if k in raw:
            return raw[k]
    return None


def load_config(path: str | Path) -> SourceConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    source_type = raw.get("source_type", "camera")

    camera = None
    carla = None
    if source_type == "camera":
        cam = _pick(raw, "Camera", "camera")
        if cam:
            calib = cam.get("calibration")
            camera = CameraSetup(
                device_id=cam.get("device_id", 0),
                width=cam.get("width", 1280),
                height=cam.get("height", 720),
                fps=cam.get("fps", 30),
                calibration=_dict_to_dataclass(calib, CameraCalibration) if calib else None,
            )
    elif source_type == "carla":
        crl = _pick(raw, "Carla", "carla")
        if crl:
            scenario = crl.get("scenario", {})
            ego_vehicle = scenario.get("ego_vehicle", {})
            cam_cfg = crl.get("camera")
            carla = CarlaConfig(
                host=crl.get("host", "localhost"),
                port=crl.get("port", 2000),
                timeout=crl.get("timeout", 10.0),
                manual=crl.get("manual", False),
                synchronous=crl.get("synchronous", True),
                autopilot=crl.get("autopilot", False),
                scenario=CarlaScenarioConfig(
                    map=scenario.get("map", "Town01"),
                    weather=scenario.get("weather", "ClearNoon"),
                    ego_vehicle=EgoVehicleConfig(
                        model=ego_vehicle.get("model", "vehicle.lincoln.mkz_2017"),
                        spawn_point_index=ego_vehicle.get("spawn_point_index", 0),
                    ),
                ),
                camera=_dict_to_dataclass(cam_cfg, CarlaCameraConfig) if cam_cfg else None,
            )

    return SourceConfig(source_type=source_type, camera=camera, carla=carla)
