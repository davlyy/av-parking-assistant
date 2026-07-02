from __future__ import annotations
import time
import numpy as np
from input.config import CarlaConfig, CarlaCameraConfig, Vector3D, Rotation3D
from input.source_iface import Frame


class CarlaSource:
    def __init__(self, config: CarlaConfig) -> None:
        self._config = config
        self._client = None
        self._world = None
        self._vehicle = None
        self._camera = None
        self._image_queue: list[np.ndarray] = []
        self._is_open = False
        self._original_settings = None
        self._pygame_display = None
        self._pygame_clock = None
        self._control = None
        self._autopilot_enabled = False

    def open(self) -> None:
        import carla

        self._client = carla.Client(self._config.host, self._config.port)
        self._client.set_timeout(self._config.timeout)

        world = self._client.get_world()
        if world.get_map().name != self._config.scenario.map:
            world = self._client.load_world(self._config.scenario.map)

        if self._config.synchronous:
            self._original_settings = world.get_settings()
            settings = world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            world.apply_settings(settings)

        if self._config.scenario.weather:
            weather = getattr(carla.WeatherParameters, self._config.scenario.weather, None)
            if weather:
                world.set_weather(weather)

        self._world = world
        blueprint_library = world.get_blueprint_library()

        vehicle_bps = blueprint_library.filter(self._config.scenario.ego_vehicle.model)
        if not vehicle_bps:
            raise RuntimeError(
                f"Blueprint '{self._config.scenario.ego_vehicle.model}' not found. "
                f"Available vehicle models: {[bp.id for bp in blueprint_library.filter('vehicle.*')[:10]]}"
            )

        vehicle_bp = vehicle_bps[0]
        if vehicle_bp.has_attribute("role_name"):
            vehicle_bp.set_attribute("role_name", "av_ego")

        self._tick_world()

        sp = self._config.scenario.ego_vehicle.spawn_point
        if sp is not None:
            preferred_transform = carla.Transform(
                carla.Location(sp.x, sp.y, sp.z),
                carla.Rotation(yaw=sp.yaw),
            )
        else:
            spawn_points = world.get_map().get_spawn_points()
            spawn_idx = self._config.scenario.ego_vehicle.spawn_point_index
            if spawn_idx >= len(spawn_points):
                raise RuntimeError(
                    f"Spawn point index {spawn_idx} out of range "
                    f"(max {len(spawn_points) - 1})"
                )
            preferred_transform = spawn_points[spawn_idx]

        self._vehicle = world.try_spawn_actor(vehicle_bp, preferred_transform)

        if self._vehicle is None and sp is not None:
            offsets = [(x, y) for x in range(-3, 4) for y in range(-3, 4) if x != 0 or y != 0]
            for dx, dy in offsets:
                t = carla.Transform(
                    carla.Location(sp.x + dx, sp.y + dy, sp.z),
                    carla.Rotation(yaw=sp.yaw),
                )
                self._vehicle = world.try_spawn_actor(vehicle_bp, t)
                if self._vehicle is not None:
                    break

        if self._vehicle is None:
            fallback_points = world.get_map().get_spawn_points()
            import random
            random.shuffle(fallback_points)

            for pt in fallback_points[:20]:
                self._vehicle = world.try_spawn_actor(vehicle_bp, pt)
                if self._vehicle is not None:
                    break

        if self._vehicle is None:
            raise RuntimeError("Failed to spawn ego vehicle.")

        if self._config.autopilot:
            self._vehicle.set_autopilot(True)
            self._autopilot_enabled = True

        if self._config.manual:
            self._init_pygame()
        else:
            self._control = carla.VehicleControl()

        self._setup_camera()
        self._tick_world()
        self._is_open = True

    def _tick_world(self) -> None:
        if self._world is None:
            return

        if self._config.synchronous:
            self._world.tick()
        else:
            self._world.wait_for_tick()

    def _init_pygame(self) -> None:
        import pygame
        import carla

        pygame.init()
        pygame.font.init()

        cam_cfg = self._config.camera or CarlaCameraConfig(
            position=Vector3D(x=-5.0, y=0.0, z=2.5),
            rotation=Rotation3D(pitch=-15.0, yaw=0.0, roll=0.0),
        )

        self._pygame_display = pygame.display.set_mode(
            (cam_cfg.width, cam_cfg.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF,
        )
        self._pygame_clock = pygame.time.Clock()
        self._control = carla.VehicleControl()

    def _setup_camera(self) -> None:
        import carla

        cam_cfg = self._config.camera or CarlaCameraConfig(
            position=Vector3D(x=-5.0, y=0.0, z=2.5),
            rotation=Rotation3D(pitch=-15.0, yaw=0.0, roll=0.0),
        )

        blueprint_library = self._world.get_blueprint_library()
        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(cam_cfg.width))
        camera_bp.set_attribute("image_size_y", str(cam_cfg.height))
        camera_bp.set_attribute("fov", str(cam_cfg.fov))

        transform = carla.Transform(
            carla.Location(cam_cfg.position.x, cam_cfg.position.y, cam_cfg.position.z),
            carla.Rotation(cam_cfg.rotation.pitch, cam_cfg.rotation.yaw, cam_cfg.rotation.roll),
        )

        self._camera = self._world.spawn_actor(camera_bp, transform, attach_to=self._vehicle)
        self._camera.listen(lambda image: self._on_image(image))

    def _on_image(self, image) -> None:
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        self._image_queue.append(array[:, :, :3].copy())

        if len(self._image_queue) > 3:
            self._image_queue = self._image_queue[-3:]

    def _handle_manual_input(self) -> bool:
        import pygame

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_ESCAPE:
                    return False

                if event.key == pygame.K_p:
                    self._autopilot_enabled = not self._autopilot_enabled
                    self._vehicle.set_autopilot(self._autopilot_enabled)

                if event.key == pygame.K_q:
                    self._control.reverse = not self._control.reverse

        if not self._autopilot_enabled:
            keys = pygame.key.get_pressed()

            forward = keys[pygame.K_UP] or keys[pygame.K_w]
            backward = keys[pygame.K_DOWN] or keys[pygame.K_s]

            if forward:
                self._control.reverse = False
                self._control.throttle = 0.5
                self._control.brake = 0.0
            elif backward:
                self._control.reverse = True
                self._control.throttle = 0.5
                self._control.brake = 0.0
            else:
                self._control.throttle = 0.0
                self._control.brake = 0.0

            steer = 0.0
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                steer = -0.2
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                steer = 0.2

            self._control.steer = steer
            self._control.hand_brake = keys[pygame.K_SPACE]
            self._vehicle.apply_control(self._control)

        return True

    def vehicle_pose(self) -> tuple[float, float, float, float]:
        t = self._vehicle.get_transform()
        return (t.location.x, t.location.y, t.location.z, t.rotation.yaw)

    def _render_pygame(self, frame: Frame) -> None:
        import pygame

        surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        self._pygame_display.blit(surface, (0, 0))

        x, y, z, yaw = self.vehicle_pose()
        font = pygame.font.Font(None, 28)

        lines = [
            f"X: {x:.1f}",
            f"Y: {y:.1f}",
            f"Z: {z:.1f}",
            f"Yaw: {yaw:.1f}",
        ]

        for i, line in enumerate(lines):
            text = font.render(line, True, (0, 255, 0), (0, 0, 0))
            self._pygame_display.blit(text, (10, 10 + i * 24))

        pygame.display.flip()
        self._pygame_clock.tick(60)

    def read(self) -> Frame | None:
        if not self._is_open:
            return None

        if self._config.manual:
            if not self._handle_manual_input():
                self._is_open = False
                return None

        self._tick_world()

        timeout = time.monotonic() + 2.0
        while not self._image_queue and time.monotonic() < timeout:
            time.sleep(0.01)

        frame = self._image_queue.pop(0) if self._image_queue else None

        if frame is not None and self._config.manual and self._pygame_display:
            self._render_pygame(frame)

        return frame

    def release(self) -> None:
        self._is_open = False

        camera = self._camera
        vehicle = self._vehicle
        world = self._world

        self._camera = None
        self._vehicle = None

        if camera is not None:
            try:
                camera.stop()
            except Exception as e:
                print("camera.stop failed:", e)

        self._safe_tick("tick after camera.stop failed")

        if camera is not None:
            try:
                camera.destroy()
            except Exception as e:
                print("camera.destroy failed:", e)

        self._safe_tick("tick after camera.destroy failed")

        if vehicle is not None:
            try:
                vehicle.set_autopilot(False)
            except Exception:
                pass

            try:
                vehicle.destroy()
            except Exception as e:
                print("vehicle.destroy failed:", e)

        self._safe_tick("tick after vehicle.destroy failed")

        if self._original_settings is not None and world is not None:
            try:
                world.apply_settings(self._original_settings)
            except Exception as e:
                print("apply original settings failed:", e)

        self._safe_tick("tick after settings restore failed")

        self._image_queue.clear()
        self._world = None
        self._client = None
        self._control = None
        self._pygame_display = None
        self._pygame_clock = None
        self._original_settings = None
        self._autopilot_enabled = False

        try:
            import pygame
            pygame.quit()
        except Exception:
            pass

    def _safe_tick(self, error_message: str) -> None:
        if self._world is None:
            return

        try:
            self._tick_world()
        except Exception as e:
            print(error_message + ":", e)

    def __iter__(self):
        return self

    def __next__(self) -> Frame:
        frame = self.read()
        if frame is None:
            raise StopIteration
        return frame