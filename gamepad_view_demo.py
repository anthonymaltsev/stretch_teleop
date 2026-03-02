#!/usr/bin/env python3

"""
Run gamepad teleoperation and camera visualization together.

- Launches the existing gamepad teleop demo in a subprocess.
- Visualizes color + depth streams for connected RealSense cameras.
- If a D405 is present, it is labeled as the wrist camera.

Press 'q' in any camera window to quit.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


def _camera_label(device_name: str, index: int) -> str:
    lower = device_name.lower()
    if "d405" in lower:
        return "Wrist Camera"
    if "d435" in lower or "d455" in lower:
        return "Head Camera"
    return f"Camera {index}"


def _start_pipeline(serial: str, fps: int) -> tuple[rs.pipeline, rs.align]:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, fps)
    pipeline.start(config)
    return pipeline, rs.align(rs.stream.color)


def _discover_cameras() -> list[dict]:
    ctx = rs.context()
    devices = list(ctx.query_devices())
    if not devices:
        return []

    # Two cameras at 30 FPS can exceed USB bandwidth on some setups.
    fps = 15 if len(devices) > 1 else 30
    cameras = []
    for i, dev in enumerate(devices, start=1):
        name = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)
        try:
            pipeline, align = _start_pipeline(serial, fps=fps)
        except RuntimeError as exc:
            print(f"Skipping {name} ({serial}): {exc}")
            continue

        label = _camera_label(name, i)
        print(f"Started {label}: {name} ({serial}) at {fps} FPS")
        cameras.append(
            {
                "name": name,
                "serial": serial,
                "label": label,
                "pipeline": pipeline,
                "align": align,
                "save_idx": 0,
                "last_frame": None,
                "error_count": 0,
            }
        )

    return cameras


def _render_camera_frame(cam: dict) -> np.ndarray | None:
    try:
        frames = cam["pipeline"].poll_for_frames()
    except RuntimeError as exc:
        cam["error_count"] += 1
        if cam["error_count"] in (1, 30):
            print(f"{cam['label']} frame error: {exc}")
        return cam["last_frame"]

    if not frames:
        return cam["last_frame"]

    aligned = cam["align"].process(frames)

    depth_frame = aligned.get_depth_frame()
    color_frame = aligned.get_color_frame()
    if not depth_frame or not color_frame:
        return cam["last_frame"]

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    depth_colormap = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_image, alpha=0.03),
        cv2.COLORMAP_JET,
    )

    color_image = cv2.rotate(color_image, cv2.ROTATE_90_CLOCKWISE)
    depth_colormap = cv2.rotate(depth_colormap, cv2.ROTATE_90_CLOCKWISE)

    composed = np.hstack((color_image, depth_colormap))
    cam["last_frame"] = composed
    cam["error_count"] = 0
    return composed


def _stop_cameras(cameras: list[dict]) -> None:
    for cam in cameras:
        try:
            cam["pipeline"].stop()
        except RuntimeError:
            pass


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    gamepad_demo = script_dir / "gamepad_demo.py"

    if not gamepad_demo.exists():
        print(f"Missing script: {gamepad_demo}")
        return 1

    cameras = _discover_cameras()
    if not cameras:
        print("No RealSense cameras found.")

    print("Starting gamepad teleop...")
    teleop_proc = subprocess.Popen([sys.executable, str(gamepad_demo)])

    try:
        print("Press 'q' in a camera window to quit, 's' to save current frames.")

        while True:
            if teleop_proc.poll() is not None:
                print("Gamepad teleop process exited.")
                break

            for cam in cameras:
                frame = _render_camera_frame(cam)
                if frame is None:
                    continue
                cv2.imshow(f"{cam['label']} - Color (Left) | Depth (Right)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Quitting...")
                break
            if key == ord("s"):
                for cam in cameras:
                    frame = cam["last_frame"]
                    if frame is None:
                        continue
                    out = f"{cam['label'].lower().replace(' ', '_')}_{cam['save_idx']}.png"
                    cv2.imwrite(out, frame)
                    print(f"Saved {out}")
                    cam["save_idx"] += 1

    finally:
        _stop_cameras(cameras)
        cv2.destroyAllWindows()

        if teleop_proc.poll() is None:
            teleop_proc.terminate()
            try:
                teleop_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                teleop_proc.kill()
                teleop_proc.wait(timeout=3)

        print("Shutdown complete.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
