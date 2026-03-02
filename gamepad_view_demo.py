#!/usr/bin/env python3

"""
Run gamepad teleoperation and camera visualization together.

- Launches the existing gamepad teleop demo in a subprocess.
- Visualizes color streams for connected RealSense cameras.
- If a D405 is present, it is labeled as the wrist camera.

Press 'q' in the camera window to quit.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
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


def _start_pipeline(serial: str, fps: int) -> rs.pipeline:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, fps)
    pipeline.start(config)
    return pipeline


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
            pipeline = _start_pipeline(serial, fps=fps)
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
                "save_idx": 0,
                "last_frame": None,
                "error_count": 0,
                "is_wrist": "d405" in name.lower(),
            }
        )

    # Keep a stable display order in the combined window.
    return sorted(cameras, key=lambda c: (0 if "Head" in c["label"] else 1, c["label"]))


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

    color_frame = frames.get_color_frame()
    if not color_frame:
        return cam["last_frame"]

    color_image = np.asanyarray(color_frame.get_data())
    # Head camera typically needs a 90-deg clockwise correction on Stretch.
    if cam["is_wrist"]:
        # Wrist camera should remain horizontal.
        display = color_image
        if display.shape[0] > display.shape[1]:
            display = cv2.rotate(display, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        display = cv2.rotate(color_image, cv2.ROTATE_90_CLOCKWISE)

    cam["last_frame"] = display
    cam["error_count"] = 0
    return display


def _tile_frames(frames: list[np.ndarray]) -> np.ndarray | None:
    if not frames:
        return None
    if len(frames) == 1:
        return frames[0]

    max_h = max(frame.shape[0] for frame in frames)
    padded = []
    for frame in frames:
        h, w = frame.shape[:2]
        if h < max_h:
            pad = np.zeros((max_h - h, w, 3), dtype=frame.dtype)
            frame = np.vstack((frame, pad))
        padded.append(frame)
    return np.hstack(padded)


def _stop_cameras(cameras: list[dict]) -> None:
    for cam in cameras:
        try:
            cam["pipeline"].stop()
        except RuntimeError:
            pass


def _start_video_writer(frame: np.ndarray, out_dir: Path, fps: int = 15) -> tuple[cv2.VideoWriter, Path] | tuple[None, None]:
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"teleop_recording_{timestamp}.mp4"

    h, w = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        return None, None
    return writer, out_path


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
    recordings_dir = script_dir / "recordings"
    video_writer: cv2.VideoWriter | None = None
    recording_path: Path | None = None

    try:
        print("Press 'q' to quit, 's' to save current frames, 'r' to start/stop video recording.")

        while True:
            if teleop_proc.poll() is not None:
                print("Gamepad teleop process exited.")
                break

            frames_for_window = []
            for cam in cameras:
                frame = _render_camera_frame(cam)
                if frame is None:
                    continue
                labeled = frame.copy()
                cv2.putText(
                    labeled,
                    cam["label"],
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                frames_for_window.append(labeled)

            tiled = _tile_frames(frames_for_window)
            if tiled is not None:
                display = tiled.copy()
                if video_writer is not None:
                    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    cv2.putText(
                        display,
                        f"REC {stamp}",
                        (10, display.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    video_writer.write(display)
                cv2.imshow("Stretch Cameras", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Quitting...")
                break
            if key == ord("r"):
                if video_writer is None:
                    if tiled is None:
                        print("No frame available yet; cannot start recording.")
                    else:
                        video_writer, recording_path = _start_video_writer(tiled, recordings_dir)
                        if video_writer is None:
                            print("Failed to start recording (VideoWriter could not open).")
                        else:
                            print(f"Recording started: {recording_path}")
                else:
                    video_writer.release()
                    print(f"Recording saved: {recording_path}")
                    video_writer = None
                    recording_path = None
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
        if video_writer is not None:
            video_writer.release()
            print(f"Recording saved: {recording_path}")
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
