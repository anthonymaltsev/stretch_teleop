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

import tkinter as tk

import cv2
import numpy as np
import pyrealsense2 as rs


def _get_screen_size() -> tuple[int, int]:
    root = tk.Tk()
    root.withdraw()
    w, h = root.winfo_screenwidth(), root.winfo_screenheight()
    root.destroy()
    return w, h


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


def _tile_frames(items: list[tuple[dict, np.ndarray]]) -> tuple[np.ndarray | None, list[dict]]:
    if not items:
        return None, []
    if len(items) == 1:
        cam, frame = items[0]
        return frame, [{"cam": cam, "x": 0, "w": frame.shape[1], "h": frame.shape[0], "max_h": frame.shape[0]}]

    max_h = max(frame.shape[0] for _, frame in items)
    padded = []
    layout = []
    x_offset = 0
    for cam, frame in items:
        h, w = frame.shape[:2]
        if h < max_h:
            pad = np.zeros((max_h - h, w, 3), dtype=frame.dtype)
            frame = np.vstack((frame, pad))
        padded.append(frame)
        layout.append({"cam": cam, "x": x_offset, "w": w, "h": h, "max_h": max_h})
        x_offset += w
    return np.hstack(padded), layout


def _draw_wrist_info_panel(
    display: np.ndarray,
    layout: list[dict],
    video_writer: cv2.VideoWriter | None,
    recording_path: Path | None,
    recording_started_at: datetime | None,
) -> None:
    wrist = None
    for item in layout:
        if item["cam"]["is_wrist"]:
            wrist = item
            break
    if wrist is None:
        return

    x0 = wrist["x"]
    x1 = wrist["x"] + wrist["w"]
    y0 = wrist["h"]
    y1 = wrist["max_h"]
    if y1 <= y0:
        return

    cv2.rectangle(display, (x0, y0), (x1, y1), (20, 40, 20), -1)
    color = (0, 255, 0)
    line_y = y0 + 24
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status = "REC" if video_writer is not None else "IDLE"
    lines = [f"Status: {status}", f"Time: {now_str}", f"Cameras: {len(layout)}"]
    if recording_started_at is not None:
        elapsed_s = int((datetime.now() - recording_started_at).total_seconds())
        lines.append(f"Elapsed: {elapsed_s}s")
    if recording_path is not None:
        lines.append(f"File: {recording_path.name}")

    for line in lines:
        if line_y >= y1 - 8:
            break
        cv2.putText(
            display,
            line,
            (x0 + 10, line_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
        line_y += 24


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

    screen_w, screen_h = _get_screen_size()

    print("Starting gamepad teleop...")
    teleop_proc = subprocess.Popen([sys.executable, str(gamepad_demo)])
    recordings_dir = script_dir / "recordings"
    video_writer: cv2.VideoWriter | None = None
    recording_path: Path | None = None
    recording_started_at: datetime | None = None
    last_saved_recording_path: Path | None = None

    try:
        print("Press 'q' to quit, 's' to save current frames, 'r' to start/stop video recording.")
        cv2.namedWindow("Stretch Cameras", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Stretch Cameras", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        while True:
            if teleop_proc.poll() is not None:
                print("Gamepad teleop process exited.")
                break

            frames_for_window = []
            for cam in cameras:
                frame = _render_camera_frame(cam)
                if frame is None:
                    continue
                frames_for_window.append((cam, frame))

            tiled, layout = _tile_frames(frames_for_window)
            if tiled is not None:
                display = tiled.copy()
                panel_path = recording_path if video_writer is not None else last_saved_recording_path
                _draw_wrist_info_panel(display, layout, video_writer, panel_path, recording_started_at)
                if video_writer is not None:
                    video_writer.write(display)
                cv2.imshow("Stretch Cameras", cv2.resize(display, (screen_w, screen_h)))

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
                            recording_started_at = datetime.now()
                            print(f"Recording started: {recording_path}")
                else:
                    video_writer.release()
                    print(f"Recording saved: {recording_path}")
                    last_saved_recording_path = recording_path
                    video_writer = None
                    recording_path = None
                    recording_started_at = None
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
            last_saved_recording_path = recording_path
            recording_started_at = None
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
