#!/usr/bin/env python3

"""
Run gamepad teleoperation and camera visualization together.

- Launches the existing gamepad teleop demo in a subprocess.
- Visualizes color streams for connected RealSense cameras.
- If a D405 is present, it is labeled as the wrist camera.

Press 'q' in the camera window to quit.
"""

from __future__ import annotations

import queue
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

TARGET_CAMERA_FPS = 30
FALLBACK_CAMERA_FPS = 15
RECORD_CODEC = "MJPG"


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

    cameras = []
    for i, dev in enumerate(devices, start=1):
        name = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)

        used_fps = TARGET_CAMERA_FPS
        try:
            pipeline = _start_pipeline(serial, fps=used_fps)
        except RuntimeError as exc:
            print(f"{name} ({serial}) failed at {used_fps} FPS: {exc}")
            try:
                used_fps = FALLBACK_CAMERA_FPS
                pipeline = _start_pipeline(serial, fps=used_fps)
            except RuntimeError as exc2:
                print(f"Skipping {name} ({serial}): {exc2}")
                continue

        label = _camera_label(name, i)
        print(f"Started {label}: {name} ({serial}) at {used_fps} FPS")
        cameras.append(
            {
                "name": name,
                "serial": serial,
                "label": label,
                "pipeline": pipeline,
                "fps": used_fps,
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
    is_recording: bool,
    recording_path: Path | None,
    recording_started_at: datetime | None,
    dropped_frames: int,
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
    status = "REC" if is_recording else "IDLE"
    lines = [f"Status: {status}", f"Time: {now_str}", f"Cameras: {len(layout)}"]
    if recording_started_at is not None:
        elapsed_s = int((datetime.now() - recording_started_at).total_seconds())
        lines.append(f"Elapsed: {elapsed_s}s")
    if dropped_frames > 0:
        lines.append(f"Dropped: {dropped_frames}")
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


def _start_video_writer(frame: np.ndarray, out_dir: Path, fps: int) -> tuple[cv2.VideoWriter, Path] | tuple[None, None]:
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"teleop_recording_{timestamp}.avi"

    h, w = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*RECORD_CODEC)
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        return None, None
    return writer, out_path


class _AsyncRecorder:
    def __init__(self, writer: cv2.VideoWriter, max_queue: int = 120) -> None:
        self.writer = writer
        self.queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self.dropped_frames = 0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while not self._stop.is_set() or not self.queue.empty():
            try:
                frame = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue
            self.writer.write(frame)

    def write(self, frame: np.ndarray) -> None:
        try:
            self.queue.put_nowait(frame.copy())
        except queue.Full:
            self.dropped_frames += 1

    def stop(self) -> None:
        self._stop.set()
        self.thread.join(timeout=3)
        self.writer.release()


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
    recorder: _AsyncRecorder | None = None
    recording_path: Path | None = None
    recording_started_at: datetime | None = None
    last_saved_recording_path: Path | None = None
    record_fps = min((cam["fps"] for cam in cameras), default=TARGET_CAMERA_FPS)

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
                frames_for_window.append((cam, frame))

            tiled, layout = _tile_frames(frames_for_window)
            if tiled is not None:
                display = tiled.copy()
                panel_path = recording_path if recorder is not None else last_saved_recording_path
                dropped = recorder.dropped_frames if recorder is not None else 0
                _draw_wrist_info_panel(display, layout, recorder is not None, panel_path, recording_started_at, dropped)
                if recorder is not None:
                    recorder.write(display)
                cv2.imshow("Stretch Cameras", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Quitting...")
                break
            if key == ord("r"):
                if recorder is None:
                    if tiled is None:
                        print("No frame available yet; cannot start recording.")
                    else:
                        video_writer, recording_path = _start_video_writer(tiled, recordings_dir, fps=record_fps)
                        if video_writer is None:
                            print("Failed to start recording (VideoWriter could not open).")
                        else:
                            recorder = _AsyncRecorder(video_writer)
                            recording_started_at = datetime.now()
                            print(f"Recording started ({record_fps} FPS): {recording_path}")
                else:
                    recorder.stop()
                    print(f"Recording saved: {recording_path}")
                    last_saved_recording_path = recording_path
                    recorder = None
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
        if recorder is not None:
            recorder.stop()
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
