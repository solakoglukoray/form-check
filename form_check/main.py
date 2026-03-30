"""Video analysis pipeline — extract pose, score form, save annotated frames."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from form_check.benchmarks import score_angles
from form_check.models import get_model_path
from form_check.poses import KEY_JOINT, check_orientation, extract_angles

# MediaPipe Tasks API (v0.10+)
_BaseOptions = mp.tasks.BaseOptions
_PoseLandmarker = mp.tasks.vision.PoseLandmarker
_PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
_RunningMode = mp.tasks.vision.RunningMode
_PoseConnections = mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS


def _draw_skeleton(
    frame: np.ndarray,
    landmarks: list,
    connections: list,
) -> None:
    """Draw pose skeleton onto frame in-place using OpenCV."""
    h, w = frame.shape[:2]
    pts = {
        i: (int(lm.x * w), int(lm.y * h))
        for i, lm in enumerate(landmarks)
    }
    for conn in connections:
        a, b = conn.start, conn.end
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], (224, 224, 224), 2, cv2.LINE_AA)
    for pt in pts.values():
        cv2.circle(frame, pt, 4, (0, 0, 255), -1, cv2.LINE_AA)


def find_rep_peaks(angle_series: list[float], min_distance: int = 3) -> list[int]:
    """
    Find indices of rep bottom positions in an angle time-series.

    The bottom of a rep corresponds to a local minimum in the key joint angle.

    Args:
        angle_series: Key joint angle at each sampled frame.
        min_distance: Minimum frames between two detected peaks.

    Returns:
        List of indices (chronological order).
    """
    if len(angle_series) < 3:
        return list(range(len(angle_series)))

    peaks: list[int] = []
    for i in range(1, len(angle_series) - 1):
        is_min = (
            angle_series[i] < angle_series[i - 1]
            and angle_series[i] < angle_series[i + 1]
        )
        if is_min and (not peaks or i - peaks[-1] >= min_distance):
            peaks.append(i)

    return peaks


def _draw_overlay(
    frame: np.ndarray,
    angles: dict[str, float],
    score: int,
    rep_num: Optional[int] = None,
) -> np.ndarray:
    """Render joint angles, rep number, and form score onto a video frame."""
    overlay = frame.copy()
    if score >= 70:
        color = (0, 200, 0)
    elif score >= 50:
        color = (0, 140, 255)
    else:
        color = (0, 0, 220)

    label = f"Rep {rep_num} - Score: {score}/100" if rep_num else f"Score: {score}/100"
    cv2.putText(
        overlay, label, (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA,
    )
    for idx, (joint, angle) in enumerate(angles.items()):
        cv2.putText(
            overlay, f"{joint}: {angle:.1f} deg", (10, 70 + idx * 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
        )
    return overlay


def analyze_video(
    video_path: str,
    exercise: str,
    output_dir: Optional[str] = None,
    sample_rate: int = 10,
) -> dict:
    """
    Analyze a workout video and return per-rep form scores.

    Strategy:
    1. Sample every Nth frame and extract joint angles via MediaPipe Tasks API.
    2. Detect rep bottom positions via local minima in the key joint angle.
    3. Score only the peak (deepest) frame of each rep.
    4. Fallback: if no reps detected, score bottom-quartile frames.

    Returns dict with keys:
        avg_score, rep_count, rep_scores, frames_analyzed,
        avg_angles, feedback, annotated_frames, orientation_warnings.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {video_path!r}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    frame_data: list[dict] = []
    orientation_warnings: list[str] = []
    frame_idx = 0

    options = _PoseLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=get_model_path()),
        running_mode=_RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    with _PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % sample_rate != 0:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((frame_idx / fps) * 1000)
            results = landmarker.detect_for_video(mp_image, timestamp_ms)

            if not results.pose_landmarks:
                continue

            landmarks = results.pose_landmarks[0]

            if not frame_data and not orientation_warnings:
                orientation_warnings = check_orientation(landmarks, exercise)

            try:
                angles = extract_angles(landmarks, exercise)
            except (IndexError, ZeroDivisionError, AttributeError):
                continue

            frame_data.append({
                "angles": angles,
                "landmarks": landmarks,
                "bgr": frame.copy() if output_dir else None,
            })

    cap.release()

    if not frame_data:
        return {
            "avg_score": 0,
            "rep_count": 0,
            "rep_scores": [],
            "frames_analyzed": 0,
            "avg_angles": {},
            "feedback": [
                "No pose detected — ensure full body is visible in frame."
            ],
            "annotated_frames": [],
            "orientation_warnings": orientation_warnings,
        }

    # --- Rep detection -------------------------------------------------------
    key_joint = KEY_JOINT[exercise]
    key_angles = [fd["angles"].get(key_joint, 180.0) for fd in frame_data]
    peak_indices = find_rep_peaks(key_angles)

    if not peak_indices:
        sorted_by_angle = sorted(range(len(key_angles)), key=lambda i: key_angles[i])
        peak_indices = sorted_by_angle[: max(1, len(sorted_by_angle) // 4)]
        peak_indices.sort()

    # --- Score peak frames ---------------------------------------------------
    rep_scores: list[int] = []
    peak_angles_list: list[dict[str, float]] = []

    for idx in peak_indices:
        result = score_angles(frame_data[idx]["angles"], exercise)
        rep_scores.append(result["score"])
        peak_angles_list.append(frame_data[idx]["angles"])

    # --- Annotated frame export (one image per rep) --------------------------
    annotated_frames: list[str] = []
    if output_dir:
        for rep_num, data_idx in enumerate(peak_indices, start=1):
            fd = frame_data[data_idx]
            result = score_angles(fd["angles"], exercise)
            annotated = _draw_overlay(fd["bgr"], fd["angles"], result["score"], rep_num)
            _draw_skeleton(annotated, fd["landmarks"], _PoseConnections)
            frame_path = str(Path(output_dir) / f"rep_{rep_num:02d}.jpg")
            cv2.imwrite(frame_path, annotated)
            annotated_frames.append(frame_path)

    # --- Aggregate over peak frames ------------------------------------------
    merged: dict[str, list[float]] = {}
    for angles in peak_angles_list:
        for joint, angle in angles.items():
            merged.setdefault(joint, []).append(angle)
    avg_angles = {joint: sum(vals) / len(vals) for joint, vals in merged.items()}

    summary = score_angles(avg_angles, exercise)

    return {
        "avg_score": int(sum(rep_scores) / len(rep_scores)),
        "rep_count": len(peak_indices),
        "rep_scores": rep_scores,
        "frames_analyzed": len(frame_data),
        "avg_angles": avg_angles,
        "feedback": summary["feedback"],
        "annotated_frames": annotated_frames,
        "orientation_warnings": orientation_warnings,
    }
