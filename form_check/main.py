"""Video analysis pipeline — extract pose, score form, save annotated frames."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from form_check.benchmarks import score_angles
from form_check.poses import extract_angles


def _draw_overlay(
    frame: np.ndarray,
    angles: dict[str, float],
    score: int,
) -> np.ndarray:
    """Render angle values and form score onto a video frame."""
    overlay = frame.copy()
    if score >= 70:
        color = (0, 200, 0)
    elif score >= 50:
        color = (0, 140, 255)
    else:
        color = (0, 0, 220)
    cv2.putText(
        overlay,
        f"Form Score: {score}/100",
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color,
        2,
        cv2.LINE_AA,
    )
    for idx, (joint, angle) in enumerate(angles.items()):
        cv2.putText(
            overlay,
            f"{joint}: {angle:.1f} deg",
            (10, 70 + idx * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return overlay


def analyze_video(
    video_path: str,
    exercise: str,
    output_dir: Optional[str] = None,
    sample_rate: int = 10,
) -> dict:
    """
    Analyze a workout video and return form scores and joint angles.

    Args:
        video_path: Path to the input video file.
        exercise: Exercise type: "squat", "deadlift", or "pushup".
        output_dir: Directory to write annotated JPEG frames. None = skip.
        sample_rate: Analyze every Nth frame (default 10).

    Returns:
        Dict with keys: avg_score, min_score, max_score, frames_analyzed,
        avg_angles, feedback, annotated_frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {video_path!r}")

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    scores: list[int] = []
    all_angles: dict[str, list[float]] = {}
    annotated_frames: list[str] = []
    frame_idx = 0
    analyzed = 0

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % sample_rate != 0:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if not results.pose_landmarks:
                continue

            try:
                angles = extract_angles(results.pose_landmarks.landmark, exercise)
            except (IndexError, ZeroDivisionError, AttributeError):
                continue

            result = score_angles(angles, exercise)
            scores.append(result["score"])

            for joint, angle in angles.items():
                all_angles.setdefault(joint, []).append(angle)

            if output_dir:
                annotated = _draw_overlay(frame, angles, result["score"])
                mp_drawing.draw_landmarks(
                    annotated,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                )
                frame_path = str(Path(output_dir) / f"frame_{analyzed:04d}.jpg")
                cv2.imwrite(frame_path, annotated)
                annotated_frames.append(frame_path)

            analyzed += 1

    cap.release()

    if not scores:
        return {
            "avg_score": 0,
            "min_score": 0,
            "max_score": 0,
            "frames_analyzed": 0,
            "avg_angles": {},
            "feedback": ["No pose detected — ensure full body is visible in frame."],
            "annotated_frames": [],
        }

    avg_angles = {joint: sum(vals) / len(vals) for joint, vals in all_angles.items()}
    summary = score_angles(avg_angles, exercise)

    return {
        "avg_score": int(sum(scores) / len(scores)),
        "min_score": min(scores),
        "max_score": max(scores),
        "frames_analyzed": analyzed,
        "avg_angles": avg_angles,
        "feedback": summary["feedback"],
        "annotated_frames": annotated_frames,
    }
