"""Biomechanical benchmarks and scoring for exercise form analysis."""

from dataclasses import dataclass


@dataclass
class AngleBenchmark:
    name: str
    min_angle: float
    max_angle: float
    weight: float  # importance weight (0-1) for final score


BENCHMARKS: dict[str, dict[str, AngleBenchmark]] = {
    "squat": {
        "knee": AngleBenchmark("Knee angle at bottom", 70.0, 120.0, 0.5),
        "hip": AngleBenchmark("Hip angle at bottom", 60.0, 100.0, 0.3),
        "back": AngleBenchmark("Back lean (deg from vertical)", 15.0, 50.0, 0.2),
    },
    "deadlift": {
        "hip": AngleBenchmark("Hip hinge angle", 30.0, 70.0, 0.4),
        "knee": AngleBenchmark("Knee bend angle", 90.0, 130.0, 0.3),
        "back": AngleBenchmark("Back angle from vertical", 30.0, 60.0, 0.3),
    },
    "pushup": {
        "elbow": AngleBenchmark("Elbow angle at bottom", 70.0, 110.0, 0.5),
        "hip": AngleBenchmark("Body alignment (hip angle)", 160.0, 180.0, 0.5),
    },
}

_MAX_DEVIATION = 30.0  # degrees past boundary before score reaches 0


def score_angles(angles: dict[str, float], exercise: str) -> dict:
    """
    Score joint angles against biomechanical benchmarks for the given exercise.

    Returns:
        dict with keys ``score`` (0-100 int) and ``feedback`` (list of strings).
    """
    if exercise not in BENCHMARKS:
        raise ValueError(
            f"Unknown exercise: {exercise!r}. Choose from: {list(BENCHMARKS.keys())}"
        )

    benchmarks = BENCHMARKS[exercise]
    total_weighted_score = 0.0
    total_weight = 0.0
    feedback: list[str] = []

    for joint, benchmark in benchmarks.items():
        if joint not in angles:
            continue

        angle = angles[joint]
        weight = benchmark.weight
        total_weight += weight

        if benchmark.min_angle <= angle <= benchmark.max_angle:
            joint_score = 100.0
            feedback.append(f"{benchmark.name}: Good ({angle:.1f} deg)")
        else:
            deviation = min(
                abs(angle - benchmark.min_angle),
                abs(angle - benchmark.max_angle),
            )
            joint_score = max(0.0, 100.0 - (deviation / _MAX_DEVIATION) * 100.0)

            if angle < benchmark.min_angle:
                feedback.append(
                    f"{benchmark.name}: Too low ({angle:.1f} deg, target "
                    f"{benchmark.min_angle:.0f}-{benchmark.max_angle:.0f})"
                )
            else:
                feedback.append(
                    f"{benchmark.name}: Too high ({angle:.1f} deg, target "
                    f"{benchmark.min_angle:.0f}-{benchmark.max_angle:.0f})"
                )

        total_weighted_score += joint_score * weight

    final_score = int(total_weighted_score / total_weight) if total_weight > 0 else 0
    return {"score": final_score, "feedback": feedback}
