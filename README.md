# form-check

[![CI](https://github.com/solakoglukoray/form-check/actions/workflows/ci.yml/badge.svg)](https://github.com/solakoglukoray/form-check/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Point a camera at your workout, get a form score out of 100 — AI checks your squat, deadlift, and push-up against biomechanical benchmarks using MediaPipe Pose estimation.

No gym trainer? No problem. `form-check` analyzes joint angles in any workout video and tells you exactly what's off — knees caving, hips sagging, back rounding — with a per-rep score and annotated frame output for every rep.

## Features

- **Per-rep scoring** — detects each rep automatically, scores only the deepest frame (not a blurry average)
- **3D joint angles** — uses MediaPipe's depth (z) coordinate so results are accurate regardless of camera angle — no need to film from a perfect side view
- **Biomechanical benchmarks** — knee, hip, and back angles compared against evidence-based ranges for each exercise
- **Annotated frames** — one annotated JPEG per rep with skeleton overlay, joint angles, and score
- **Camera warnings** — flags low joint visibility before the result so you know if a re-film is needed
- **Three exercises** — squat, deadlift, push-up
- **Runs offline** — no cloud API, no account, works on any MP4/MOV/AVI

## Installation

```bash
pip install form-check
```

Or with Docker:
```bash
docker run --rm -v $(pwd):/data ghcr.io/solakoglukoray/form-check analyze /data/squat.mp4 --exercise squat
```

## Usage

```bash
# Score your squat form
form-check analyze my_squat.mp4 --exercise squat

# Deadlift with annotated frame output (one JPEG per rep)
form-check analyze deadlift.mp4 --exercise deadlift --output ./frames

# Push-up — analyze every 5th frame for higher granularity
form-check analyze pushup.mp4 --exercise pushup --sample-rate 5
```

### Example output

```
Analyzing squat form...
Video: squat.mp4

+-------------------------------------+
|           SQUAT RESULT              |
|  Form Score: 84/100 — Excellent     |
|  5 reps detected                    |
+-------------------------------------+

Per-Rep Scores
 Rep   Score   Grade
  1      89    Excellent
  2      91    Excellent
  3      78    Excellent
  4      84    Excellent
  5      79    Excellent

Joint Angles at Bottom Position (avg)
 Joint    Angle
 Knee     82.3 deg
 Hip      74.1 deg
 Back     28.6 deg

Feedback:
  + Knee angle at bottom: Good (82.3 deg)
  + Hip angle at bottom: Good (74.1 deg)
  + Back lean (deg from vertical): Good (28.6 deg)

Frames analyzed: 63
Saved 5 annotated frame(s) -> ./frames
```

## How It Works

1. **Frame sampling** — every Nth frame is extracted from the video (default: every 10th)
2. **Pose estimation** — MediaPipe Pose locates 33 body landmarks in 3D per frame
3. **Angle calculation** — joint angles are computed using full x/y/z coordinates, making results camera-orientation agnostic
4. **Rep detection** — local minima in the key joint angle series identify each rep's deepest point; frames where the joint never bends past the depth threshold are ignored
5. **Scoring** — each rep's deepest frame is scored against exercise-specific biomechanical benchmarks
6. **Output** — per-rep breakdown in the terminal; annotated JPEG written for each rep if `--output` is set

## Biomechanical benchmarks

| Exercise | Joint | Target Range | Weight |
|----------|-------|-------------|--------|
| Squat    | Knee  | 70-120°     | 50%    |
| Squat    | Hip   | 60-100°     | 30%    |
| Squat    | Back lean | 15-50° | 20%    |
| Deadlift | Hip   | 30-70°      | 40%    |
| Deadlift | Knee  | 90-130°     | 30%    |
| Deadlift | Back lean | 30-60° | 30%  |
| Push-up  | Elbow | 70-110°     | 50%    |
| Push-up  | Body alignment | 160-180° | 50% |

## Development

```bash
git clone https://github.com/solakoglukoray/form-check
cd form-check
pip install -e ".[dev]"
pytest
```

## Contributing

PRs welcome. Run `ruff check .` and `pytest` before submitting.

## License

MIT
