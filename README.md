# form-check

[![CI](https://github.com/solakoglukoray/form-check/actions/workflows/ci.yml/badge.svg)](https://github.com/solakoglukoray/form-check/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Point a camera at your workout, get a form score out of 100 — AI checks your squat, deadlift, and push-up against biomechanical benchmarks using MediaPipe Pose estimation.

No gym trainer? No problem. `form-check` analyzes joint angles in any workout video and tells you exactly what's off — knees caving, hips sagging, back rounding — with a per-rep score and frame-by-frame annotated output.

## Features

- **Pose detection** — MediaPipe Pose extracts 33 body landmarks per frame at 30fps
- **Biomechanical scoring** — knee, hip, and back angles compared against evidence-based ranges for each exercise
- **Annotated frames** — save overlaid JPEG frames with joint angles and score burned in
- **Three exercises** — squat, deadlift, push-up (more coming)
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

# Deadlift with annotated frame output
form-check analyze deadlift.mp4 --exercise deadlift --output ./frames

# Push-up — analyze every 5th frame for more granularity
form-check analyze pushup.mp4 --exercise pushup --sample-rate 5
```

### Example output

```
Analyzing pushup form...
Video: pushup.mp4

+-------------------------------+
|       PUSHUP RESULT          |
|  Form Score: 84/100 - Excellent |
+-------------------------------+

Average Joint Angles
 Joint    Avg Angle
 Elbow    94.3 deg
 Hip     167.8 deg

Feedback:
  + Elbow angle at bottom: Good (94.3 deg)
  + Body alignment (hip angle): Good (167.8 deg)

Frames analyzed: 47 | Min: 71 | Max: 100
```

## How It Works

1. **Frame sampling** — every Nth frame is extracted from the video (default: every 10th)
2. **Pose estimation** — MediaPipe Pose locates 33 body landmarks per frame
3. **Angle calculation** — relevant joint angles are computed using 2D landmark coordinates
4. **Benchmark comparison** — angles are scored against evidence-based ranges weighted by joint importance
5. **Aggregation** — per-frame scores are averaged; annotated frames are written if `--output` is set

### Biomechanical benchmarks

| Exercise | Joint | Target Range | Weight |
|----------|-------|-------------|--------|
| Squat    | Knee  | 70-120°     | 50%    |
| Squat    | Hip   | 60-100°     | 30%    |
| Squat    | Back lean | 15-50° | 20%    |
| Deadlift | Hip   | 30-70°      | 40%    |
| Deadlift | Knee  | 90-130°     | 30%    |
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
