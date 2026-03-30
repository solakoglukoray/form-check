"""CLI entry point for form-check."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from form_check.main import analyze_video

app = typer.Typer(
    help="Score your gym exercise form using AI pose analysis.",
    add_completion=False,
)
console = Console()

EXERCISES = ["squat", "deadlift", "pushup"]


@app.command()
def analyze(
    video: str = typer.Argument(
        ...,
        help="Path to the workout video file (MP4, MOV, AVI).",
    ),
    exercise: str = typer.Option(
        ...,
        "--exercise",
        "-e",
        help="Exercise type: squat, deadlift, or pushup.",
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Directory to save annotated frames (optional).",
    ),
    sample_rate: int = typer.Option(
        10,
        "--sample-rate",
        "-s",
        help="Analyze every Nth frame. Lower = more accurate, slower (default: 10).",
    ),
) -> None:
    """Analyze a workout video and score your exercise form (0-100)."""
    exercise = exercise.lower()
    if exercise not in EXERCISES:
        console.print(
            f"[red]Unknown exercise '{exercise}'. "
            f"Choose from: {', '.join(EXERCISES)}[/red]"
        )
        raise typer.Exit(1)

    if not Path(video).exists():
        console.print(f"[red]Video file not found: {video}[/red]")
        raise typer.Exit(1)

    console.print(f"\n[cyan]Analyzing {exercise} form...[/cyan]")
    console.print(f"[dim]Video: {video}[/dim]\n")

    try:
        results = analyze_video(video, exercise, output_dir, sample_rate)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    except Exception as exc:
        console.print(f"[red]Analysis failed: {exc}[/red]")
        raise typer.Exit(1)

    score = results["avg_score"]
    if score >= 80:
        color, grade = "green", "Excellent"
    elif score >= 60:
        color, grade = "yellow", "Needs Work"
    else:
        color, grade = "red", "Poor Form"

    console.print(
        Panel(
            f"[bold {color}]Form Score: {score}/100 — {grade}[/bold {color}]",
            title=f"[bold]{exercise.upper()} RESULT[/bold]",
            border_style=color,
        )
    )

    if results["avg_angles"]:
        table = Table(
            title="Average Joint Angles",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Joint", style="cyan", min_width=12)
        table.add_column("Avg Angle", justify="right")
        for joint, angle in results["avg_angles"].items():
            table.add_row(joint.capitalize(), f"{angle:.1f} deg")
        console.print(table)

    console.print("\n[bold]Feedback:[/bold]")
    for line in results["feedback"]:
        marker = "[green]+[/green]" if "Good" in line else "[yellow]![/yellow]"
        console.print(f"  {marker} {line}")

    console.print(
        f"\n[dim]Frames analyzed: {results['frames_analyzed']} | "
        f"Min: {results['min_score']} | Max: {results['max_score']}[/dim]"
    )

    if results.get("annotated_frames"):
        count = len(results["annotated_frames"])
        console.print(
            f"\n[green]Saved {count} annotated frames -> {output_dir}[/green]"
        )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
