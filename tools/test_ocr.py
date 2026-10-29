from pathlib import Path
from time import perf_counter

import cv2
import easyocr
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich.table import Table


console = Console()


def pick_best_result(results):
    if not results:
        return None
    return max(results, key=lambda item: item[2])


def test_ocr():
    reader = easyocr.Reader(["en"], gpu=True)
    test_dir = Path(__file__).resolve().parent.parent / "datasets" / "test-dataset-ocr"
    image_paths = sorted(test_dir.glob("*.jpg"))

    if not image_paths:
        console.print(
            Panel(
                f"No JPG images were found in [bold]{test_dir.name}[/bold].",
                title="OCR Run",
                style="yellow",
            )
        )
        return

    total_start = perf_counter()
    total_ocr_time = 0.0
    recognized_images = []
    unrecognized_images = []
    failed_images = []

    console.print(
        Panel(
            f"[bold cyan]Starting OCR test[/bold cyan]\n"
            f"Images queued: [bold]{len(image_paths)}[/bold]\n"
            f"Language: [bold]English[/bold] | Allowlist: [bold]0123456789[/bold]",
            title="OCR Feedback",
            border_style="cyan",
        )
    )

    for img_path in track(image_paths, description="Running OCR on images"):
        image = cv2.imread(str(img_path))

        if image is None:
            failed_images.append(img_path.name)
            console.print(f"[red]Failed to read image:[/red] {img_path.name}")
            continue

        start = perf_counter()
        results = reader.readtext(image, allowlist="0123456789")
        elapsed = perf_counter() - start
        total_ocr_time += elapsed

        best_result = pick_best_result(results)
        if best_result is None:
            unrecognized_images.append(img_path.name)
            console.print(f"[yellow]{img_path.name}[/yellow] -> [dim]No number recognized[/dim] ({elapsed:.2f}s)")
            continue

        _, text, confidence = best_result
        recognized_images.append(
            {
                "file": img_path.name,
                "number": text,
                "confidence": confidence,
                "time": elapsed,
            }
        )
        console.print(
            f"[green]{img_path.name}[/green] -> [bold]{text}[/bold] "
            f"[dim](confidence {confidence:.2%}, {elapsed:.2f}s)[/dim]"
        )

    total_elapsed = perf_counter() - total_start
    ocr_input_count = len(recognized_images) + len(unrecognized_images)
    loaded_images = ocr_input_count + len(failed_images)
    recognized_count = len(recognized_images)
    no_recognition_count = len(unrecognized_images)
    failed_count = len(failed_images)
    average_time = total_ocr_time / ocr_input_count if ocr_input_count else 0.0
    average_recognized_time = (
        sum(item["time"] for item in recognized_images) / recognized_count if recognized_count else 0.0
    )
    average_unrecognized_time = (
        total_ocr_time - sum(item["time"] for item in recognized_images)
    ) / no_recognition_count if no_recognition_count else 0.0

    summary_table = Table.grid(padding=(0, 2))
    summary_table.add_column(style="cyan", justify="right")
    summary_table.add_column(style="white")
    summary_table.add_row("Images queued", str(len(image_paths)))
    summary_table.add_row("Images loaded successfully", str(loaded_images - failed_count))
    summary_table.add_row("Images sent to OCR", str(ocr_input_count))
    summary_table.add_row("Images recognized", str(recognized_count))
    summary_table.add_row("Images not recognized", str(no_recognition_count))
    summary_table.add_row("Images failed to load", str(failed_count))
    summary_table.add_row("Total OCR time", f"{total_ocr_time:.2f}s")
    summary_table.add_row("Total wall time", f"{total_elapsed:.2f}s")
    summary_table.add_row("Average OCR time per OCR input image", f"{average_time:.2f}s")
    summary_table.add_row("Average time per recognized image", f"{average_recognized_time:.2f}s")
    summary_table.add_row("Average time per unrecognized image", f"{average_unrecognized_time:.2f}s")

    console.print()
    console.print(Panel(summary_table, title="OCR Summary", border_style="green", box=box.ROUNDED))

    if recognized_images:
        results_table = Table(title="Recognized Images", box=box.SIMPLE_HEAVY, show_lines=False)
        results_table.add_column("File", style="bold white", overflow="fold")
        results_table.add_column("Recognized Number", style="bold green", justify="center")
        results_table.add_column("Confidence", style="cyan", justify="right")
        results_table.add_column("OCR Time", style="magenta", justify="right")

        for item in recognized_images:
            results_table.add_row(
                item["file"],
                item["number"],
                f"{item['confidence']:.2%}",
                f"{item['time']:.2f}s",
            )

        console.print(results_table)
    else:
        console.print("[yellow]No images produced a number that could be recognized.[/yellow]")

    if failed_images:
        console.print()
        console.print(Panel("\n".join(failed_images), title="Failed Images", border_style="red"))


if __name__ == "__main__":
    test_ocr()