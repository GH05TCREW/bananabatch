"""BananaBatch CLI application.

This module provides the command-line interface for BananaBatch,
built with Typer and Rich for beautiful terminal output.
"""

import asyncio
import os
from pathlib import Path
from typing import Annotated, Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from bananabatch.core.engine import BatchEditProcessor, BatchProcessor, FileManager, TemplateEngine
from bananabatch.core.models import EditJobConfig, EditType, JobConfig, JobStatus, ModelName
from bananabatch.providers.gemini import GeminiProvider

# Load environment variables
load_dotenv()

# Initialize Typer app and Rich console
app = typer.Typer(
    name="bananabatch",
    help="🍌 BananaBatch - Batch image processing using Nano Banana",
    add_completion=False,
)
console = Console()


def get_api_key(api_key: Optional[str] = None) -> str:
    """Get the API key from argument or environment variable.

    Args:
        api_key: Optional API key passed as argument.

    Returns:
        The API key to use.

    Raises:
        typer.Exit: If no API key is found.
    """
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        console.print(
            "[red]Error:[/red] No API key provided. "
            "Set GEMINI_API_KEY environment variable or use --api-key option."
        )
        raise typer.Exit(1)
    return key


def print_banner() -> None:
    """Print the BananaBatch banner."""
    banner = """
[yellow]🍌 BananaBatch[/yellow]
[dim]Batch Image Processing with Nano Banana[/dim]
    """
    console.print(Panel(banner.strip(), border_style="yellow"))


def print_results_table(results: list, output_dir: Path) -> None:
    """Print a summary table of generation results.

    Args:
        results: List of ImageResult objects.
        output_dir: Path to the output directory.
    """
    table = Table(title="Generation Results", show_header=True, header_style="bold cyan")
    table.add_column("ID", style="dim")
    table.add_column("Status", justify="center")
    table.add_column("Model")
    table.add_column("Time (ms)", justify="right")
    table.add_column("Output File")

    for result in results:
        status_icon = "✓" if result.status == JobStatus.COMPLETED else "✗"
        status_style = "green" if result.status == JobStatus.COMPLETED else "red"

        time_str = f"{result.generation_time_ms:.0f}" if result.generation_time_ms else "-"
        output_file = result.output_path.name if result.output_path else result.error_message or "-"

        table.add_row(
            result.request_id[:8],
            f"[{status_style}]{status_icon}[/{status_style}]",
            result.model_used.split("-")[1] if "-" in result.model_used else result.model_used,
            time_str,
            output_file[:40],
        )

    console.print(table)
    console.print(f"\n[dim]Output directory:[/dim] {output_dir}")


@app.command()
def generate(
    input_file: Annotated[
        Path,
        typer.Option(
            "--input",
            "-i",
            help="Path to input CSV or JSON file with prompts",
            exists=True,
            readable=True,
        ),
    ],
    model: Annotated[
        str,
        typer.Option(
            "--model",
            "-m",
            help="Gemini model to use for generation",
        ),
    ] = "gemini-2.5-flash-image",
    output_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Output directory for generated images",
        ),
    ] = None,
    template: Annotated[
        Optional[str],
        typer.Option(
            "--template",
            "-t",
            help="Jinja2 template for prompts (e.g., 'A photo of {{ item }}')",
        ),
    ] = None,
    max_workers: Annotated[
        int,
        typer.Option(
            "--workers",
            "-w",
            help="Maximum concurrent generations (1-20)",
            min=1,
            max=20,
        ),
    ] = 5,
    api_key: Annotated[
        Optional[str],
        typer.Option(
            "--api-key",
            "-k",
            help="Gemini API key (or set GEMINI_API_KEY env var)",
            envvar="GEMINI_API_KEY",
        ),
    ] = None,
) -> None:
    """Generate images from a batch of prompts.

    Read prompts from a CSV or JSON file and generate images using
    Google Gemini. Supports concurrent generation and Jinja2 templating.

    Example usage:
        bananabatch generate --input prompts.csv --model gemini-2.5-flash-image
        bananabatch generate -i ideas.json -t "A photo of {{ item }} in {{ style }} style"
    """
    print_banner()

    # Validate API key
    key = get_api_key(api_key)

    # Validate model
    try:
        model_enum = ModelName(model)
    except ValueError:
        console.print(f"[red]Error:[/red] Invalid model '{model}'")
        console.print(f"[dim]Supported models: {[m.value for m in ModelName]}[/dim]")
        raise typer.Exit(1)

    # Create job config
    try:
        config = JobConfig(
            input_file=input_file,
            output_dir=output_dir or Path("outputs"),
            prompt_template=template,
            model=model_enum,
            max_workers=max_workers,
            api_key=key,
        )
    except Exception as e:
        console.print(f"[red]Configuration Error:[/red] {e}")
        raise typer.Exit(1)

    console.print(f"[dim]Input file:[/dim] {input_file}")
    console.print(f"[dim]Model:[/dim] {model}")
    console.print(f"[dim]Max workers:[/dim] {max_workers}")
    if template:
        console.print(f"[dim]Template:[/dim] {template}")
    console.print()

    # Initialize components
    provider = GeminiProvider(api_key=key, default_model=model)
    file_manager = FileManager(config.output_dir)
    processor = BatchProcessor(
        provider=provider,
        config=config,
        file_manager=file_manager,
    )

    # Run with progress bar
    try:
        asyncio.run(_run_with_progress(processor, file_manager))
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user. Exiting...[/yellow]")
        raise typer.Exit(130)


async def _run_with_progress(processor: BatchProcessor, file_manager: FileManager) -> None:
    """Run the batch processor with a Rich progress bar.

    Args:
        processor: The batch processor to run.
        file_manager: The file manager for output paths.
    """
    from rich.live import Live
    from rich.spinner import Spinner

    results = []
    completed_count = 0

    # Show spinner while loading
    with Live(Spinner("dots", text="[cyan]Loading..."), console=console, refresh_per_second=10) as live:
        async for result in processor.process_stream():
            results.append(result)
            completed_count += 1

            # On first result, switch to progress bar
            if completed_count == 1:
                progress = Progress(
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    MofNCompleteColumn(),
                    TaskProgressColumn(),
                    TimeElapsedColumn(),
                    console=console,
                )
                task = progress.add_task("[cyan]Processing...", total=processor.progress.total)
                live.update(progress)

            # Build description based on status
            if result.status == JobStatus.COMPLETED:
                desc = f"[green]✓[/green] {result.request_id[:8]}..."
            else:
                desc = f"[red]✗[/red] {result.request_id[:8]}..."

            progress.update(task, completed=completed_count, description=desc)

    console.print()

    # Print results table
    print_results_table(results, file_manager.output_dir)

    # Print summary
    summary = processor.get_summary()
    console.print()

    if summary["failed"] == 0:
        console.print(
            f"[green]✓ Successfully generated {summary['completed']} images![/green]"
        )
    else:
        console.print(
            f"[yellow]Completed: {summary['completed']} succeeded, "
            f"{summary['failed']} failed[/yellow]"
        )


@app.command()
def validate(
    input_file: Annotated[
        Path,
        typer.Option(
            "--input",
            "-i",
            help="Path to input CSV or JSON file to validate",
            exists=True,
            readable=True,
        ),
    ],
    template: Annotated[
        Optional[str],
        typer.Option(
            "--template",
            "-t",
            help="Jinja2 template to validate against the input",
        ),
    ] = None,
) -> None:
    """Validate an input file and optional template.

    Check that the input file is properly formatted and that the
    template (if provided) can be rendered with the input data.
    """
    print_banner()

    console.print(f"[dim]Validating:[/dim] {input_file}")

    # Read and validate input file
    file_manager = FileManager()

    try:
        records = asyncio.run(file_manager.read_input_file(input_file))
        console.print(f"[green]✓[/green] Found {len(records)} records")
    except Exception as e:
        console.print(f"[red]✗ Error reading file:[/red] {e}")
        raise typer.Exit(1)

    # Show sample record
    if records:
        console.print("\n[dim]Sample record:[/dim]")
        sample = records[0]
        for key, value in sample.items():
            console.print(f"  {key}: {value}")

    # Validate template if provided
    if template:
        console.print(f"\n[dim]Validating template:[/dim] {template}")
        template_engine = TemplateEngine()

        try:
            if records:
                rendered = template_engine.render(template, records[0])
                console.print(f"[green]✓[/green] Template valid")
                console.print(f"[dim]Sample output:[/dim] {rendered[:100]}...")
            else:
                console.print("[yellow]Warning: No records to test template with[/yellow]")
        except Exception as e:
            console.print(f"[red]✗ Template error:[/red] {e}")
            raise typer.Exit(1)

    console.print("\n[green]Validation complete![/green]")


@app.command()
def models() -> None:
    """List available Gemini models for image generation."""
    print_banner()

    table = Table(title="Available Models", show_header=True, header_style="bold cyan")
    table.add_column("Model ID", style="green")
    table.add_column("Description")

    table.add_row(
        ModelName.GEMINI_FLASH.value,
        "Gemini 2.5 Flash - Fast image generation (default)",
    )
    table.add_row(
        ModelName.GEMINI_PRO.value,
        "Gemini 3 Pro - Higher quality image generation (preview)",
    )

    console.print(table)


@app.command()
def edit(
    input_file: Annotated[
        Path,
        typer.Option(
            "--input",
            "-i",
            help="Path to input CSV or JSON file with edit instructions",
            exists=True,
            readable=True,
        ),
    ],
    base_image_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--images",
            "-d",
            help="Directory containing base images (if using relative paths in CSV)",
            exists=True,
        ),
    ] = None,
    default_image: Annotated[
        Optional[str],
        typer.Option(
            "--default-image",
            "-b",
            help="Default image filename to use for all rows (when base_image column is missing)",
        ),
    ] = None,
    model: Annotated[
        str,
        typer.Option(
            "--model",
            "-m",
            help="Gemini model to use for editing",
        ),
    ] = "gemini-2.5-flash-image",
    output_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Output directory for edited images",
        ),
    ] = None,
    template: Annotated[
        Optional[str],
        typer.Option(
            "--template",
            "-t",
            help="Jinja2 template for edit prompts (e.g., 'Add {anomaly_type} to the image')",
        ),
    ] = None,
    edit_type: Annotated[
        str,
        typer.Option(
            "--type",
            "-e",
            help="Default edit type: transform, inpaint, style_transfer, anomaly, variation",
        ),
    ] = "transform",
    strength: Annotated[
        float,
        typer.Option(
            "--strength",
            "-s",
            help="Edit strength (0.0 to 1.0)",
            min=0.0,
            max=1.0,
        ),
    ] = 0.75,
    max_workers: Annotated[
        int,
        typer.Option(
            "--workers",
            "-w",
            help="Maximum concurrent edits (1-20)",
            min=1,
            max=20,
        ),
    ] = 2,
    api_key: Annotated[
        Optional[str],
        typer.Option(
            "--api-key",
            "-k",
            help="Gemini API key (or set GEMINI_API_KEY env var)",
            envvar="GEMINI_API_KEY",
        ),
    ] = None,
) -> None:
    """Batch edit images with AI-powered transformations.

    Read edit instructions from a CSV or JSON file and apply edits to base images
    using Google Gemini. Supports inpainting, style transfer, anomaly generation,
    and general transformations.

    Example usage:
        bananabatch edit --input edits.csv --images ./base_images/
        bananabatch edit -i defects.csv -d ./products/ -t "Add a {defect_type} to this image"
        bananabatch edit -i styles.csv --type style_transfer --strength 0.8
    """
    print_banner()
    console.print("[bold]Batch Image Editing Mode[/bold]\n")

    # Validate API key
    key = get_api_key(api_key)

    # Validate model
    try:
        model_enum = ModelName(model)
    except ValueError:
        console.print(f"[red]Error:[/red] Invalid model '{model}'")
        console.print(f"[dim]Supported models: {[m.value for m in ModelName]}[/dim]")
        raise typer.Exit(1)

    # Validate edit type
    try:
        edit_type_enum = EditType(edit_type)
    except ValueError:
        console.print(f"[red]Error:[/red] Invalid edit type '{edit_type}'")
        console.print(f"[dim]Supported types: {[e.value for e in EditType]}[/dim]")
        raise typer.Exit(1)

    # Create edit job config
    try:
        config = EditJobConfig(
            input_file=input_file,
            base_image_dir=base_image_dir,
            default_image=default_image,
            output_dir=output_dir or Path("outputs"),
            edit_template=template,
            default_edit_type=edit_type_enum,
            default_strength=strength,
            model=model_enum,
            max_workers=max_workers,
            api_key=key,
        )
    except Exception as e:
        console.print(f"[red]Configuration Error:[/red] {e}")
        raise typer.Exit(1)

    console.print(f"[dim]Input file:[/dim] {input_file}")
    if base_image_dir:
        console.print(f"[dim]Base images:[/dim] {base_image_dir}")
    if default_image:
        console.print(f"[dim]Default image:[/dim] {default_image}")
    console.print(f"[dim]Model:[/dim] {model}")
    console.print(f"[dim]Edit type:[/dim] {edit_type}")
    console.print(f"[dim]Strength:[/dim] {strength}")
    console.print(f"[dim]Max workers:[/dim] {max_workers}")
    if template:
        console.print(f"[dim]Template:[/dim] {template}")
    console.print()

    # Initialize components
    provider = GeminiProvider(api_key=key, default_model=model)
    file_manager = FileManager(config.output_dir)
    processor = BatchEditProcessor(
        provider=provider,
        config=config,
        file_manager=file_manager,
    )

    # Run with progress bar
    try:
        asyncio.run(_run_edit_with_progress(processor, file_manager))
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user. Exiting...[/yellow]")
        raise typer.Exit(130)


async def _run_edit_with_progress(processor: BatchEditProcessor, file_manager: FileManager) -> None:
    """Run the batch edit processor with a Rich progress bar.

    Args:
        processor: The batch edit processor to run.
        file_manager: The file manager for output paths.
    """
    from rich.live import Live
    from rich.spinner import Spinner

    results = []
    completed_count = 0

    # Show spinner while loading
    with Live(Spinner("dots", text="[cyan]Loading..."), console=console, refresh_per_second=10) as live:
        async for result in processor.process_stream():
            results.append(result)
            completed_count += 1

            # On first result, switch to progress bar
            if completed_count == 1:
                progress = Progress(
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    MofNCompleteColumn(),
                    TaskProgressColumn(),
                    TimeElapsedColumn(),
                    console=console,
                )
                task = progress.add_task("[cyan]Editing...", total=processor.progress.total)
                live.update(progress)

            # Build description based on status
            if result.status == JobStatus.COMPLETED:
                desc = f"[green]✓[/green] {result.request_id[:8]}..."
            else:
                desc = f"[red]✗[/red] {result.request_id[:8]}..."

            progress.update(task, completed=completed_count, description=desc)

    console.print()

    # Print results table
    print_edit_results_table(results, file_manager.output_dir)

    # Print summary
    summary = processor.get_summary()
    console.print()

    if summary["failed"] == 0:
        console.print(
            f"[green]✓ Successfully edited {summary['completed']} images![/green]"
        )
    else:
        console.print(
            f"[yellow]Completed: {summary['completed']} succeeded, "
            f"{summary['failed']} failed[/yellow]"
        )


def print_edit_results_table(results: list, output_dir: Path) -> None:
    """Print a summary table of edit results.

    Args:
        results: List of ImageResult objects.
        output_dir: Path to the output directory.
    """
    table = Table(title="Edit Results", show_header=True, header_style="bold cyan")
    table.add_column("ID", style="dim")
    table.add_column("Status", justify="center")
    table.add_column("Model")
    table.add_column("Time (ms)", justify="right")
    table.add_column("Output / Error", max_width=60)

    for result in results:
        status_icon = "✓" if result.status == JobStatus.COMPLETED else "✗"
        status_style = "green" if result.status == JobStatus.COMPLETED else "red"

        time_str = f"{result.generation_time_ms:.0f}" if result.generation_time_ms else "-"
        output_file = result.output_path.name if result.output_path else result.error_message or "-"

        table.add_row(
            result.request_id[:8],
            f"[{status_style}]{status_icon}[/{status_style}]",
            result.model_used.split("-")[1] if "-" in result.model_used else result.model_used,
            time_str,
            output_file,
        )

    console.print(table)
    console.print(f"\n[dim]Output directory:[/dim] {output_dir}")


@app.command()
def version() -> None:
    """Show the BananaBatch version."""
    from bananabatch import __version__

    console.print(f"BananaBatch v{__version__}")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-v",
            help="Show version and exit",
            is_eager=True,
        ),
    ] = False,
) -> None:
    """🍌 BananaBatch - Batch image processing using Nano Banana."""
    if version:
        from bananabatch import __version__

        console.print(f"BananaBatch v{__version__}")
        raise typer.Exit()

    if ctx.invoked_subcommand is None:
        print_banner()
        console.print("Use [cyan]bananabatch --help[/cyan] to see available commands.")


if __name__ == "__main__":
    app()
