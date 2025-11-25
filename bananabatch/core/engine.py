"""Core engine for BananaBatch batch image generation and editing.

This module contains the main orchestration logic including:
- BatchProcessor: Handles concurrent image generation
- BatchEditProcessor: Handles concurrent image editing
- TemplateEngine: Jinja2-based prompt templating
- FileManager: Input/output file handling and organization
"""

import asyncio
import csv
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Optional

import aiofiles
from jinja2 import Environment, TemplateError, UndefinedError

from bananabatch.core.models import (
    BatchProgress,
    EditJobConfig,
    EditType,
    ImageEditRequest,
    ImageMetadata,
    ImageRequest,
    ImageResult,
    JobConfig,
    JobStatus,
    ModelName,
)
from bananabatch.providers.base import ImageProvider, ProviderError


class TemplateEngine:
    """Jinja2-based template engine for prompt rendering.

    This engine handles variable substitution in prompt templates,
    allowing dynamic prompt generation from structured data.
    
    Supports both Jinja2 syntax {{ var }} and simple {var} syntax.
    """

    def __init__(self) -> None:
        """Initialize the template engine with Jinja2 environment."""
        self._env = Environment(
            autoescape=False,
            keep_trailing_newline=True,
        )

    def _normalize_template(self, template: str) -> str:
        """Convert simple {var} syntax to Jinja2 {{ var }} syntax.
        
        Args:
            template: Template string with either syntax.
            
        Returns:
            Template string with Jinja2 syntax.
        """
        import re
        # Match {word} but not {{ word }} (already Jinja2 syntax)
        # Negative lookbehind for { and negative lookahead for }
        pattern = r'(?<!\{)\{([a-zA-Z_][a-zA-Z0-9_]*)\}(?!\})'
        return re.sub(pattern, r'{{ \1 }}', template)

    def render(self, template: str, variables: dict[str, Any]) -> str:
        """Render a template with the given variables.

        Args:
            template: Jinja2 template string (or simple {var} syntax).
            variables: Dictionary of variables to substitute.

        Returns:
            The rendered template string.

        Raises:
            TemplateRenderError: If template rendering fails.
        """
        try:
            # Normalize template to Jinja2 syntax
            normalized = self._normalize_template(template)
            jinja_template = self._env.from_string(normalized)
            return jinja_template.render(**variables)
        except (TemplateError, UndefinedError) as e:
            raise TemplateRenderError(f"Failed to render template: {e}") from e

    def validate_template(self, template: str, sample_vars: dict[str, Any]) -> bool:
        """Validate that a template can be rendered with sample variables.

        Args:
            template: The template string to validate.
            sample_vars: Sample variables to test rendering.

        Returns:
            True if the template is valid.

        Raises:
            TemplateRenderError: If the template is invalid.
        """
        try:
            self.render(template, sample_vars)
            return True
        except Exception as e:
            raise TemplateRenderError(f"Invalid template: {e}") from e


class TemplateRenderError(Exception):
    """Exception raised when template rendering fails."""

    pass


class FileManager:
    """Handles file I/O operations for BananaBatch.

    This class manages:
    - Reading input files (CSV/JSON)
    - Creating output directories
    - Saving generated images
    - Writing metadata files
    """

    def __init__(self, output_base_dir: Path = Path("outputs")) -> None:
        """Initialize the file manager.

        Args:
            output_base_dir: Base directory for output files.
        """
        self.output_base_dir = output_base_dir
        self._output_dir: Optional[Path] = None

    @property
    def output_dir(self) -> Path:
        """Get the current output directory, creating if needed.

        Returns:
            Path to the timestamped output directory.
        """
        if self._output_dir is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
            self._output_dir = self.output_base_dir / timestamp
            self._output_dir.mkdir(parents=True, exist_ok=True)
        return self._output_dir

    def set_output_dir(self, path: Path) -> None:
        """Set a custom output directory.

        Args:
            path: The output directory path.
        """
        self._output_dir = path
        self._output_dir.mkdir(parents=True, exist_ok=True)

    async def read_input_file(self, file_path: Path) -> list[dict[str, Any]]:
        """Read an input file (CSV or JSON) and return records.

        Args:
            file_path: Path to the input file.

        Returns:
            List of dictionaries containing the input data.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file format is unsupported.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")

        suffix = file_path.suffix.lower()

        if suffix == ".csv":
            return await self._read_csv(file_path)
        elif suffix == ".json":
            return await self._read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    async def _read_csv(self, file_path: Path) -> list[dict[str, Any]]:
        """Read a CSV file asynchronously.

        Args:
            file_path: Path to the CSV file.

        Returns:
            List of row dictionaries.
        """
        async with aiofiles.open(file_path, mode="r", encoding="utf-8") as f:
            content = await f.read()

        # Parse CSV content
        lines = content.strip().split("\n")
        if not lines:
            return []

        reader = csv.DictReader(lines)
        return list(reader)

    async def _read_json(self, file_path: Path) -> list[dict[str, Any]]:
        """Read a JSON file asynchronously.

        Args:
            file_path: Path to the JSON file.

        Returns:
            List of record dictionaries.
        """
        async with aiofiles.open(file_path, mode="r", encoding="utf-8") as f:
            content = await f.read()

        data = json.loads(content)

        # Handle both array and object with "items" key
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "items" in data:
            return data["items"]
        elif isinstance(data, dict):
            # Single item, wrap in list
            return [data]
        else:
            raise ValueError("JSON must be an array or object with 'items' key")

    async def save_image(
        self,
        image_data: bytes,
        filename: str,
    ) -> Path:
        """Save image data to a file.

        Args:
            image_data: Raw image bytes.
            filename: The filename to save as.

        Returns:
            Path to the saved image file.
        """
        # Ensure filename has an extension
        if not any(filename.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp"]):
            filename = f"{filename}.png"

        output_path = self.output_dir / filename

        async with aiofiles.open(output_path, mode="wb") as f:
            await f.write(image_data)

        return output_path

    async def save_metadata(
        self,
        metadata: ImageMetadata,
        filename: str,
    ) -> Path:
        """Save metadata to a JSON file.

        Args:
            metadata: The metadata to save.
            filename: Base filename (will add .json extension).

        Returns:
            Path to the saved metadata file.
        """
        # Ensure filename has .json extension
        if not filename.lower().endswith(".json"):
            filename = f"{filename}.json"

        output_path = self.output_dir / filename

        async with aiofiles.open(output_path, mode="w", encoding="utf-8") as f:
            await f.write(metadata.model_dump_json(indent=2))

        return output_path


class BatchProcessor:
    """Orchestrates batch image generation with concurrent processing.

    This is the main engine that coordinates:
    - Reading input data
    - Rendering prompt templates
    - Concurrent image generation
    - Saving results and metadata
    - Progress tracking
    """

    def __init__(
        self,
        provider: ImageProvider,
        config: JobConfig,
        file_manager: Optional[FileManager] = None,
        template_engine: Optional[TemplateEngine] = None,
    ) -> None:
        """Initialize the batch processor.

        Args:
            provider: The image generation provider.
            config: Job configuration.
            file_manager: Optional file manager (created if not provided).
            template_engine: Optional template engine (created if not provided).
        """
        self.provider = provider
        self.config = config
        self.file_manager = file_manager or FileManager(config.output_dir)
        self.template_engine = template_engine or TemplateEngine()
        self.progress = BatchProgress()
        self._results: list[ImageResult] = []
        self._progress_callback: Optional[Callable[[BatchProgress], None]] = None

    def set_progress_callback(
        self,
        callback: Callable[[BatchProgress], None],
    ) -> None:
        """Set a callback function to receive progress updates.

        Args:
            callback: Function to call with progress updates.
        """
        self._progress_callback = callback

    def _notify_progress(self) -> None:
        """Notify the progress callback of current progress."""
        if self._progress_callback:
            self._progress_callback(self.progress)

    async def process(self) -> list[ImageResult]:
        """Run the batch processing job.

        Returns:
            List of ImageResult objects for all processed items.
        """
        # Read input data
        records = await self.file_manager.read_input_file(self.config.input_file)

        # Create image requests from records
        requests = self._create_requests(records)

        # Update progress
        self.progress.total = len(requests)
        self.progress.pending = len(requests)
        self._notify_progress()

        # Process requests concurrently with semaphore for rate limiting
        semaphore = asyncio.Semaphore(self.config.max_workers)

        async def process_with_semaphore(request: ImageRequest) -> ImageResult:
            async with semaphore:
                return await self._process_single(request)

        # Run all tasks concurrently
        tasks = [process_with_semaphore(req) for req in requests]
        self._results = await asyncio.gather(*tasks)

        return self._results

    async def process_stream(self) -> AsyncIterator[ImageResult]:
        """Run batch processing and yield results as they complete.

        Yields:
            ImageResult objects as each image completes.
        """
        # Read input data
        records = await self.file_manager.read_input_file(self.config.input_file)

        # Create image requests from records
        requests = self._create_requests(records)

        # Update progress
        self.progress.total = len(requests)
        self.progress.pending = len(requests)
        self._notify_progress()

        # Process requests concurrently with semaphore
        semaphore = asyncio.Semaphore(self.config.max_workers)

        async def process_with_semaphore(request: ImageRequest) -> ImageResult:
            async with semaphore:
                return await self._process_single(request)

        # Create tasks and process
        tasks = [asyncio.create_task(process_with_semaphore(req)) for req in requests]

        for coro in asyncio.as_completed(tasks):
            result = await coro
            self._results.append(result)
            yield result

    def _create_requests(self, records: list[dict[str, Any]]) -> list[ImageRequest]:
        """Create ImageRequest objects from input records.

        Args:
            records: List of input records.

        Returns:
            List of ImageRequest objects.
        """
        requests = []

        for i, record in enumerate(records):
            # Get prompt - either from template or direct 'prompt' field
            if self.config.prompt_template:
                prompt = self.template_engine.render(
                    self.config.prompt_template,
                    record,
                )
            elif "prompt" in record:
                prompt = record["prompt"]
            else:
                raise ValueError(f"Record {i} has no 'prompt' field and no template provided")

            # Get model from record or config
            model_str = record.get("model", self.config.model.value)
            try:
                model = ModelName(model_str)
            except ValueError:
                model = self.config.model

            # Get optional filename
            output_filename = record.get("filename", record.get("output_filename"))

            # Create request
            request = ImageRequest(
                id=record.get("id", str(uuid.uuid4())),
                prompt=prompt,
                template_vars=record,
                model=model,
                output_filename=output_filename,
            )
            requests.append(request)

        return requests

    async def _process_single(self, request: ImageRequest) -> ImageResult:
        """Process a single image generation request.

        Args:
            request: The image request to process.

        Returns:
            The result of the generation.
        """
        # Update progress
        self.progress.pending -= 1
        self.progress.in_progress += 1
        self._notify_progress()

        start_time = time.time()

        try:
            # Generate image
            image_data = await self.provider.generate(
                prompt=request.prompt,
                model=request.model.value,
            )

            generation_time = (time.time() - start_time) * 1000

            # Generate filename
            filename = request.output_filename or f"{request.id}"

            # Save image
            output_path = await self.file_manager.save_image(
                image_data,
                filename,
            )

            # Save metadata
            metadata = ImageMetadata(
                request_id=request.id,
                prompt=request.prompt,
                model=request.model.value,
                template_vars=request.template_vars,
                generation_time_ms=generation_time,
                output_filename=output_path.name,
            )

            metadata_path = await self.file_manager.save_metadata(
                metadata,
                f"{output_path.stem}_metadata",
            )

            # Update progress
            self.progress.in_progress -= 1
            self.progress.completed += 1
            self._notify_progress()

            return ImageResult(
                request_id=request.id,
                status=JobStatus.COMPLETED,
                output_path=output_path,
                metadata_path=metadata_path,
                prompt_used=request.prompt,
                model_used=request.model.value,
                generation_time_ms=generation_time,
            )

        except ProviderError as e:
            # Update progress
            self.progress.in_progress -= 1
            self.progress.failed += 1
            self._notify_progress()

            return ImageResult(
                request_id=request.id,
                status=JobStatus.FAILED,
                prompt_used=request.prompt,
                model_used=request.model.value,
                error_message=str(e),
            )

        except Exception as e:
            # Update progress
            self.progress.in_progress -= 1
            self.progress.failed += 1
            self._notify_progress()

            return ImageResult(
                request_id=request.id,
                status=JobStatus.FAILED,
                prompt_used=request.prompt,
                model_used=request.model.value,
                error_message=f"Unexpected error: {e}",
            )

    @property
    def results(self) -> list[ImageResult]:
        """Get all results from the batch job.

        Returns:
            List of ImageResult objects.
        """
        return self._results.copy()

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of the batch job results.

        Returns:
            Dictionary containing job summary statistics.
        """
        return {
            "total": self.progress.total,
            "completed": self.progress.completed,
            "failed": self.progress.failed,
            "success_rate": self.progress.success_rate,
            "output_directory": str(self.file_manager.output_dir),
        }


class BatchEditProcessor:
    """Orchestrates batch image editing with concurrent processing.

    This engine coordinates:
    - Reading input data with base images
    - Rendering edit prompt templates
    - Concurrent image editing
    - Saving results and metadata
    - Progress tracking
    """

    def __init__(
        self,
        provider: ImageProvider,
        config: EditJobConfig,
        file_manager: Optional[FileManager] = None,
        template_engine: Optional[TemplateEngine] = None,
    ) -> None:
        """Initialize the batch edit processor.

        Args:
            provider: The image generation/editing provider.
            config: Edit job configuration.
            file_manager: Optional file manager (created if not provided).
            template_engine: Optional template engine (created if not provided).
        """
        self.provider = provider
        self.config = config
        self.file_manager = file_manager or FileManager(config.output_dir)
        self.template_engine = template_engine or TemplateEngine()
        self.progress = BatchProgress()
        self._results: list[ImageResult] = []
        self._progress_callback: Optional[Callable[[BatchProgress], None]] = None

    def set_progress_callback(
        self,
        callback: Callable[[BatchProgress], None],
    ) -> None:
        """Set a callback function to receive progress updates.

        Args:
            callback: Function to call with progress updates.
        """
        self._progress_callback = callback

    def _notify_progress(self) -> None:
        """Notify the progress callback of current progress."""
        if self._progress_callback:
            self._progress_callback(self.progress)

    async def process(self) -> list[ImageResult]:
        """Run the batch editing job.

        Returns:
            List of ImageResult objects for all processed items.
        """
        records = await self.file_manager.read_input_file(self.config.input_file)
        requests = self._create_edit_requests(records)

        self.progress.total = len(requests)
        self.progress.pending = len(requests)
        self._notify_progress()

        semaphore = asyncio.Semaphore(self.config.max_workers)

        async def process_with_semaphore(request: ImageEditRequest) -> ImageResult:
            async with semaphore:
                return await self._process_single_edit(request)

        tasks = [process_with_semaphore(req) for req in requests]
        self._results = await asyncio.gather(*tasks)

        return self._results

    async def process_stream(self) -> AsyncIterator[ImageResult]:
        """Run batch editing and yield results as they complete.

        Yields:
            ImageResult objects as each edit completes.
        """
        records = await self.file_manager.read_input_file(self.config.input_file)
        requests = self._create_edit_requests(records)

        self.progress.total = len(requests)
        self.progress.pending = len(requests)
        self._notify_progress()

        semaphore = asyncio.Semaphore(self.config.max_workers)

        async def process_with_semaphore(request: ImageEditRequest) -> ImageResult:
            async with semaphore:
                return await self._process_single_edit(request)

        tasks = [asyncio.create_task(process_with_semaphore(req)) for req in requests]

        for coro in asyncio.as_completed(tasks):
            result = await coro
            self._results.append(result)
            yield result

    def _create_edit_requests(self, records: list[dict[str, Any]]) -> list[ImageEditRequest]:
        """Create ImageEditRequest objects from input records.

        Args:
            records: List of input records.

        Returns:
            List of ImageEditRequest objects.
        """
        requests = []

        for i, record in enumerate(records):
            # Get base image path - check CSV fields first, then fall back to default_image
            base_image_str = record.get("base_image", record.get("image", record.get("input_image")))
            if not base_image_str:
                # Use default_image if no base_image in CSV
                if self.config.default_image:
                    base_image_str = self.config.default_image
                else:
                    raise ValueError(
                        f"Record {i} has no 'base_image' field and no --default-image provided"
                    )

            base_image = Path(base_image_str)
            # If relative path and base_image_dir is set, resolve against it
            if not base_image.is_absolute() and self.config.base_image_dir:
                base_image = self.config.base_image_dir / base_image

            # Get edit prompt - either from template or direct field
            if self.config.edit_template:
                edit_prompt = self.template_engine.render(
                    self.config.edit_template,
                    record,
                )
            elif "edit_prompt" in record:
                edit_prompt = record["edit_prompt"]
            elif "prompt" in record:
                edit_prompt = record["prompt"]
            else:
                raise ValueError(f"Record {i} has no 'edit_prompt' field and no template provided")

            # Get optional mask image
            mask_image = None
            if "mask_image" in record or "mask" in record:
                mask_str = record.get("mask_image", record.get("mask"))
                if mask_str:
                    mask_image = Path(mask_str)
                    if not mask_image.is_absolute() and self.config.base_image_dir:
                        mask_image = self.config.base_image_dir / mask_image

            # Get edit type
            edit_type_str = record.get("edit_type", self.config.default_edit_type.value)
            try:
                edit_type = EditType(edit_type_str)
            except ValueError:
                edit_type = self.config.default_edit_type

            # Get strength
            strength = float(record.get("strength", self.config.default_strength))

            # Get model
            model_str = record.get("model", self.config.model.value)
            try:
                model = ModelName(model_str)
            except ValueError:
                model = self.config.model

            # Get optional output filename
            output_filename = record.get("filename", record.get("output_filename"))

            request = ImageEditRequest(
                id=record.get("id", str(uuid.uuid4())),
                base_image=base_image,
                edit_prompt=edit_prompt,
                mask_image=mask_image,
                edit_type=edit_type,
                strength=strength,
                template_vars=record,
                model=model,
                output_filename=output_filename,
            )
            requests.append(request)

        return requests

    async def _process_single_edit(self, request: ImageEditRequest) -> ImageResult:
        """Process a single image edit request.

        Args:
            request: The image edit request to process.

        Returns:
            The result of the edit.
        """
        self.progress.pending -= 1
        self.progress.in_progress += 1
        self._notify_progress()

        start_time = time.time()

        try:
            # Edit image
            image_data = await self.provider.edit(
                base_image=request.base_image,
                prompt=request.edit_prompt,
                mask_image=request.mask_image,
                model=request.model.value,
                strength=request.strength,
            )

            generation_time = (time.time() - start_time) * 1000

            # Generate filename
            if request.output_filename:
                filename = request.output_filename
            else:
                # Create filename from base image name + edit type
                base_stem = request.base_image.stem
                filename = f"{base_stem}_{request.edit_type.value}_{request.id[:8]}"

            # Save image
            output_path = await self.file_manager.save_image(
                image_data,
                filename,
            )

            # Save metadata
            metadata = ImageMetadata(
                request_id=request.id,
                prompt=request.edit_prompt,
                model=request.model.value,
                template_vars=request.template_vars,
                generation_time_ms=generation_time,
                output_filename=output_path.name,
                base_image=str(request.base_image),
                edit_type=request.edit_type.value,
            )

            metadata_path = await self.file_manager.save_metadata(
                metadata,
                f"{output_path.stem}_metadata",
            )

            self.progress.in_progress -= 1
            self.progress.completed += 1
            self._notify_progress()

            return ImageResult(
                request_id=request.id,
                status=JobStatus.COMPLETED,
                output_path=output_path,
                metadata_path=metadata_path,
                prompt_used=request.edit_prompt,
                model_used=request.model.value,
                generation_time_ms=generation_time,
            )

        except ProviderError as e:
            self.progress.in_progress -= 1
            self.progress.failed += 1
            self._notify_progress()

            return ImageResult(
                request_id=request.id,
                status=JobStatus.FAILED,
                prompt_used=request.edit_prompt,
                model_used=request.model.value,
                error_message=str(e),
            )

        except Exception as e:
            self.progress.in_progress -= 1
            self.progress.failed += 1
            self._notify_progress()

            return ImageResult(
                request_id=request.id,
                status=JobStatus.FAILED,
                prompt_used=request.edit_prompt,
                model_used=request.model.value,
                error_message=f"Unexpected error: {e}",
            )

    @property
    def results(self) -> list[ImageResult]:
        """Get all results from the batch edit job.

        Returns:
            List of ImageResult objects.
        """
        return self._results.copy()

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of the batch edit job results.

        Returns:
            Dictionary containing job summary statistics.
        """
        return {
            "total": self.progress.total,
            "completed": self.progress.completed,
            "failed": self.progress.failed,
            "success_rate": self.progress.success_rate,
            "output_directory": str(self.file_manager.output_dir),
        }
