"""Pydantic models for BananaBatch configuration and data structures.

This module defines the core data models used throughout the application
for validation, serialization, and type safety.
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class ModelName(str, Enum):
    """Supported Gemini model names for image generation."""

    GEMINI_FLASH = "gemini-2.5-flash-image"
    GEMINI_PRO = "gemini-3-pro-image-preview"


class JobStatus(str, Enum):
    """Status of an image generation job."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class ImageRequest(BaseModel):
    """Request model for a single image generation task.

    Attributes:
        id: Unique identifier for this request.
        prompt: The text prompt for image generation.
        template_vars: Variables to substitute into the prompt template.
        model: The Gemini model to use for generation.
        output_filename: Optional custom filename for the output image.
    """

    id: str = Field(..., description="Unique identifier for this request")
    prompt: str = Field(..., description="The text prompt for image generation")
    template_vars: dict[str, Any] = Field(
        default_factory=dict,
        description="Variables to substitute into the prompt template",
    )
    model: ModelName = Field(
        default=ModelName.GEMINI_FLASH,
        description="The Gemini model to use for generation",
    )
    output_filename: Optional[str] = Field(
        default=None,
        description="Optional custom filename for the output image",
    )

    @field_validator("prompt")
    @classmethod
    def prompt_not_empty(cls, v: str) -> str:
        """Validate that the prompt is not empty or whitespace."""
        if not v or not v.strip():
            raise ValueError("Prompt cannot be empty")
        return v.strip()


class ImageResult(BaseModel):
    """Result model for a completed image generation task.

    Attributes:
        request_id: The ID of the original request.
        status: The status of the generation job.
        output_path: Path to the saved image file (if successful).
        metadata_path: Path to the metadata JSON file (if successful).
        prompt_used: The final prompt after template rendering.
        model_used: The model that was used for generation.
        error_message: Error message if generation failed.
        generation_time_ms: Time taken to generate the image in milliseconds.
        created_at: Timestamp when the result was created.
    """

    request_id: str = Field(..., description="The ID of the original request")
    status: JobStatus = Field(..., description="The status of the generation job")
    output_path: Optional[Path] = Field(
        default=None,
        description="Path to the saved image file",
    )
    metadata_path: Optional[Path] = Field(
        default=None,
        description="Path to the metadata JSON file",
    )
    prompt_used: str = Field(..., description="The final prompt after template rendering")
    model_used: str = Field(..., description="The model that was used for generation")
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if generation failed",
    )
    generation_time_ms: Optional[float] = Field(
        default=None,
        description="Time taken to generate the image in milliseconds",
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp when the result was created",
    )

    model_config = {"arbitrary_types_allowed": True}


class JobConfig(BaseModel):
    """Configuration for a batch image generation job.

    Attributes:
        input_file: Path to the input CSV or JSON file.
        output_dir: Directory to save generated images.
        prompt_template: Jinja2 template string for prompts.
        model: Default model to use for generation.
        max_workers: Maximum number of concurrent generation tasks.
        api_key: Optional API key (can be set via environment variable).
    """

    input_file: Path = Field(..., description="Path to the input CSV or JSON file")
    output_dir: Path = Field(
        default=Path("outputs"),
        description="Directory to save generated images",
    )
    prompt_template: Optional[str] = Field(
        default=None,
        description="Jinja2 template string for prompts",
    )
    model: ModelName = Field(
        default=ModelName.GEMINI_FLASH,
        description="Default model to use for generation",
    )
    max_workers: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of concurrent generation tasks",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key (can be set via GEMINI_API_KEY environment variable)",
    )

    @field_validator("input_file")
    @classmethod
    def validate_input_file(cls, v: Path) -> Path:
        """Validate that the input file exists and has a supported extension."""
        if not v.exists():
            raise ValueError(f"Input file does not exist: {v}")
        if v.suffix.lower() not in [".csv", ".json"]:
            raise ValueError(f"Unsupported file format: {v.suffix}. Use .csv or .json")
        return v

    @field_validator("max_workers")
    @classmethod
    def validate_max_workers(cls, v: int) -> int:
        """Validate max_workers is within acceptable range."""
        if v < 1:
            raise ValueError("max_workers must be at least 1")
        if v > 20:
            raise ValueError("max_workers cannot exceed 20 to avoid rate limiting")
        return v

    model_config = {"arbitrary_types_allowed": True}


class BatchProgress(BaseModel):
    """Progress information for a batch job.

    Attributes:
        total: Total number of images to generate.
        completed: Number of successfully completed images.
        failed: Number of failed generations.
        in_progress: Number of currently processing images.
        pending: Number of images waiting to be processed.
    """

    total: int = Field(default=0, description="Total number of images to generate")
    completed: int = Field(default=0, description="Number of successfully completed images")
    failed: int = Field(default=0, description="Number of failed generations")
    in_progress: int = Field(default=0, description="Number of currently processing images")
    pending: int = Field(default=0, description="Number of images waiting to be processed")

    @property
    def progress_percent(self) -> float:
        """Calculate the progress percentage."""
        if self.total == 0:
            return 0.0
        return ((self.completed + self.failed) / self.total) * 100

    @property
    def success_rate(self) -> float:
        """Calculate the success rate as a percentage."""
        done = self.completed + self.failed
        if done == 0:
            return 0.0
        return (self.completed / done) * 100


class ImageMetadata(BaseModel):
    """Metadata saved alongside each generated image.

    Attributes:
        request_id: The ID of the original request.
        prompt: The exact prompt used for generation.
        model: The model used for generation.
        template_vars: Template variables that were substituted.
        generation_time_ms: Time taken to generate the image.
        created_at: Timestamp of generation.
        output_filename: Name of the output image file.
        base_image: Path to the base image (for edits).
        edit_type: Type of edit operation performed.
    """

    request_id: str = Field(..., description="The ID of the original request")
    prompt: str = Field(..., description="The exact prompt used for generation")
    model: str = Field(..., description="The model used for generation")
    template_vars: dict[str, Any] = Field(
        default_factory=dict,
        description="Template variables that were substituted",
    )
    generation_time_ms: Optional[float] = Field(
        default=None,
        description="Time taken to generate the image",
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp of generation",
    )
    output_filename: str = Field(..., description="Name of the output image file")
    base_image: Optional[str] = Field(
        default=None,
        description="Path to the base image (for edits)",
    )
    edit_type: Optional[str] = Field(
        default=None,
        description="Type of edit operation performed",
    )


class EditType(str, Enum):
    """Types of image editing operations."""

    INPAINT = "inpaint"
    TRANSFORM = "transform"
    STYLE_TRANSFER = "style_transfer"
    ANOMALY = "anomaly"
    VARIATION = "variation"


class ImageEditRequest(BaseModel):
    """Request model for a single image edit task.

    Attributes:
        id: Unique identifier for this request.
        base_image: Path to the base image to edit.
        edit_prompt: The text prompt describing the edit.
        mask_image: Optional path to a mask image for inpainting.
        edit_type: Type of edit operation.
        strength: Edit strength (0.0 to 1.0).
        template_vars: Variables to substitute into the prompt template.
        model: The Gemini model to use for editing.
        output_filename: Optional custom filename for the output image.
    """

    id: str = Field(..., description="Unique identifier for this request")
    base_image: Path = Field(..., description="Path to the base image to edit")
    edit_prompt: str = Field(..., description="The text prompt describing the edit")
    mask_image: Optional[Path] = Field(
        default=None,
        description="Optional path to a mask image for inpainting",
    )
    edit_type: EditType = Field(
        default=EditType.TRANSFORM,
        description="Type of edit operation",
    )
    strength: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Edit strength (0.0 to 1.0)",
    )
    template_vars: dict[str, Any] = Field(
        default_factory=dict,
        description="Variables to substitute into the prompt template",
    )
    model: ModelName = Field(
        default=ModelName.GEMINI_FLASH,
        description="The Gemini model to use for editing",
    )
    output_filename: Optional[str] = Field(
        default=None,
        description="Optional custom filename for the output image",
    )

    @field_validator("base_image")
    @classmethod
    def validate_base_image(cls, v: Path) -> Path:
        """Validate that the base image exists."""
        if not v.exists():
            raise ValueError(f"Base image does not exist: {v}")
        return v

    @field_validator("mask_image")
    @classmethod
    def validate_mask_image(cls, v: Optional[Path]) -> Optional[Path]:
        """Validate that the mask image exists if provided."""
        if v is not None and not v.exists():
            raise ValueError(f"Mask image does not exist: {v}")
        return v

    model_config = {"arbitrary_types_allowed": True}


class EditJobConfig(BaseModel):
    """Configuration for a batch image editing job.

    Attributes:
        input_file: Path to the input CSV or JSON file.
        base_image_dir: Directory containing base images.
        default_image: Default image to use when base_image is not in CSV.
        output_dir: Directory to save edited images.
        edit_template: Jinja2 template string for edit prompts.
        default_edit_type: Default type of edit operation.
        default_strength: Default edit strength.
        model: Default model to use for editing.
        max_workers: Maximum number of concurrent edit tasks.
        api_key: Optional API key.
    """

    input_file: Path = Field(..., description="Path to the input CSV or JSON file")
    base_image_dir: Optional[Path] = Field(
        default=None,
        description="Directory containing base images (if not absolute paths in CSV)",
    )
    default_image: Optional[str] = Field(
        default=None,
        description="Default image filename to use when base_image column is missing from CSV",
    )
    output_dir: Path = Field(
        default=Path("outputs"),
        description="Directory to save edited images",
    )
    edit_template: Optional[str] = Field(
        default=None,
        description="Jinja2 template string for edit prompts",
    )
    default_edit_type: EditType = Field(
        default=EditType.TRANSFORM,
        description="Default type of edit operation",
    )
    default_strength: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Default edit strength",
    )
    model: ModelName = Field(
        default=ModelName.GEMINI_FLASH,
        description="Default model to use for editing",
    )
    max_workers: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of concurrent edit tasks",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key (can be set via GEMINI_API_KEY environment variable)",
    )

    @field_validator("input_file")
    @classmethod
    def validate_input_file(cls, v: Path) -> Path:
        """Validate that the input file exists and has a supported extension."""
        if not v.exists():
            raise ValueError(f"Input file does not exist: {v}")
        if v.suffix.lower() not in [".csv", ".json"]:
            raise ValueError(f"Unsupported file format: {v.suffix}. Use .csv or .json")
        return v

    model_config = {"arbitrary_types_allowed": True}

