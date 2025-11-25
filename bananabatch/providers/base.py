"""Abstract base class for image generation providers.

This module defines the interface that all image providers must implement,
enabling easy swapping of providers (e.g., Gemini, DALL-E, Stable Diffusion).
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional


class ImageProvider(ABC):
    """Abstract base class for image generation providers.

    This class defines the interface for image generation providers.
    Concrete implementations must handle API initialization, image generation,
    image editing, and proper error handling.

    Attributes:
        api_key: The API key for the provider.
        default_model: The default model to use for generation.
    """

    def __init__(self, api_key: str, default_model: Optional[str] = None) -> None:
        """Initialize the image provider.

        Args:
            api_key: The API key for authentication with the provider.
            default_model: Optional default model to use for generation.
        """
        self.api_key = api_key
        self.default_model = default_model

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> bytes:
        """Generate an image from a text prompt.

        Args:
            prompt: The text prompt describing the desired image.
            model: Optional model override. Uses default_model if not specified.
            **kwargs: Additional provider-specific parameters.

        Returns:
            The generated image as raw bytes (PNG/JPEG format).

        Raises:
            ProviderError: If image generation fails.
            RateLimitError: If rate limits are exceeded.
            AuthenticationError: If API key is invalid.
        """
        pass

    @abstractmethod
    async def edit(
        self,
        base_image: Path,
        prompt: str,
        mask_image: Optional[Path] = None,
        model: Optional[str] = None,
        strength: float = 0.75,
        **kwargs: Any,
    ) -> bytes:
        """Edit an existing image based on a text prompt.

        Args:
            base_image: Path to the base image to edit.
            prompt: The text prompt describing the desired edit.
            mask_image: Optional path to a mask image for inpainting.
            model: Optional model override. Uses default_model if not specified.
            strength: Edit strength from 0.0 (subtle) to 1.0 (dramatic).
            **kwargs: Additional provider-specific parameters.

        Returns:
            The edited image as raw bytes (PNG/JPEG format).

        Raises:
            ProviderError: If image editing fails.
            RateLimitError: If rate limits are exceeded.
            AuthenticationError: If API key is invalid.
        """
        pass

    @abstractmethod
    async def validate_connection(self) -> bool:
        """Validate that the provider connection is working.

        Returns:
            True if the connection is valid and authenticated.

        Raises:
            AuthenticationError: If the API key is invalid.
        """
        pass

    @property
    @abstractmethod
    def supported_models(self) -> list[str]:
        """Get the list of supported model names.

        Returns:
            List of model identifiers supported by this provider.
        """
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get the name of this provider.

        Returns:
            Human-readable name of the provider.
        """
        pass

    @property
    def supports_editing(self) -> bool:
        """Check if this provider supports image editing.

        Returns:
            True if the provider supports image editing.
        """
        return True


class ProviderError(Exception):
    """Base exception for provider-related errors."""

    def __init__(self, message: str, provider: str, original_error: Optional[Exception] = None):
        """Initialize the provider error.

        Args:
            message: Human-readable error message.
            provider: Name of the provider that raised the error.
            original_error: The original exception that caused this error.
        """
        super().__init__(message)
        self.provider = provider
        self.original_error = original_error


class RateLimitError(ProviderError):
    """Exception raised when rate limits are exceeded."""

    def __init__(
        self,
        provider: str,
        retry_after: Optional[int] = None,
        original_error: Optional[Exception] = None,
    ):
        """Initialize the rate limit error.

        Args:
            provider: Name of the provider that raised the error.
            retry_after: Seconds to wait before retrying (if available).
            original_error: The original exception that caused this error.
        """
        message = f"Rate limit exceeded for {provider}"
        if retry_after:
            message += f". Retry after {retry_after} seconds."
        super().__init__(message, provider, original_error)
        self.retry_after = retry_after


class AuthenticationError(ProviderError):
    """Exception raised when authentication fails."""

    def __init__(self, provider: str, original_error: Optional[Exception] = None):
        """Initialize the authentication error.

        Args:
            provider: Name of the provider that raised the error.
            original_error: The original exception that caused this error.
        """
        message = f"Authentication failed for {provider}. Check your API key."
        super().__init__(message, provider, original_error)


class ImageGenerationError(ProviderError):
    """Exception raised when image generation fails."""

    def __init__(
        self,
        provider: str,
        prompt: str,
        original_error: Optional[Exception] = None,
    ):
        """Initialize the image generation error.

        Args:
            provider: Name of the provider that raised the error.
            prompt: The prompt that failed to generate an image.
            original_error: The original exception that caused this error.
        """
        message = f"Image generation failed for {provider}"
        if original_error:
            message += f": {original_error}"
        super().__init__(message, provider, original_error)
        self.prompt = prompt
