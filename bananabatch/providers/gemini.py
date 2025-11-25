"""Google Gemini image generation provider.

This module implements the ImageProvider interface using Google's
new google-genai SDK for image generation and editing with Gemini models.
"""

import asyncio
import base64
from pathlib import Path
from typing import Any, Optional

from google import genai
from google.genai import types

from bananabatch.providers.base import (
    AuthenticationError,
    ImageGenerationError,
    ImageProvider,
    ProviderError,
    RateLimitError,
)


class GeminiProvider(ImageProvider):
    """Image generation provider using Google Gemini API.

    This provider uses the new google-genai SDK to generate images
    using Gemini 2.5 Flash and Gemini 3 Pro models.

    Attributes:
        api_key: The Gemini API key.
        default_model: The default model to use (defaults to gemini-2.5-flash-image).
        client: The initialized Gemini client.
    """

    SUPPORTED_MODELS = [
        "gemini-2.5-flash-image",
        "gemini-3-pro-image-preview",
    ]

    def __init__(
        self,
        api_key: str,
        default_model: str = "gemini-2.5-flash-image",
    ) -> None:
        """Initialize the Gemini provider.

        Args:
            api_key: The Gemini API key for authentication.
            default_model: The default model to use for generation.
        """
        super().__init__(api_key, default_model)
        # Initialize client eagerly to avoid race conditions in concurrent requests
        self._client: genai.Client = genai.Client(api_key=self.api_key)

    @property
    def client(self) -> genai.Client:
        """Get the Gemini client.

        Returns:
            The initialized Gemini client.
        """
        return self._client

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> bytes:
        """Generate an image from a text prompt using Gemini.

        Args:
            prompt: The text prompt describing the desired image.
            model: Optional model override. Uses default_model if not specified.
            **kwargs: Additional generation parameters.

        Returns:
            The generated image as raw bytes.

        Raises:
            ImageGenerationError: If image generation fails.
            RateLimitError: If rate limits are exceeded.
            AuthenticationError: If API key is invalid.
        """
        model_to_use = model or self.default_model

        if model_to_use not in self.SUPPORTED_MODELS:
            raise ProviderError(
                f"Unsupported model: {model_to_use}. Supported models: {self.SUPPORTED_MODELS}",
                self.provider_name,
            )

        try:
            # Run the synchronous API call in a thread pool to not block async
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self._generate_sync,
                prompt,
                model_to_use,
            )
            return response

        except Exception as e:
            error_str = str(e).lower()

            # Check for authentication errors
            if "api key" in error_str or "authentication" in error_str or "401" in error_str:
                raise AuthenticationError(self.provider_name, e)

            # Check for rate limit errors
            if "rate limit" in error_str or "quota" in error_str or "429" in error_str:
                raise RateLimitError(self.provider_name, original_error=e)

            # Generic image generation error
            raise ImageGenerationError(self.provider_name, prompt, e)

    def _generate_sync(self, prompt: str, model: str) -> bytes:
        """Synchronous image generation (called from thread pool).

        Args:
            prompt: The text prompt for generation.
            model: The model to use.

        Returns:
            The generated image as raw bytes.

        Raises:
            Exception: If generation fails or no image is returned.
        """
        response = self.client.models.generate_content(
            model=model,
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
            ),
        )

        # Extract image data from response
        if response.candidates:
            for candidate in response.candidates:
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, "inline_data") and part.inline_data:
                            image_data = part.inline_data.data
                            # Handle both bytes and base64 string
                            if isinstance(image_data, str):
                                return base64.b64decode(image_data)
                            return image_data

        # Fallback: Check response.parts directly (SDK version variations)
        if hasattr(response, "parts"):
            for part in response.parts:
                if hasattr(part, "inline_data") and part.inline_data:
                    image_data = part.inline_data.data
                    if isinstance(image_data, str):
                        return base64.b64decode(image_data)
                    return image_data

        raise Exception("No image found in Gemini response")

    async def validate_connection(self) -> bool:
        """Validate that the Gemini API connection is working.

        Returns:
            True if the connection is valid and authenticated.

        Raises:
            AuthenticationError: If the API key is invalid.
        """
        try:
            # Try a simple API call to validate the key
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._validate_sync,
            )
            return True
        except Exception as e:
            raise AuthenticationError(self.provider_name, e)

    def _validate_sync(self) -> None:
        """Synchronous validation (called from thread pool)."""
        # List models to validate the API key works
        _ = self.client.models.list()

    @property
    def supported_models(self) -> list[str]:
        """Get the list of supported Gemini model names.

        Returns:
            List of supported model identifiers.
        """
        return self.SUPPORTED_MODELS.copy()

    @property
    def provider_name(self) -> str:
        """Get the provider name.

        Returns:
            The string 'Gemini'.
        """
        return "Gemini"

    async def edit(
        self,
        base_image: Path,
        prompt: str,
        mask_image: Optional[Path] = None,
        model: Optional[str] = None,
        strength: float = 0.75,
        **kwargs: Any,
    ) -> bytes:
        """Edit an existing image based on a text prompt using Gemini.

        Args:
            base_image: Path to the base image to edit.
            prompt: The text prompt describing the desired edit.
            mask_image: Optional path to a mask image for inpainting.
            model: Optional model override. Uses default_model if not specified.
            strength: Edit strength from 0.0 (subtle) to 1.0 (dramatic).
            **kwargs: Additional provider-specific parameters.

        Returns:
            The edited image as raw bytes.

        Raises:
            ImageGenerationError: If image editing fails.
            RateLimitError: If rate limits are exceeded.
            AuthenticationError: If API key is invalid.
        """
        model_to_use = model or self.default_model

        if model_to_use not in self.SUPPORTED_MODELS:
            raise ProviderError(
                f"Unsupported model: {model_to_use}. Supported models: {self.SUPPORTED_MODELS}",
                self.provider_name,
            )

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self._edit_sync,
                base_image,
                prompt,
                mask_image,
                model_to_use,
                strength,
            )
            return response

        except Exception as e:
            error_str = str(e).lower()

            if "api key" in error_str or "authentication" in error_str or "401" in error_str:
                raise AuthenticationError(self.provider_name, e)

            if "rate limit" in error_str or "quota" in error_str or "429" in error_str:
                raise RateLimitError(self.provider_name, original_error=e)

            raise ImageGenerationError(self.provider_name, prompt, e)

    def _edit_sync(
        self,
        base_image: Path,
        prompt: str,
        mask_image: Optional[Path],
        model: str,
        strength: float,
    ) -> bytes:
        """Synchronous image editing (called from thread pool).

        Args:
            base_image: Path to the base image.
            prompt: The edit prompt.
            mask_image: Optional mask image path.
            model: The model to use.
            strength: Edit strength.

        Returns:
            The edited image as raw bytes.
        """
        # Read and encode the base image
        with open(base_image, "rb") as f:
            base_image_bytes = f.read()

        # Determine mime type from extension
        suffix = base_image.suffix.lower()
        mime_type = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }.get(suffix, "image/png")

        # Build the content parts
        contents = [
            types.Part.from_bytes(data=base_image_bytes, mime_type=mime_type),
        ]

        # Add mask if provided
        if mask_image is not None:
            with open(mask_image, "rb") as f:
                mask_bytes = f.read()
            mask_suffix = mask_image.suffix.lower()
            mask_mime = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
            }.get(mask_suffix, "image/png")
            contents.append(types.Part.from_bytes(data=mask_bytes, mime_type=mask_mime))

        # Add the text prompt
        contents.append(prompt)

        # Make the API call
        response = self.client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
            ),
        )

        # Extract image data from response
        if response.candidates:
            for candidate in response.candidates:
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, "inline_data") and part.inline_data:
                            image_data = part.inline_data.data
                            if isinstance(image_data, str):
                                return base64.b64decode(image_data)
                            return image_data

        if hasattr(response, "parts"):
            for part in response.parts:
                if hasattr(part, "inline_data") and part.inline_data:
                    image_data = part.inline_data.data
                    if isinstance(image_data, str):
                        return base64.b64decode(image_data)
                    return image_data

        raise Exception("No image found in Gemini edit response")
