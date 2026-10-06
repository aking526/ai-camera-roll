"""Image generation and editing through a replaceable provider interface."""

import base64
import binascii
import io
import os
from abc import ABC, abstractmethod
from collections.abc import Sequence

import httpx
from dotenv import load_dotenv
from PIL import Image, ImageOps, UnidentifiedImageError


class ImageResponseError(ValueError):
    """The image provider returned a malformed or unsuccessful response."""


def png_bytes(content: bytes, *, reference: bool = False) -> bytes:
    """Validate an image, normalize orientation, and optionally fit a white 448px canvas."""
    try:
        with Image.open(io.BytesIO(content)) as source:
            source.load()
            image = ImageOps.exif_transpose(source).convert("RGBA")
            background = Image.new("RGBA", image.size, "white")
            background.alpha_composite(image)
            image = background.convert("RGB")
            if reference:
                image.thumbnail((448, 448), Image.Resampling.LANCZOS)
                canvas = Image.new("RGB", (448, 448), "white")
                canvas.paste(
                    image, ((448 - image.width) // 2, (448 - image.height) // 2)
                )
                image = canvas
            destination = io.BytesIO()
            image.save(destination, format="PNG")
            return destination.getvalue()
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise ImageResponseError(
            "Provider did not return a decodable image."
        ) from error


class ImageModel(ABC):
    model: str

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        references: Sequence[bytes] = (),
        width: int = 1024,
        height: int = 1024,
        seed: int | None = None,
    ) -> bytes:
        """Return image bytes, optionally editing or conditioning on reference images."""


class CloudflareImageModel(ImageModel):
    def __init__(
        self,
        *,
        api_token: str | None = None,
        account_id: str | None = None,
        model: str | None = None,
        timeout: float = 300,
    ) -> None:
        load_dotenv(".env", override=False)
        self.api_token = api_token or os.environ.get("CLOUDFLARE_API_TOKEN")
        self.account_id = account_id or os.environ.get("CLOUDFLARE_ACCOUNT_ID")
        self.model = model or os.environ.get(
            "CLOUDFLARE_IMAGE_MODEL", "@cf/black-forest-labs/flux-2-klein-4b"
        )
        self.timeout = timeout
        if not self.api_token or not self.account_id:
            raise ValueError(
                "Set CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID, or pass them explicitly."
            )
        if self.model != "@cf/black-forest-labs/flux-2-klein-4b":
            raise ValueError(
                "This adapter currently supports @cf/black-forest-labs/flux-2-klein-4b only."
            )

    def generate(
        self,
        prompt: str,
        *,
        references: Sequence[bytes] = (),
        width: int = 1024,
        height: int = 1024,
        seed: int | None = None,
    ) -> bytes:
        if not prompt.strip():
            raise ValueError("Image prompt must not be empty.")
        if len(references) > 4:
            raise ValueError("Cloudflare Klein accepts at most four reference images.")
        if any(
            type(value) is not int or not 256 <= value <= 1920
            for value in (width, height)
        ):
            raise ValueError("Image dimensions must be integers between 256 and 1920.")
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**32):
            raise ValueError("seed must be an integer between 0 and 2**32 - 1.")
        # Sending prompt fields via files also creates multipart requests without image inputs.
        parts = [
            ("prompt", (None, prompt)),
            ("width", (None, str(width))),
            ("height", (None, str(height))),
        ]
        if seed is not None:
            parts.append(("seed", (None, str(seed))))
        for index, content in enumerate(references):
            parts.append(
                (
                    f"input_image_{index}",
                    (
                        f"reference_{index}.png",
                        png_bytes(content, reference=True),
                        "image/png",
                    ),
                )
            )
        response = httpx.post(
            f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/run/{self.model}",
            headers={"Authorization": f"Bearer {self.api_token}"},
            files=parts,
            timeout=self.timeout,
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            # Preserve the HTTP exception and response, but expose actionable provider
            # codes (for example, temporary capacity versus exhausted daily quota).
            details = []
            try:
                payload = response.json()
                for item in payload.get("errors", []):
                    if isinstance(item, dict):
                        details.append(
                            f"{item.get('code', 'unknown')}: {item.get('message', '')}"
                        )
            except (ValueError, AttributeError, TypeError):
                pass
            if response.headers.get("Retry-After"):
                details.append(f"Retry-After: {response.headers['Retry-After']}")
            if details:
                raise httpx.HTTPStatusError(
                    f"{error}\nCloudflare: {'; '.join(details)}",
                    request=error.request,
                    response=error.response,
                ) from error
            raise
        try:
            payload = response.json()
            if not payload.get("success", False):
                raise ImageResponseError(
                    f"Cloudflare image generation failed: {payload.get('errors', [])}"
                )
            encoded = payload["result"]["image"]
            content = base64.b64decode(encoded, validate=True)
        except (
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            binascii.Error,
        ) as error:
            if isinstance(error, ImageResponseError):
                raise
            raise ImageResponseError("Malformed Cloudflare image response.") from error
        return png_bytes(content)
