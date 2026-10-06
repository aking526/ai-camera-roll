import base64
import io
from email.parser import BytesParser
from email.policy import default

import httpx
import pytest
from PIL import Image

from ai_camera_roll.images import CloudflareImageModel, ImageResponseError, png_bytes


def image_bytes(size=(1024, 768), mode="RGB", color="blue"):
    stream = io.BytesIO()
    Image.new(mode, size, color).save(stream, format="PNG")
    return stream.getvalue()


def install(monkeypatch, response):
    requests = []

    def handle(request):
        requests.append(request)
        return response

    client = httpx.Client(transport=httpx.MockTransport(handle))
    monkeypatch.setattr("ai_camera_roll.images.httpx.post", client.post)
    return requests, client


@pytest.mark.parametrize("reference_count", [0, 4])
def test_cloudflare_multipart_generation_and_editing(monkeypatch, reference_count):
    image = image_bytes()
    requests, client = install(
        monkeypatch,
        httpx.Response(
            200,
            json={
                "success": True,
                "result": {"image": base64.b64encode(image).decode()},
            },
        ),
    )
    try:
        adapter = CloudflareImageModel(
            api_token="test-token", account_id="test-account"
        )
        output = adapter.generate(
            "A photograph.", references=[image] * reference_count, seed=42
        )
    finally:
        client.close()
    assert Image.open(io.BytesIO(output)).size == (1024, 768)
    request = requests[0]
    assert str(request.url).endswith(
        "/accounts/test-account/ai/run/@cf/black-forest-labs/flux-2-klein-4b"
    )
    assert request.headers["authorization"] == "Bearer test-token"
    assert request.extensions["timeout"]["read"] == 300
    parsed = BytesParser(policy=default).parsebytes(
        f"Content-Type: {request.headers['content-type']}\r\n\r\n".encode()
        + request.content
    )
    parts = {
        part.get_param("name", header="content-disposition"): part.get_payload(
            decode=True
        )
        for part in parsed.iter_parts()
    }
    assert parts["prompt"] == b"A photograph."
    assert parts["width"] == parts["height"] == b"1024"
    assert parts["seed"] == b"42"
    assert len(parts) == 4 + reference_count
    for index in range(reference_count):
        assert Image.open(io.BytesIO(parts[f"input_image_{index}"])).size == (448, 448)


def test_reference_padding_preserves_proportions_and_transparency():
    normalized = png_bytes(
        image_bytes((200, 100), "RGBA", (255, 0, 0, 128)), reference=True
    )
    image = Image.open(io.BytesIO(normalized))
    assert image.size == (448, 448)
    assert image.getpixel((0, 0)) == (255, 255, 255)
    assert image.getpixel((224, 224)) == (255, 127, 127)
    assert image.getpixel((224, 160)) == (255, 255, 255)


@pytest.mark.parametrize(
    "payload",
    [
        {"success": False, "errors": [{"message": "Denied"}]},
        {"success": True, "result": {}},
        {"success": True, "result": {"image": "not base64"}},
        {
            "success": True,
            "result": {"image": base64.b64encode(b"not an image").decode()},
        },
        [],
    ],
)
def test_malformed_response_does_not_retry(monkeypatch, payload):
    requests, client = install(monkeypatch, httpx.Response(200, json=payload))
    try:
        with pytest.raises(ImageResponseError):
            CloudflareImageModel(api_token="test", account_id="test").generate(
                "A photo"
            )
    finally:
        client.close()
    assert len(requests) == 1


def test_http_errors_propagate(monkeypatch):
    requests, client = install(monkeypatch, httpx.Response(403, json={"errors": []}))
    try:
        with pytest.raises(httpx.HTTPStatusError):
            CloudflareImageModel(api_token="test", account_id="test").generate(
                "A photo"
            )
    finally:
        client.close()
    assert len(requests) == 1


def test_capacity_error_includes_provider_code_and_retry_delay(monkeypatch):
    requests, client = install(
        monkeypatch,
        httpx.Response(
            429,
            headers={"Retry-After": "30"},
            json={
                "errors": [{"code": 3040, "message": "Capacity temporarily exceeded"}]
            },
        ),
    )
    try:
        with pytest.raises(
            httpx.HTTPStatusError, match="3040: Capacity temporarily exceeded"
        ) as caught:
            CloudflareImageModel(api_token="test", account_id="test").generate(
                "A photo"
            )
        assert caught.value.response.status_code == 429
        assert "Retry-After: 30" in str(caught.value)
    finally:
        client.close()
    assert len(requests) == 1


@pytest.mark.parametrize(
    "kwargs", [{"references": [b"x"] * 5}, {"width": 0}, {"seed": -1}]
)
def test_invalid_request_never_calls_provider(monkeypatch, kwargs):
    requests, client = install(monkeypatch, httpx.Response(500))
    try:
        with pytest.raises(ValueError):
            CloudflareImageModel(api_token="test", account_id="test").generate(
                "A photo", **kwargs
            )
    finally:
        client.close()
    assert requests == []
