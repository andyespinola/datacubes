from __future__ import annotations

from pathlib import Path

from ImagesMangGenerator.preview_web_viewer import create_app


PNG_1X1 = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR"
    b"\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde"
    b"\x00\x00\x00\x0cIDATx\x9cc\xf8\xff\xff?\x00\x05\xfe\x02\xfeA\xe2`\x82"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


def write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(PNG_1X1)


def test_images_endpoint_paginates_and_sorts(tmp_path: Path) -> None:
    write_png(tmp_path / "galaxy-c.png")
    write_png(tmp_path / "galaxy-a.png")
    write_png(tmp_path / "galaxy-b.png")

    app = create_app(tmp_path)
    client = app.test_client()

    response = client.get("/api/images?limit=2&offset=0&sort=name")
    assert response.status_code == 200
    payload = response.get_json()

    assert payload["total"] == 3
    assert payload["filtered_total"] == 3
    assert payload["limit"] == 2
    assert [item["name"] for item in payload["items"]] == ["galaxy-a.png", "galaxy-b.png"]


def test_images_endpoint_filters_and_serves_nested_png(tmp_path: Path) -> None:
    write_png(tmp_path / "mangia" / "TNG50-87-141934-0-127_v0.png")
    write_png(tmp_path / "manga" / "7443-12703_v1.PNG")
    (tmp_path / "notes.txt").write_text("no es una imagen")

    app = create_app(tmp_path)
    client = app.test_client()

    response = client.get("/api/images?q=7443")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["total"] == 2
    assert payload["filtered_total"] == 1
    assert payload["items"][0]["path"] == "manga/7443-12703_v1.PNG"

    image_response = client.get("/previews/manga/7443-12703_v1.PNG")
    assert image_response.status_code == 200
    assert image_response.mimetype == "image/png"
