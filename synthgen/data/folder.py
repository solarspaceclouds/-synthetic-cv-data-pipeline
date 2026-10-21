"""Unannotated image folder as a source (face jobs: labels come from the face parser)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class ImageFolderSource:
    def __init__(self, images_dir: str | Path):
        self.root = Path(images_dir)
        files = sorted(p for p in self.root.iterdir() if p.suffix.lower() in _EXT)
        if not files:
            raise FileNotFoundError(f"no images in {self.root}")
        self.images = {}
        for i, p in enumerate(files, 1):
            with Image.open(p) as im:
                self.images[i] = {"id": i, "file_name": p.name, "width": im.width, "height": im.height}

    def load_image(self, image_id: int) -> Image.Image:
        return Image.open(self.root / self.images[image_id]["file_name"]).convert("RGB")
