from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pypdfium2 as pdfium


def pdf_to_images(
    pdf_path: str | Path,
    dpi: int = 200,
    page_range: Optional[Tuple[int, int]] = None,  # (start_inclusive, end_exclusive), 0-indexed
) -> List[np.ndarray]:
    """
    Render a PDF into a list of RGB numpy arrays (H, W, 3), dtype=uint8.

    - dpi: 150-220 is a good range for OCR.
    - page_range: to avoid rendering huge PDFs in one go.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    pdf = pdfium.PdfDocument(str(pdf_path))
    n_pages = len(pdf)

    start = 0
    end = n_pages
    if page_range is not None:
        start, end = page_range
        start = max(0, start)
        end = min(n_pages, end)

    images: List[np.ndarray] = []
    scale = dpi / 72.0  # PDF points are 72 dpi baseline

    for i in range(start, end):
        page = pdf[i]
        # Render page to a PIL image, then to numpy
        pil_image = page.render(scale=scale).to_pil()
        img = np.array(pil_image.convert("RGB"))
        images.append(img)

    return images