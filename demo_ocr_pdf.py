from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Safety knobs for stability / reproducibility
os.environ.setdefault("DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "HF")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

from paddleocr import PaddleOCR
from pdf_to_images import pdf_to_images


def ocr_page_to_records(page_img: np.ndarray, ocr: PaddleOCR) -> List[Dict[str, Any]]:
    """
    Returns list of {box, text, score} for one page.
    box: 4 points [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    """
    results = ocr.predict(input=page_img)
    # PaddleOCR 3.x returns a list of PageResult-like objects
    # We’ll normalize into plain dicts.
    records: List[Dict[str, Any]] = []
    for page_res in results:
        # page_res might contain fields; easiest: page_res.json or page_res.as_dict if available
        # But to be robust, use page_res.get(...) style fallback:
        if hasattr(page_res, "to_dict"):
            d = page_res.to_dict()
            # Try common structures
            # Expect something like d["rec_texts"], d["rec_scores"], d["dt_polys"] or similar
            # If your version differs, print(d.keys()) once to adapt.
            dt_polys = d.get("dt_polys") or d.get("dt_boxes") or []
            rec_texts = d.get("rec_texts") or []
            rec_scores = d.get("rec_scores") or []

            for box, text, score in zip(dt_polys, rec_texts, rec_scores):
                records.append({"box": box, "text": text, "score": float(score)})
        else:
            # Worst-case fallback: just store printable output
            records.append({"raw": str(page_res)})
    return records


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, help="Path to PDF")
    parser.add_argument("--out", default="output_ocr", help="Output directory")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--lang", default="en")
    parser.add_argument("--start", type=int, default=0, help="Start page (0-indexed)")
    parser.add_argument("--end", type=int, default=-1, help="End page (exclusive); -1 means all")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create OCR pipeline
    ocr = PaddleOCR(
        lang=args.lang,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    page_range = None if args.end == -1 else (args.start, args.end)
    images = pdf_to_images(args.pdf, dpi=args.dpi, page_range=page_range)

    all_pages: List[Dict[str, Any]] = []
    for idx, img in enumerate(images):
        page_idx = args.start + idx
        records = ocr_page_to_records(img, ocr)
        all_pages.append({"page_index": page_idx, "items": records})
        print(f"Page {page_idx}: {len(records)} lines")

    (out_dir / "ocr_results.json").write_text(json.dumps(all_pages, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved:", out_dir / "ocr_results.json")


if __name__ == "__main__":
    main()