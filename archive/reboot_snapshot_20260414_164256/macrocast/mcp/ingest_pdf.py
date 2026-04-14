"""Extract and cache plain text from Coulombe-corpus PDFs.

Uses PyMuPDF (fitz) for fast, lightweight extraction.  Equation-heavy
content is acceptable as-is; the methodology document already covers
the math in clean form.

Run directly:
    uv run python macrocast/mcp/ingest_pdf.py
"""

import json
import logging
from pathlib import Path

import fitz  # PyMuPDF

from macrocast.mcp.config import PAPER_METADATA, PAPERS_CACHE_DIR, PDF_SOURCE_DIR

logger = logging.getLogger(__name__)


def extract_text(pdf_path: Path) -> str:
    """Return the full plain-text content of a PDF file."""
    doc = fitz.open(str(pdf_path))
    pages = []
    for page in doc:
        text = page.get_text("text")
        pages.append(text)
    doc.close()
    return "\n".join(pages)


def ingest_all(force: bool = False) -> dict[str, Path]:
    """Extract text from all papers in PAPER_METADATA and cache to JSON.

    Parameters
    ----------
    force:
        If True, re-extract even if cached file already exists.

    Returns
    -------
    dict mapping paper key → cache file path for successfully processed papers.
    """
    PAPERS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, Path] = {}

    for key, meta in PAPER_METADATA.items():
        cache_file = PAPERS_CACHE_DIR / f"{key}.json"

        if cache_file.exists() and not force:
            logger.info("Cache hit: %s", key)
            results[key] = cache_file
            continue

        pdf_path = PDF_SOURCE_DIR / meta["filename"]
        if not pdf_path.exists():
            logger.warning("PDF not found, skipping: %s (%s)", key, pdf_path)
            continue

        logger.info("Extracting: %s", pdf_path.name)
        text = extract_text(pdf_path)

        record = {
            "key": key,
            "title": meta["title"],
            "authors": meta["authors"],
            "year": meta["year"],
            "journal": meta["journal"],
            "filename": meta["filename"],
            "text": text,
        }
        cache_file.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        logger.info("  → %d characters saved to %s", len(text), cache_file.name)
        results[key] = cache_file

    logger.info("Ingested %d / %d papers.", len(results), len(PAPER_METADATA))
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ingest_all(force=False)
