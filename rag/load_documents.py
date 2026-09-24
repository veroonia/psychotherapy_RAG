"""
load_documents.py
Extracts raw text from all PDFs in rag/documents/ and saves it as JSON
for the next pipeline step (chunk_documents.py).

Usage:
    python load_documents.py
"""

import json
from pathlib import Path
from pypdf import PdfReader

DOCUMENTS_DIR = Path(__file__).parent / "documents"
OUTPUT_DIR = Path(__file__).parent / "data"
OUTPUT_FILE = OUTPUT_DIR / "documents.json"


def load_pdf(pdf_path: Path) -> list[dict]:
    """Extract text page-by-page from a single PDF."""
    reader = PdfReader(str(pdf_path))
    pages = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue  # skip blank / image-only pages
        pages.append({
            "source": pdf_path.name,
            "page": page_num,
            "text": text,
        })
    return pages


def load_all_documents() -> list[dict]:
    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {DOCUMENTS_DIR}")

    all_pages = []
    for pdf_path in pdf_files:
        print(f"Loading {pdf_path.name} ...")
        pages = load_pdf(pdf_path)
        print(f"  -> extracted {len(pages)} pages of text")
        all_pages.extend(pages)

    return all_pages


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    documents = load_all_documents()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(documents)} pages from across all PDFs to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()