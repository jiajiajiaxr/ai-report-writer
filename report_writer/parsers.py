from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

import fitz

from .models import DocumentMetadata, DocumentRecord
from .utils import normalize_whitespace, sha1_file, utc_now


def _infer_tags(path: Path) -> list[str]:
    tags = []
    if path.parent.name:
        tags.append(path.parent.name)
    tags.append(path.suffix.lower().lstrip("."))
    return tags


def parse_pdf(path: Path) -> DocumentRecord:
    checksum = sha1_file(path)
    pdf = fitz.open(path)
    page_texts = [normalize_whitespace(page.get_text("text")) for page in pdf]
    text = normalize_whitespace("\n\n".join(page_texts))
    metadata = DocumentMetadata(
        doc_id=checksum[:12],
        file_name=path.name,
        source_path=str(path.resolve()),
        file_type="pdf",
        title=path.stem,
        checksum=checksum,
        char_count=len(text),
        page_count=len(page_texts),
        tags=_infer_tags(path),
        created_at=utc_now(),
    )
    return DocumentRecord(metadata=metadata, text=text, page_texts=page_texts)


def parse_docx(path: Path) -> DocumentRecord:
    checksum = sha1_file(path)
    paragraphs: list[str] = []
    with ZipFile(path) as archive:
        document_xml = archive.read("word/document.xml")
    root = ElementTree.fromstring(document_xml)
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    for paragraph in root.findall(".//w:body/w:p", namespace):
        texts = [node.text or "" for node in paragraph.findall(".//w:t", namespace)]
        joined = "".join(texts).strip()
        if joined:
            paragraphs.append(joined)
    text = normalize_whitespace("\n\n".join(paragraphs))
    metadata = DocumentMetadata(
        doc_id=checksum[:12],
        file_name=path.name,
        source_path=str(path.resolve()),
        file_type="docx",
        title=path.stem,
        checksum=checksum,
        char_count=len(text),
        page_count=1,
        tags=_infer_tags(path),
        created_at=utc_now(),
    )
    return DocumentRecord(metadata=metadata, text=text, page_texts=[text] if text else [])


def parse_text(path: Path) -> DocumentRecord:
    checksum = sha1_file(path)
    text = normalize_whitespace(path.read_text(encoding="utf-8"))
    metadata = DocumentMetadata(
        doc_id=checksum[:12],
        file_name=path.name,
        source_path=str(path.resolve()),
        file_type=path.suffix.lower().lstrip(".") or "text",
        title=path.stem,
        checksum=checksum,
        char_count=len(text),
        page_count=1,
        tags=_infer_tags(path),
        created_at=utc_now(),
    )
    return DocumentRecord(metadata=metadata, text=text, page_texts=[text] if text else [])


def parse_document(path: str | Path) -> DocumentRecord:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(file_path)
    if suffix == ".docx":
        return parse_docx(file_path)
    if suffix in {".md", ".txt"}:
        return parse_text(file_path)
    raise ValueError(f"Unsupported document type: {file_path.suffix}")


def parse_folder(folder: str | Path) -> list[DocumentRecord]:
    source_dir = Path(folder)
    documents: list[DocumentRecord] = []
    for path in sorted(source_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".pdf", ".docx", ".md", ".txt"}:
            documents.append(parse_document(path))
    if not documents:
        raise ValueError(f"No supported documents found in {source_dir}")
    return documents
