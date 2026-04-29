from __future__ import annotations

import re

from .models import ChunkRecord, DocumentRecord
from .utils import estimate_tokens, normalize_whitespace


def _split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n{2,}", text)
    return [normalize_whitespace(part) for part in parts if normalize_whitespace(part)]


def _guess_section_title(text: str) -> str:
    first_line = text.splitlines()[0].strip() if text.strip() else ""
    return first_line if 0 < len(first_line) <= 40 else ""


def _pack_units(units: list[tuple[int, str]], max_chars: int, min_chars: int) -> list[tuple[int, int, str]]:
    packed: list[tuple[int, int, str]] = []
    if not units:
        return packed
    start_page = units[0][0]
    end_page = start_page
    buffer: list[str] = []
    for page_no, unit in units:
        candidate = "\n\n".join(buffer + [unit]).strip()
        if buffer and len(candidate) > max_chars and len("\n\n".join(buffer)) >= min_chars:
            packed.append((start_page, end_page, "\n\n".join(buffer).strip()))
            buffer = [unit]
            start_page = page_no
            end_page = page_no
            continue
        if not buffer:
            start_page = page_no
        buffer.append(unit)
        end_page = page_no
    if buffer:
        packed.append((start_page, end_page, "\n\n".join(buffer).strip()))
    return packed


def build_dual_chunks(
    documents: list[DocumentRecord],
    structure_chars: int = 1400,
    window_chars: int = 2400,
    min_chars: int = 450,
) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    by_doc: dict[str, list[ChunkRecord]] = {}
    sequence = 1

    for document in documents:
        units: list[tuple[int, str]] = []
        for page_no, page_text in enumerate(document.page_texts or [document.text], start=1):
            for paragraph in _split_paragraphs(page_text):
                units.append((page_no, paragraph))
        current: list[ChunkRecord] = []
        for position, (page_start, page_end, text) in enumerate(
            _pack_units(units, max_chars=structure_chars, min_chars=min_chars),
            start=1,
        ):
            chunk = ChunkRecord(
                chunk_id=f"CH-{sequence:06d}",
                doc_id=document.metadata.doc_id,
                chunk_type="structure",
                text=text,
                token_estimate=estimate_tokens(text),
                page_start=page_start,
                page_end=page_end,
                section_title=_guess_section_title(text),
                position=position,
                file_name=document.metadata.file_name,
            )
            current.append(chunk)
            chunks.append(chunk)
            sequence += 1
        by_doc[document.metadata.doc_id] = current

    for document in documents:
        structure_chunks = by_doc.get(document.metadata.doc_id, [])
        for index in range(len(structure_chunks)):
            merged: list[str] = []
            page_start = structure_chunks[index].page_start
            page_end = structure_chunks[index].page_end
            for offset in range(index, min(index + 3, len(structure_chunks))):
                next_chunk = structure_chunks[offset]
                candidate = "\n\n".join(merged + [next_chunk.text]).strip()
                if merged and len(candidate) > window_chars:
                    break
                merged.append(next_chunk.text)
                page_end = next_chunk.page_end
            if not merged:
                continue
            text = "\n\n".join(merged).strip()
            chunk = ChunkRecord(
                chunk_id=f"CH-{sequence:06d}",
                doc_id=document.metadata.doc_id,
                chunk_type="window",
                text=text,
                token_estimate=estimate_tokens(text),
                page_start=page_start,
                page_end=page_end,
                section_title=structure_chunks[index].section_title,
                position=index + 1,
                file_name=document.metadata.file_name,
            )
            chunks.append(chunk)
            sequence += 1

    return chunks
