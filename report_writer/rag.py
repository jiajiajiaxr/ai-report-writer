from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass

import numpy as np

from .models import ChunkRecord
from .siliconflow import SiliconFlowClient


@dataclass(slots=True)
class RetrievalResult:
    chunk: ChunkRecord
    score: float


TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_+-]{2,}")
_INLINE_PAPER_CITATION_RE = re.compile(r"\[(?:\d+\s*(?:[-,]\s*\d+)*)\]")
_REFERENCE_LINE_RE = re.compile(r"(?:^|\n)\s*(?:\[\d+\]|\d+\s*[.)])\s+[A-Z\u4e00-\u9fff]")
_REFERENCE_CUE_RE = re.compile(r"\b(?:doi|transactions on|ieee|vol\.|no\.|pp\.|et al\.)\b", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_BIBLIOGRAPHY_NAME_RE = re.compile(r"\b[A-Z][A-Za-z'`.-]+(?:,\s*[A-Z][A-Za-z'`.-]+)*\s*\((?:19|20)\d{2}\)")
_FRONT_MATTER_CUE_RE = re.compile(
    r"\b(?:received\s+\d|digital object identifier|abstract|keywords:|corresponding author|date of publication)\b",
    re.IGNORECASE,
)
_TITLE_PAGE_CUE_RE = re.compile(
    r"\b(?:contents lists available|journal homepage|corresponding author|school of|college of|department of|university)\b",
    re.IGNORECASE,
)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.\w+\b")


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"\s+", " ", text).strip()


def _tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(_normalize_text(text))


def _looks_reference_block(text: str) -> bool:
    normalized = text.strip()
    if not normalized:
        return False
    ref_lines = len(_REFERENCE_LINE_RE.findall(normalized))
    cue_hits = len(_REFERENCE_CUE_RE.findall(normalized))
    year_hits = len(_YEAR_RE.findall(normalized))
    bibliography_hits = len(_BIBLIOGRAPHY_NAME_RE.findall(normalized))
    return (
        ref_lines >= 2
        or (ref_lines >= 1 and cue_hits >= 2 and year_hits >= 3)
        or (bibliography_hits >= 4 and year_hits >= 6)
    )


def _looks_front_matter_block(text: str) -> bool:
    normalized = text.strip()
    if not normalized:
        return False
    cue_hits = len(_FRONT_MATTER_CUE_RE.findall(normalized))
    return cue_hits >= 2


def _looks_title_page_block(text: str) -> bool:
    normalized = text.strip()
    if not normalized:
        return False
    cue_hits = len(_TITLE_PAGE_CUE_RE.findall(normalized))
    email_hits = len(_EMAIL_RE.findall(normalized))
    has_abstract = "abstract" in normalized.lower()
    return not has_abstract and (cue_hits >= 3 or email_hits >= 2 or (cue_hits >= 2 and email_hits >= 1))


def _looks_digest_block(text: str) -> bool:
    normalized = text.strip()
    if not normalized:
        return False
    return (
        "文献：" in normalized
        and ("核心创新点" in normalized or "应用与结论" in normalized or "见上节" in normalized)
    )


def build_local_embeddings(texts: list[str], dim: int = 1024) -> np.ndarray:
    matrix = np.zeros((len(texts), dim), dtype=np.float32)
    for row, text in enumerate(texts):
        tokens = _tokenize(text)
        if not tokens:
            continue
        for token in tokens:
            digest = int(hashlib.sha1(token.encode("utf-8")).hexdigest()[:8], 16)
            sign = -1.0 if digest & 1 else 1.0
            index = (digest >> 1) % dim
            matrix[row, index] += sign
    return matrix


class RetrievalEngine:
    def __init__(
        self,
        chunks: list[ChunkRecord],
        embeddings: np.ndarray,
        client: SiliconFlowClient,
        *,
        embedding_backend: str = "siliconflow",
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Chunk count and embedding count do not match.")
        self.chunks = chunks
        self.embeddings = self._normalize(embeddings)
        self.client = client
        self.embedding_backend = embedding_backend

    @staticmethod
    def _normalize(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return matrix / norms

    def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        if self.embedding_backend == "local-hash":
            query_vector = build_local_embeddings([query], dim=self.embeddings.shape[1])[0]
        else:
            query_vector = self.client.embed_texts([query])[0]
        query_vector = query_vector / max(np.linalg.norm(query_vector), 1e-8)
        scores = self.embeddings @ query_vector
        ordered = [int(idx) for idx in np.argsort(scores)[::-1]]
        filtered_ordered = [
            idx
            for idx in ordered
            if not _looks_reference_block(self.chunks[idx].text)
            and not _looks_digest_block(self.chunks[idx].text)
            and not _looks_front_matter_block(self.chunks[idx].text)
            and not _looks_title_page_block(self.chunks[idx].text)
        ] or [idx for idx in ordered if not _looks_reference_block(self.chunks[idx].text)] or ordered

        shortlisted_pdf: list[int] = []
        shortlisted_other: list[int] = []
        bucket = {"structure": 0, "window": 0}
        doc_counts: dict[str, int] = {}
        for idx in filtered_ordered:
            chunk = self.chunks[idx]
            kind = chunk.chunk_type
            if bucket[kind] >= max(4, top_k // 2):
                continue
            if doc_counts.get(chunk.doc_id, 0) >= 2:
                continue
            target = shortlisted_pdf if chunk.file_name.lower().endswith(".pdf") else shortlisted_other
            if target is shortlisted_other and len(shortlisted_other) >= 2:
                continue
            target.append(int(idx))
            bucket[kind] += 1
            doc_counts[chunk.doc_id] = doc_counts.get(chunk.doc_id, 0) + 1
            if len(shortlisted_pdf) >= max(top_k * 2, 12) and len(shortlisted_other) >= 1:
                break

        shortlisted = shortlisted_pdf + shortlisted_other

        candidates = [self.chunks[idx].text[:1200] for idx in shortlisted]
        try:
            reranked = self.client.rerank(query, candidates, top_n=min(max(top_k * 2, 12), len(candidates)))
        except Exception:
            reranked = []
        if reranked:
            ranked_candidates: list[RetrievalResult] = []
            for item in reranked:
                original_idx = shortlisted[item["index"]]
                ranked_candidates.append(
                    RetrievalResult(
                        chunk=self.chunks[original_idx],
                        score=float(item["relevance_score"]),
                    )
                )

            results: list[RetrievalResult] = []
            seen: set[str] = set()
            doc_counts: dict[str, int] = {}

            def append_candidate(result: RetrievalResult) -> None:
                chunk = result.chunk
                if chunk.chunk_id in seen:
                    return
                if doc_counts.get(chunk.doc_id, 0) >= 2:
                    return
                seen.add(chunk.chunk_id)
                doc_counts[chunk.doc_id] = doc_counts.get(chunk.doc_id, 0) + 1
                results.append(result)

            pdf_target = max(4, top_k // 2)
            for result in ranked_candidates:
                if len(results) >= pdf_target:
                    break
                if result.chunk.file_name.lower().endswith(".pdf"):
                    append_candidate(result)

            for result in ranked_candidates:
                if len(results) >= top_k:
                    break
                append_candidate(result)

            return results

        return [RetrievalResult(chunk=self.chunks[idx], score=float(scores[idx])) for idx in shortlisted[:top_k]]


def format_evidence(results: list[RetrievalResult]) -> str:
    lines = []
    for result in results:
        chunk = result.chunk
        excerpt = _INLINE_PAPER_CITATION_RE.sub("", chunk.text[:900])
        lines.append(
            f"证据编号={chunk.chunk_id} 文件={chunk.file_name} 页码={chunk.page_start}-{chunk.page_end} "
            f"相关度={result.score:.4f}\n"
            f"引用本证据时只能使用 [{chunk.chunk_id}]，不要使用原文内部参考文献编号。\n"
            f"{excerpt}"
        )
    return "\n\n".join(lines)
