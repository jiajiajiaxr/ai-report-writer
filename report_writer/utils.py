from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


_CITATION_BLOCK_RE = re.compile(r"\[(?:CH-\d{6})(?:\s*,\s*CH-\d{6})*\]")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`]+`")
_HEADING_PREFIX_RE = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")
_LATIN_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[._/+:-][A-Za-z0-9]+)*")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def make_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


def sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_whitespace(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 2.2))


def slugify(text: str) -> str:
    cleaned = re.sub(r"[^\w\u4e00-\u9fff-]+", "-", text.strip().lower())
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-")
    return cleaned or "report"


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", text)
        text = re.sub(r"\n```$", "", text)
    return text.strip()


def extract_json_block(text: str) -> str:
    cleaned = strip_code_fences(text)
    if cleaned.startswith("{") or cleaned.startswith("["):
        return cleaned
    for left, right in (("{", "}"), ("[", "]")):
        start = cleaned.find(left)
        end = cleaned.rfind(right)
        if start != -1 and end != -1 and end > start:
            return cleaned[start : end + 1]
    return cleaned


def dump_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def flatten_outline(nodes: Iterable) -> list:
    flat: list = []
    for node in nodes:
        flat.append(node)
        flat.extend(flatten_outline(node.children))
    return flat


def truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def plain_text_for_word_count(text: str) -> str:
    cleaned = _FENCED_CODE_RE.sub(" ", text)
    cleaned = _INLINE_CODE_RE.sub(" ", cleaned)
    cleaned = _MARKDOWN_LINK_RE.sub(r"\1", cleaned)
    cleaned = _CITATION_BLOCK_RE.sub(" ", cleaned)
    cleaned = _HEADING_PREFIX_RE.sub("", cleaned)
    cleaned = re.sub(r"[>*_~|-]", " ", cleaned)
    return normalize_whitespace(cleaned)


def report_word_count(text: str) -> int:
    cleaned = plain_text_for_word_count(text)
    cjk_count = len(_CJK_CHAR_RE.findall(cleaned))
    non_cjk = _CJK_CHAR_RE.sub(" ", cleaned)
    latin_count = len(_LATIN_TOKEN_RE.findall(non_cjk))
    return cjk_count + latin_count


def target_word_range(target_words: int, *, tolerance: float = 0.25) -> tuple[int, int]:
    lower = max(1, math.floor(target_words * (1 - tolerance)))
    upper = max(lower, math.ceil(target_words * (1 + tolerance)))
    return lower, upper


def collect_citation_ids(text: str) -> list[str]:
    found = re.findall(r"CH-\d{6}", text)
    unique: list[str] = []
    for item in found:
        if item not in unique:
            unique.append(item)
    return unique


_MARKDOWN_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+")


def markdown_paragraphs(text: str) -> list[str]:
    paragraphs: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            paragraphs.append("\n".join(current).strip())
            current.clear()

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            flush()
            continue
        if _MARKDOWN_HEADING_RE.match(line):
            flush()
            continue
        current.append(line)
    flush()
    return [paragraph for paragraph in paragraphs if paragraph]


def uncited_substantive_paragraphs(text: str, *, min_length: int = 60) -> list[str]:
    return [
        paragraph
        for paragraph in markdown_paragraphs(text)
        if len(paragraph) >= min_length and not collect_citation_ids(paragraph)
    ]


def min_unique_citations_for_paragraphs(paragraph_count: int) -> int:
    if paragraph_count <= 1:
        return 1
    if paragraph_count <= 3:
        return 2
    return min(8, 2 + paragraph_count // 3)
