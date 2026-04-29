from __future__ import annotations

import re
from collections import Counter

from .models import ChunkRecord, GlossaryTerm, UserProfile
from .siliconflow import SiliconFlowClient
from .utils import truncate


TERM_PATTERNS = [
    re.compile(r"\b[A-Z]{2,}(?:[-/][A-Z0-9]{2,})*\b"),
    re.compile(r"[A-Za-z][A-Za-z0-9-]+(?:\s+[A-Za-z][A-Za-z0-9-]+){0,4}"),
    re.compile(r"[\u4e00-\u9fff]{2,16}(?:优化|调度|系统|策略|模型|方法|风险|评估|控制|能源|鲁棒)"),
]


def extract_candidate_terms(text: str, limit: int = 100) -> list[str]:
    counter: Counter[str] = Counter()
    for pattern in TERM_PATTERNS:
        for match in pattern.findall(text):
            term = match.strip(" ,.;:()[]{}，。；：（）")
            if len(term) < 2 or term.isdigit():
                continue
            counter[term] += 1
    return [term for term, _ in counter.most_common(limit)]


def infer_user_profile(
    user_request: str,
    document_titles: list[str],
    sample_chunks: list[ChunkRecord],
    client: SiliconFlowClient,
) -> UserProfile:
    snippet = "\n".join(
        f"- {chunk.file_name} p.{chunk.page_start}-{chunk.page_end}: {truncate(chunk.text, 220)}"
        for chunk in sample_chunks[:8]
    )
    titles = "\n".join(f"- {title}" for title in document_titles[:20])
    schema = """{
  "domain": "string",
  "objectives": ["string"],
  "focus_topics": ["string"],
  "writing_preferences": ["string"],
  "inferred_from": ["string"]
}"""
    try:
        raw = client.chat_json(
            system_prompt="你是企业级写作系统的资料分析器，负责从输入资料中推断用户画像和写作侧重点。",
            user_prompt=(
                f"用户需求:\n{user_request}\n\n资料标题:\n{titles}\n\n资料样本:\n{snippet}\n\n"
                "请推断用户的专业领域、写作目标、重点议题和写作偏好。"
            ),
            schema_hint=schema,
            max_tokens=1200,
        )
        return UserProfile.model_validate(raw)
    except Exception:
        return UserProfile(
            domain="能源系统与鲁棒优化报告",
            objectives=["梳理核心研究逻辑", "生成可溯源的结构化专业报告"],
            focus_topics=document_titles[:6],
            writing_preferences=["逻辑严谨", "引用明确", "避免超出资料范围推断"],
            inferred_from=document_titles[:6],
        )


def build_glossary(chunks: list[ChunkRecord], client: SiliconFlowClient, *, max_terms: int = 30) -> list[GlossaryTerm]:
    merged_text = "\n".join(chunk.text for chunk in chunks[:30])
    candidates = extract_candidate_terms(merged_text, limit=60)
    evidence_map: dict[str, list[str]] = {}
    for term in candidates:
        evidence_map[term] = [chunk.chunk_id for chunk in chunks if term in chunk.text][:3]
    if not candidates:
        return []
    schema = """[
  {
    "term": "string",
    "definition": "string",
    "aliases": ["string"]
  }
]"""
    try:
        raw = client.chat_json(
            system_prompt="你是术语整理助手，负责将专业资料中的术语清洗成可用于写作约束的术语库。",
            user_prompt=(
                "请基于下面候选术语，筛选最重要的术语并给出简洁定义。"
                "只保留在当前资料语境下真正关键的术语，避免普通词。\n\n"
                f"候选术语:\n{candidates[:50]}"
            ),
            schema_hint=schema,
            max_tokens=1500,
        )
        terms = []
        for item in raw[:max_terms]:
            terms.append(
                GlossaryTerm(
                    term=item["term"],
                    definition=item["definition"],
                    aliases=item.get("aliases", []),
                    evidence_ids=evidence_map.get(item["term"], []),
                )
            )
        return terms
    except Exception:
        return [
            GlossaryTerm(term=term, definition="待补充定义", evidence_ids=evidence_map.get(term, []))
            for term in candidates[:max_terms]
        ]
