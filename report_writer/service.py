from __future__ import annotations

import json
import logging
from pathlib import Path
import math
import re

logger = logging.getLogger(__name__)

from .chunking import build_dual_chunks
from .config import Settings
from .editor import ReportEditor
from .exporters import compose_markdown, compose_structured_markdown, export_docx, export_html, export_markdown, export_trace
from .models import ProjectManifest, ReportArtifact, ReviewLoopResult, SectionDraft
from .parsers import parse_folder
from .planner import OutlinePlanner
from .preprocess import build_glossary, infer_user_profile
from .rag import RetrievalEngine, build_local_embeddings
from .reviewer import (
    BOUNDARY_CUE_RE,
    ReportReviewer,
    TERM_SUPPORT_RULES,
    _entry_support_blob,
    _trace_summary,
    lint_report,
    lint_trace,
)
from .siliconflow import SiliconFlowClient
from .storage import ProjectStorage
from .utils import (
    collect_citation_ids,
    flatten_outline,
    make_id,
    markdown_paragraphs,
    normalize_whitespace,
    report_word_count,
    target_word_range,
    truncate,
    uncited_substantive_paragraphs,
    utc_now,
)
from .validator import validate_report
from .writer import SectionWriter


SECTION_REWRITE_RE = re.compile(
    r"(?ms)^(###\s+(?P<num>\d+\.\d+)[^\n]*\n)(?P<body>.*?)(?=^###\s+\d+\.\d+|\Z)"
)
SECTION_REWRITE_POLICIES: dict[str, dict[str, object]] = {
    "3.1": {
        "heading": "### 3.1 两阶段 RO 调度模型：能量与备用协同优化",
        "directive": (
            "把本节改写成两阶段 RO 调度模型的综述。正文第一句不能写 DRO 或分布鲁棒。"
            "不要使用模糊集、最坏情况分布、分布不确定性等 DRO 专属措辞。"
            "如需比较 DRO，只能作为一句审慎的对照，不能把本节主体写成 DRO。"
        ),
    },
    "3.2": {
        "heading": "### 3.2 时空相关性与约束扩展：建模启示",
        "directive": (
            "把本节写成围绕时空相关性、约束扩展与建模取舍的综述或启示，"
            "不要写成已被一手证据完整验证的 IES-DRO 扩展模型。"
            "除非证据摘录里直接出现 Copula、Markov、SAIFI、规划等术语，否则不要主动展开这些专名。"
            "不要从调度证据继续外推到规划收益或可靠性指标。"
        ),
    },
    "3.3": {
        "heading": "### 3.3 常见求解思路：列与约束生成及其加速",
        "directive": (
            "不要写“C&CG 是两阶段 DRO 主流方法”。"
            "可以把 C&CG 写成两阶段 min-max、鲁棒优化或部分分布鲁棒模型中常见的分解框架，"
            "但要保持表述克制，不要把局部证据扩大成对 DRO 的普遍结论。"
        ),
    },
    "4.2": {
        "heading": "### 4.2 规划层面：鲁棒优化应用的直接启示",
        "directive": (
            "只总结 RO 在规划层的直接启示，不要再写与 DRO 的逻辑衔接、自然延伸或潜在收益。"
            "如果证据只支持 RO，就保持 RO；最多一句说明它可作为后续比较基线，"
            "但不能宣称 DRO 在规划问题中已经被证明有效。"
        ),
    },
}
SECTION_FORBIDDEN_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "3.1": [
        re.compile(r"(?:\bDRO\b|分布鲁棒|distributionally robust)", re.IGNORECASE),
        re.compile(r"模糊集"),
        re.compile(r"最坏情况分布"),
    ],
    "3.2": [
        re.compile(r"DRO扩展模型"),
        re.compile(r"IES-DRO", re.IGNORECASE),
    ],
    "3.3": [
        re.compile(r"C&CG[^。！？\n]{0,30}(?:主流方法|主流求解方法)[^。！？\n]{0,30}DRO", re.IGNORECASE),
        re.compile(r"DRO[^。！？\n]{0,30}(?:主流方法|主流求解方法)[^。！？\n]{0,30}C&CG", re.IGNORECASE),
    ],
    "4.2": [
        re.compile(r"与分布鲁棒优化（DRO）的逻辑衔接"),
        re.compile(r"DRO框架可能成为一个自然的延伸选择"),
        re.compile(r"DRO在规划中的优势"),
    ],
}
META_SECTION_TITLE_RE = re.compile(r"(?m)^\*\*[^*\n]*(?:逻辑衔接|上下文模型|方法谱系)[^*\n]*\*\*\s*$")
META_PARAGRAPH_RE = re.compile(r"(?:在逻辑衔接上|从逻辑衔接上看|从方法谱系上看|与分布鲁棒优化（DRO）的逻辑衔接在于)")
TEMPLATE_PHRASE_REPLACEMENTS = [
    ("直接机制", "具体做法"),
    ("成立的关键前提", "成立前提"),
    ("证据边界", "适用边界"),
    ("与上下文的逻辑衔接", "与前文关系"),
    ("与上下文模型的逻辑衔接", "与前文关系"),
    ("从方法谱系上看", "进一步看"),
    ("从逻辑衔接上看", "进一步看"),
    ("完全摒弃", "不再依赖"),
]
MISOCP_CONVEX_RE = re.compile(r"MISOCP([^。！？\n]{0,60})凸优化问题", re.IGNORECASE)
H3_NUMBER_RE = re.compile(r"(?m)^###\s+(\d+\.\d+)")


class ReportWriterService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.storage = ProjectStorage(settings.data_dir)
        self.client = SiliconFlowClient(settings)
        self.editor = ReportEditor(self.client)
        self.reviewer = ReportReviewer(self.client)

    def close(self) -> None:
        self.client.close()

    def check_api(self) -> dict:
        reply = self.client.chat(
            [{"role": "user", "content": "请只回复ok"}],
            temperature=0,
            max_tokens=100,
        )
        return {
            "provider": "siliconflow",
            "base_url": self.settings.base_url,
            "chat_model": self.settings.chat_model,
            "ok": reply.strip().lower().startswith("ok"),
            "reply": reply.strip(),
        }

    def ingest_folder(self, *, project_name: str, source_dir: str, user_request: str, target_words: int) -> ProjectManifest:
        documents = parse_folder(source_dir)
        chunks = build_dual_chunks(documents)
        embedding_backend = "siliconflow"
        try:
            embeddings = self.client.embed_texts([chunk.text for chunk in chunks])
        except Exception:
            embeddings = build_local_embeddings([chunk.text for chunk in chunks])
            embedding_backend = "local-hash"

        profile = infer_user_profile(
            user_request=user_request,
            document_titles=[doc.metadata.title for doc in documents],
            sample_chunks=chunks[:12],
            client=self.client,
        )
        glossary = build_glossary(chunks, self.client)

        manifest = ProjectManifest(
            project_id=make_id("proj"),
            name=project_name,
            source_dir=str(Path(source_dir).resolve()),
            user_request=user_request,
            target_words=target_words or self.settings.default_target_words,
            embedding_backend=embedding_backend,
            created_at=utc_now(),
            user_profile=profile,
            glossary=glossary,
            document_ids=[doc.metadata.doc_id for doc in documents],
            chunk_count=len(chunks),
        )
        self.storage.save_manifest(manifest)
        self.storage.save_documents(manifest.project_id, documents)
        self.storage.save_chunks(manifest.project_id, chunks)
        self.storage.save_embeddings(manifest.project_id, embeddings)
        return manifest

    def generate_outline(
        self,
        project_id: str,
        *,
        title_hint: str | None = None,
        additional_requirements: str | None = None,
    ) -> ProjectManifest:
        manifest = self.storage.load_manifest(project_id)
        chunks = self.storage.load_chunks(project_id)
        embeddings = self.storage.load_embeddings(project_id)
        retriever = RetrievalEngine(
            chunks,
            embeddings,
            self.client,
            embedding_backend=manifest.embedding_backend,
        )
        evidence = retriever.search(manifest.user_request, top_k=10)
        planner = OutlinePlanner(self.client)
        manifest.outline = planner.plan(
            manifest,
            evidence,
            title_hint=title_hint,
            additional_requirements=additional_requirements,
        )
        self.storage.save_manifest(manifest)
        return manifest

    def _collect_cited_chunks(self, chunks, markdown_text: str):
        citation_ids = set(collect_citation_ids(markdown_text))
        return [chunk for chunk in chunks if chunk.chunk_id in citation_ids]

    @staticmethod
    def _resolve_artifact(
        manifest: ProjectManifest,
        artifacts: list[ReportArtifact],
        report_id: str | None = None,
    ) -> ReportArtifact:
        if report_id:
            artifact = next((item for item in artifacts if item.report_id == report_id), None)
            if artifact is None:
                raise ValueError(f"Report {report_id} not found.")
            return artifact

        if manifest.latest_report_id:
            artifact = next((item for item in artifacts if item.report_id == manifest.latest_report_id), None)
            if artifact is not None:
                return artifact

        return max(artifacts, key=lambda item: item.created_at)

    @staticmethod
    def _markdown_blocks(markdown_text: str) -> list[dict]:
        blocks: list[dict] = []
        current_kind: str | None = None
        current_lines: list[str] = []
        current_h2: str | None = None

        def flush() -> None:
            nonlocal current_kind, current_lines, current_h2
            if current_kind and current_lines:
                blocks.append(
                    {
                        "kind": current_kind,
                        "text": "\n".join(current_lines).strip(),
                        "h2": current_h2,
                    }
                )
            current_kind = None
            current_lines = []

        for line in markdown_text.splitlines():
            if line.startswith("# "):
                flush()
                current_kind = "title"
                current_lines = [line]
                continue
            if line.startswith("## "):
                flush()
                current_h2 = line.strip()
                current_kind = "h2"
                current_lines = [line]
                continue
            if line.startswith("### "):
                flush()
                current_kind = "h3"
                current_lines = [line]
                continue

            if current_kind is None:
                current_kind = "body"
            current_lines.append(line)

        flush()
        return [block for block in blocks if block["text"]]

    @staticmethod
    def _join_markdown_blocks(blocks: list[dict]) -> str:
        return "\n\n".join(block["text"].strip() for block in blocks if block["text"].strip()).strip() + "\n"

    @staticmethod
    def _h3_numbers(markdown_text: str) -> list[str]:
        return H3_NUMBER_RE.findall(markdown_text)

    def _preserves_section_structure(self, candidate: str, reference: str) -> bool:
        return self._h3_numbers(candidate) == self._h3_numbers(reference)

    @staticmethod
    def _trace_entry_map(trace_text: str) -> dict[str, list[dict]]:
        try:
            payload = json.loads(trace_text)
        except json.JSONDecodeError:
            return {}

        if not isinstance(payload, dict):
            return {}

        entries = payload.get("trace")
        if not isinstance(entries, list):
            return {}

        entry_map: dict[str, list[dict]] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            citation_id = str(entry.get("citation_id", "")).strip()
            if citation_id:
                entry_map.setdefault(citation_id, []).append(entry)
        return entry_map

    def _unsupported_paragraph_payloads(self, markdown_text: str, trace_text: str) -> list[dict]:
        entry_map = self._trace_entry_map(trace_text)
        if not entry_map:
            return []

        payloads: list[dict] = []
        for paragraph in markdown_paragraphs(markdown_text):
            if BOUNDARY_CUE_RE.search(paragraph):
                continue
            citations = collect_citation_ids(paragraph)
            if not citations:
                continue

            support_blob = "\n".join(
                _entry_support_blob(entry)
                for citation_id in citations
                for entry in entry_map.get(citation_id, [])
            )
            if not support_blob:
                continue

            for label, paragraph_pattern, source_patterns in TERM_SUPPORT_RULES:
                if paragraph_pattern.search(paragraph) and not any(pattern.search(support_blob) for pattern in source_patterns):
                    payloads.append(
                        {
                            "label": label,
                            "paragraph": paragraph,
                            "citations": citations,
                            "support_blob": truncate(support_blob, 1800),
                        }
                    )
                    break
        return payloads

    def _repair_unsupported_report_paragraphs(
        self,
        manifest: ProjectManifest,
        markdown_text: str,
        trace_text: str,
    ) -> str:
        payloads = self._unsupported_paragraph_payloads(markdown_text, trace_text)
        if not payloads:
            return markdown_text

        repaired = markdown_text
        for payload in payloads:
            paragraph = payload["paragraph"]
            citations = payload["citations"]
            citation_set = set(citations)
            try:
                candidate = self.client.chat(
                    [
                        {
                            "role": "system",
                            "content": (
                                "你是证据边界收缩助手。"
                                "你只需要重写一个正文段落。"
                                "如果现有引文不能直接支持段落中的具体机制或强判断，就把该段改写成更保守、更短的表述；"
                                "如果连保守改写也无法成立，就只输出 DELETE。"
                                "不得新增资料外事实，不得新增新的引用编号。"
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"用户需求：\n{manifest.user_request}\n\n"
                                f"当前被审查命中的风险类型：{payload['label']}\n"
                                f"本段允许使用的引用编号：{', '.join(citations)}\n\n"
                                "请严格遵守：\n"
                                "1. 只输出一个修复后的正文段落，或者只输出 DELETE。\n"
                                "2. 只能使用本段已有的 [CH-xxxxxx] 引用编号。\n"
                                "3. 如果引文摘录没有直接出现设备名、备用、储能、可靠性指标、极端天气、尾部风险、统计一致性、规划层收益或复杂度比较，就不要写这些词。\n"
                                "4. 优先保留被摘录直接支持的框架性结论；更细的机制、应用外推和比较结论要删掉或降级成边界说明。\n\n"
                                "5. 如果引文实际支持的是鲁棒优化（RO）而不是分布鲁棒优化（DRO），就必须按 RO 重写，不能偷换成 DRO。\n"
                                "6. 不要使用‘直接机制’‘成立的关键前提’‘证据边界’‘与上下文的逻辑衔接’这类模板化脚手架句式。\n\n"
                                f"引文摘录摘要：\n{payload['support_blob']}\n\n"
                                f"待修复段落：\n{paragraph}"
                            ),
                        },
                    ],
                    temperature=0,
                    max_tokens=min(1800, max(320, int(len(paragraph) * 1.2))),
                ).strip()
            except Exception:
                continue

            if candidate == "DELETE":
                replacement = ""
            else:
                candidate_citations = set(collect_citation_ids(candidate))
                if not candidate_citations or not candidate_citations.issubset(citation_set):
                    continue
                replacement = candidate

            repaired = repaired.replace(paragraph, replacement, 1)

        repaired = normalize_whitespace(re.sub(r"\n{3,}", "\n\n", repaired))
        return repaired + "\n"

    def _repair_uncited_report_paragraphs(self, manifest: ProjectManifest, markdown_text: str) -> str:
        uncited = uncited_substantive_paragraphs(markdown_text)
        if not uncited:
            return markdown_text

        original_citations = set(collect_citation_ids(markdown_text))
        if not original_citations:
            return markdown_text

        try:
            repaired = self.client.chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "你是全文引用修复助手。"
                            "请修复整篇 Markdown 报告中缺少 [CH-xxxxxx] 引用的实质段落。"
                            "如果现有证据可以直接支持，就补上原文已出现过的 [CH-xxxxxx]；"
                            "如果不能直接支持，就删短或改成保守的边界说明。"
                            "不得新增资料外事实，不得新增新的引用编号。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"用户需求：\n{manifest.user_request}\n\n"
                            "请直接输出修复后的完整 Markdown 报告，并严格满足：\n"
                            "1. 只能使用当前报告里已经出现过的 [CH-xxxxxx] 引用编号。\n"
                            "2. 每个实质段落都必须至少保留一个 [CH-xxxxxx] 引用；如果某段没有直接证据，就删掉或改成一句边界说明。\n"
                                "3. 不要新增设备细节、极端天气、尾部风险、可靠性指标或计算复杂度等正文未直接支持的术语。\n"
                                "4. 保留现有标题结构与主线逻辑，不要把局部修复扩写成大改写。\n\n"
                                "5. 如果某段原本把 RO 写成了 DRO，修复时必须改回与证据一致的 RO 或‘对 DRO 的启示’，不能保留错配表述。\n"
                                "6. 不要生成‘与上下文的逻辑衔接’‘证据边界在于’之类的模板化元话语。\n\n"
                                "以下段落当前缺少引用，优先修复它们：\n"
                                + "\n".join(f"- {truncate(item, 220)}" for item in uncited)
                                + f"\n\n当前报告：\n{markdown_text}"
                        ),
                    },
                ],
                temperature=0,
                max_tokens=min(12000, max(2600, int(len(markdown_text) * 1.2))),
            ).strip()
        except Exception:
            return markdown_text

        repaired_citations = set(collect_citation_ids(repaired))
        if not repaired.startswith("# "):
            return markdown_text
        if not repaired_citations or not repaired_citations.issubset(original_citations):
            return markdown_text
        if uncited_substantive_paragraphs(repaired):
            return markdown_text
        return repaired

    def _citation_support_blob(
        self,
        citation_ids: set[str],
        trace_text: str,
        *,
        chunks=None,
    ) -> str:
        parts: list[str] = []
        entry_map = self._trace_entry_map(trace_text) if trace_text and trace_text.strip() not in {"", "{}"} else {}
        for citation_id in sorted(citation_ids):
            entries = entry_map.get(citation_id, [])
            if entries:
                for entry in entries[:2]:
                    parts.append(
                        "\n".join(
                            [
                                f"{citation_id}",
                                f"来源: {entry.get('file_name', '')}",
                                f"章节: {entry.get('section_title', '')}",
                                f"摘录: {truncate(str(entry.get('excerpt', '')), 900)}",
                            ]
                        ).strip()
                    )
                continue

            if chunks is None:
                continue

            for chunk in chunks:
                if str(getattr(chunk, "chunk_id", "")) != citation_id:
                    continue
                parts.append(
                    "\n".join(
                        [
                            f"{citation_id}",
                            f"来源: {getattr(chunk, 'file_name', '')}",
                            f"章节: {getattr(chunk, 'section_title', '')}",
                            f"摘录: {truncate(getattr(chunk, 'text', ''), 900)}",
                        ]
                    ).strip()
                )
                break

        return "\n\n".join(part for part in parts if part).strip()

    def _clean_report_meta_phrasing(self, markdown_text: str) -> str:
        cleaned = META_SECTION_TITLE_RE.sub("", markdown_text)
        blocks = [block.strip() for block in re.split(r"\n\s*\n", cleaned) if block.strip()]
        kept: list[str] = []

        for block in blocks:
            if block.startswith("#"):
                kept.append(block)
                continue

            normalized = normalize_whitespace(block)
            if normalized.startswith("在逻辑衔接上，本节所述的两阶段"):
                continue
            if normalized.startswith("本节讨论的考虑时空相关性的DRO扩展模型"):
                continue
            if normalized.startswith("综上所述，在调度层面，DRO通过其可调的模糊集机制"):
                continue
            if META_PARAGRAPH_RE.search(normalized) and not collect_citation_ids(normalized):
                continue

            kept.append(block)

        cleaned = "\n\n".join(kept)
        for original, replacement in TEMPLATE_PHRASE_REPLACEMENTS:
            cleaned = cleaned.replace(original, replacement)

        cleaned = MISOCP_CONVEX_RE.sub(
            lambda match: f"MISOCP{match.group(1)}混合整数锥规划问题",
            cleaned,
        )
        cleaned = normalize_whitespace(re.sub(r"\n{3,}", "\n\n", cleaned))
        return cleaned.strip() + "\n"

    def _fallback_section_cleanup(self, section_num: str, section_text: str) -> str:
        policy = SECTION_REWRITE_POLICIES.get(section_num)
        if not policy:
            return self._clean_report_meta_phrasing(section_text)

        lines = section_text.strip().splitlines()
        heading = str(policy["heading"])
        if lines:
            lines[0] = heading
        else:
            lines = [heading]
        cleaned = "\n".join(lines)

        if section_num == "3.1":
            replacements = [
                ("两阶段分布鲁棒优化（DRO）", "两阶段鲁棒优化（RO）"),
                ("分布鲁棒优化（DRO）", "鲁棒优化（RO）"),
                ("Distributionally Robust Optimization, DRO", "Robust Optimization, RO"),
                ("分布鲁棒优化", "鲁棒优化"),
                ("模糊集", "不确定集"),
                ("最坏情况分布", "最坏情形"),
                ("最坏情况下的期望", "最坏情形下的"),
                ("概率分布信息不完全精确", "不确定性信息不完全精确"),
            ]
            for original, replacement in replacements:
                cleaned = cleaned.replace(original, replacement)
            cleaned = re.sub(r"\bDRO\b", "RO", cleaned)

        if section_num == "3.2":
            cleaned = cleaned.replace("考虑时空相关性的DRO扩展模型", "时空相关性与约束扩展")
            cleaned = cleaned.replace("分布鲁棒优化（DRO）扩展模型", "相关性约束扩展")
            cleaned = cleaned.replace("DRO扩展模型", "约束扩展")

        if section_num == "3.3":
            cleaned = cleaned.replace("主流求解算法：列与约束生成算法及其加速", "常见求解思路：列与约束生成及其加速")
            cleaned = cleaned.replace(
                "列与约束生成（Column-and-Constraint Generation, C&CG）算法是求解两阶段分布鲁棒优化（Distributionally Robust Optimization, DRO）模型的主流方法",
                "列与约束生成（Column-and-Constraint Generation, C&CG）算法，是处理两阶段 min-max 结构模型的一类常见分解方法",
            )

        if section_num == "4.2":
            paragraphs = [item.strip() for item in re.split(r"\n\s*\n", cleaned) if item.strip()]
            kept = [paragraphs[0]] if paragraphs else []
            for paragraph in paragraphs[1:]:
                if re.search(r"(?:\bDRO\b|分布鲁棒|distributionally robust)", paragraph, re.IGNORECASE):
                    continue
                kept.append(paragraph)
            cleaned = "\n\n".join(kept)

        return self._clean_report_meta_phrasing(cleaned)

    def _rewrite_policy_sections(
        self,
        manifest: ProjectManifest,
        markdown_text: str,
        trace_text: str,
        *,
        chunks=None,
    ) -> str:
        def replace_section(match: re.Match[str]) -> str:
            section_num = match.group("num")
            policy = SECTION_REWRITE_POLICIES.get(section_num)
            if not policy:
                return match.group(0)

            original_section = match.group(0).strip()
            target_heading = str(policy["heading"])
            section_citations = set(collect_citation_ids(original_section))
            if not section_citations:
                return self._fallback_section_cleanup(section_num, original_section).rstrip() + "\n\n"

            support_blob = self._citation_support_blob(section_citations, trace_text, chunks=chunks)
            try:
                candidate = self.client.chat(
                    [
                        {
                            "role": "system",
                            "content": (
                                "你是综述章节定向返修助手。"
                                "你只重写一个指定的小节，目标是让标题、术语、证据边界和语气与引文严格一致。"
                                "不能引入资料外事实，不能新增引用编号。"
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"用户需求：\n{manifest.user_request}\n\n"
                                f"目标小节编号：{section_num}\n"
                                f"目标标题：{target_heading}\n"
                                f"本节返修要求：{policy['directive']}\n"
                                f"本节允许使用的引用编号：{', '.join(sorted(section_citations))}\n\n"
                                "请严格遵守：\n"
                                "1. 直接输出重写后的完整 Markdown 小节，第一行必须是目标标题。\n"
                                "2. 只能使用本节已有的 [CH-xxxxxx] 引用编号。\n"
                                "3. 每个实质段落都必须保留至少一个引用；无法直接支撑的细节就删除。\n"
                                "4. 不要使用“直接机制”“成立的关键前提”“证据边界”“与上下文的逻辑衔接”等模板化脚手架句式。\n"
                                "5. 如果证据更接近 RO 而不是 DRO，就按 RO 写，最多只保留为对后续研究的克制启示。\n"
                                "6. 如果证据没有直接出现某个术语、指标、设备或扩展方向，就不要自行补写。\n\n"
                                f"引文摘录摘要：\n{support_blob or '当前仅能使用本节已有引文编号，请保持最保守的重写。'}\n\n"
                                f"待重写小节：\n{original_section}"
                            ),
                        },
                    ],
                    temperature=0.05,
                    max_tokens=min(4200, max(1200, int(len(original_section) * 1.4))),
                ).strip()
            except Exception:
                candidate = original_section

            candidate = self._fallback_section_cleanup(section_num, candidate)
            candidate_citations = set(collect_citation_ids(candidate))
            forbidden_patterns = SECTION_FORBIDDEN_PATTERNS.get(section_num, [])
            if (
                not candidate.startswith(target_heading)
                or not candidate_citations
                or not candidate_citations.issubset(section_citations)
                or uncited_substantive_paragraphs(candidate)
                or any(pattern.search(candidate) for pattern in forbidden_patterns)
            ):
                candidate = self._fallback_section_cleanup(section_num, original_section)

            return candidate.rstrip() + "\n\n"

        rewritten = SECTION_REWRITE_RE.sub(replace_section, markdown_text)
        return self._clean_report_meta_phrasing(rewritten)

    def _expand_short_report_sections(
        self,
        manifest: ProjectManifest,
        markdown_text: str,
        *,
        min_words: int,
        max_words: int,
    ) -> str:
        blocks = self._markdown_blocks(markdown_text)
        subsection_indexes = [idx for idx, block in enumerate(blocks) if block["kind"] == "h3"]
        if not subsection_indexes:
            return markdown_text

        global_citations = set(collect_citation_ids(markdown_text))
        current_words = report_word_count(markdown_text)
        gap = min_words - current_words
        if gap <= 0:
            return markdown_text

        extra_per_section = max(800, math.ceil(gap / len(subsection_indexes)))
        updated_blocks = blocks[:]

        for idx in subsection_indexes:
            if report_word_count(self._join_markdown_blocks(updated_blocks)) >= min_words:
                break

            block = updated_blocks[idx]
            subsection_text = block["text"]
            subsection_citations = set(collect_citation_ids(subsection_text))
            if not subsection_citations:
                continue

            current_section_words = report_word_count(subsection_text)
            target_section_words = min(3200, current_section_words + extra_per_section)
            section_heading = block["h2"] or ""
            try:
                expanded = self.client.chat(
                    [
                        {
                            "role": "system",
                            "content": (
                                "你是报告扩写助手。"
                                "请在不引入资料外事实的前提下，把一个已有的小节扩写得更完整。"
                                "只能围绕当前小节已被引用支撑的事实，补充直接机制解释、成立前提、证据边界、局限和逻辑衔接。"
                                "不得新增新的 [CH-xxxxxx] 编号。"
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"用户需求：\n{manifest.user_request}\n\n"
                                f"所属上级章节：\n{section_heading or '无'}\n\n"
                                f"当前小节估算字数：{current_section_words}\n"
                                f"目标小节字数：约 {target_section_words}\n\n"
                                "请直接输出扩写后的完整 Markdown 小节，并严格满足：\n"
                                "1. 第一行保留原有 `###` 标题，不要改标题层级。\n"
                                "2. 只能使用当前小节里已经出现过的 [CH-xxxxxx] 引用编号。\n"
                                "3. 不能加入资料外新事实，不能把边界说明扩写成更强结论。\n"
                                "4. 优先补充：定义重述、直接机制解释、成立条件、限制、证据边界、与上下文的逻辑衔接。\n"
                                "5. 每个实质段落都必须至少带一个 [CH-xxxxxx] 引用。\n"
                                "6. 不得写‘合理推断可外推到规划层’之类的延伸判断。\n"
                                "7. 如果篇幅仍然偏短，请继续补充在现有证据范围内可以直接成立的前提、边界和限制，不要发散到设备细节、可靠性指标、极端天气、尾部风险、规划层推断或额外复杂度结论。\n\n"
                                f"当前小节：\n{subsection_text}"
                            ),
                        },
                    ],
                    temperature=0.1,
                    max_tokens=3200,
                ).strip()
            except Exception:
                continue

            expanded_citations = set(collect_citation_ids(expanded))
            expanded_words = report_word_count(expanded)
            if not expanded.startswith("### "):
                continue
            if not expanded_citations or not expanded_citations.issubset(subsection_citations) or not expanded_citations.issubset(global_citations):
                continue
            if expanded_words <= current_section_words:
                continue
            updated_blocks[idx] = {**block, "text": expanded}

        expanded_report = self._join_markdown_blocks(updated_blocks)
        return expanded_report if report_word_count(expanded_report) > current_words else markdown_text

    @staticmethod
    def _retarget_outline(nodes, target_words: int) -> None:
        leaves = [node for node in flatten_outline(nodes) if not node.children]
        if not leaves:
            return

        current_total = sum(max(1, node.target_words) for node in leaves)
        if current_total <= 0:
            return

        scaled = [
            max(240, min(1200, int(round(node.target_words * target_words / current_total / 10.0)) * 10))
            for node in leaves
        ]
        diff = target_words - sum(scaled)
        if diff < 0:
            largest_index = max(range(len(scaled)), key=lambda idx: scaled[idx])
            scaled[largest_index] = max(240, scaled[largest_index] + diff)

        for node, new_target in zip(leaves, scaled):
            node.target_words = new_target

        def refresh(node) -> int:
            if not node.children:
                return node.target_words
            node.target_words = sum(refresh(child) for child in node.children)
            return node.target_words

        for node in nodes:
            refresh(node)

    def _fit_report_length(self, manifest: ProjectManifest, markdown_text: str) -> str:
        target_words = manifest.target_words
        actual_words = report_word_count(markdown_text)
        min_words, max_words = target_word_range(target_words)
        if min_words <= actual_words <= max_words:
            return markdown_text

        original_citations = set(collect_citation_ids(markdown_text))
        if not original_citations:
            return markdown_text

        if actual_words < min_words:
            previous_words = -1
            for _ in range(3):
                if actual_words >= min_words or actual_words <= previous_words:
                    break
                previous_words = actual_words
                markdown_text = self._expand_short_report_sections(
                    manifest,
                    markdown_text,
                    min_words=min_words,
                    max_words=max_words,
                )
                actual_words = report_word_count(markdown_text)
            if min_words <= actual_words <= max_words:
                return markdown_text

        try:
            adjusted = self.client.chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "你是报告长度校准助手。"
                            "请在不引入资料外事实的前提下，把整篇 Markdown 报告调整到目标字数范围内。"
                            "必须保留原有标题结构和 [CH-xxxxxx] 引用体系。"
                            "如果当前偏短，就基于现有引文扩展直接机制解释、边界、局限与上下文衔接；"
                            "如果当前偏长，就压缩重复论述，但不要删掉关键结论。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"用户需求：\n{manifest.user_request}\n\n"
                            f"目标字数：{target_words}\n"
                            f"允许范围：{min_words}-{max_words}\n"
                            f"当前估算字数：{actual_words}\n\n"
                            "请直接输出调整后的完整 Markdown 报告，并严格满足：\n"
                            "1. 保留现有 Markdown 标题层级与整体逻辑顺序。\n"
                            "2. 只能使用现有报告里已经出现过的 [CH-xxxxxx] 引用编号。\n"
                            "3. 每个实质段落都必须保留至少一个 [CH-xxxxxx] 引用。\n"
                            "4. 不得加入资料外新事实，不得写‘合理推断可外推到规划层’之类的延伸判断。\n"
                            "5. 优先通过展开直接机制、条件、限制、证据边界和上下文衔接来补足篇幅，避免空话。\n"
                            "6. 如果当前证据未直接出现设备名称、备用、储能、可靠性指标、极端天气、尾部风险、规划层结论、统计一致性或求解复杂度等术语，就不要自行补入这些细节。\n\n"
                            f"现有报告：\n{markdown_text}"
                        ),
                    },
                ],
                temperature=0.1,
                max_tokens=9000,
            ).strip()
        except Exception:
            return markdown_text

        adjusted_citations = set(collect_citation_ids(adjusted))
        adjusted_words = report_word_count(adjusted)
        if not adjusted.startswith("# "):
            return markdown_text
        if not self._preserves_section_structure(adjusted, markdown_text):
            return markdown_text
        if not adjusted_citations or not adjusted_citations.issubset(original_citations):
            return markdown_text
        if not (min_words <= adjusted_words <= max_words):
            return markdown_text
        adjusted = self._repair_unsupported_report_paragraphs(manifest, adjusted, "{}")
        return self._repair_uncited_report_paragraphs(manifest, adjusted)

    def _expand_short_report_sections_strict(
        self,
        manifest: ProjectManifest,
        markdown_text: str,
        *,
        min_words: int,
        max_words: int,
    ) -> str:
        blocks = self._markdown_blocks(markdown_text)
        subsection_indexes = [idx for idx, block in enumerate(blocks) if block["kind"] == "h3"]
        if not subsection_indexes:
            return markdown_text

        global_citations = set(collect_citation_ids(markdown_text))
        current_words = report_word_count(markdown_text)
        gap = min_words - current_words
        if gap <= 0:
            return markdown_text

        extra_per_section = max(800, math.ceil(gap / len(subsection_indexes)))
        updated_blocks = blocks[:]

        for idx in subsection_indexes:
            if report_word_count(self._join_markdown_blocks(updated_blocks)) >= min_words:
                break

            block = updated_blocks[idx]
            subsection_text = block["text"]
            subsection_citations = set(collect_citation_ids(subsection_text))
            if not subsection_citations:
                continue

            current_section_words = report_word_count(subsection_text)
            target_section_words = min(3200, current_section_words + extra_per_section)
            section_heading = block["h2"] or ""
            try:
                expanded = self.client.chat(
                    [
                        {
                            "role": "system",
                            "content": (
                                "You expand one Markdown subsection of a Chinese literature review. "
                                "Keep the same subsection heading and citation ids. "
                                "Do not add facts beyond the cited material. "
                                "Grow the subsection by adding grounded comparison, scope, tradeoffs, assumptions, and limitations in natural prose. "
                                "Do not use scaffold phrases such as 'direct mechanism', 'key prerequisite', 'evidence boundary', or 'logical connection to the context'."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"User request:\n{manifest.user_request}\n\n"
                                f"Parent section:\n{section_heading or 'N/A'}\n\n"
                                f"Current subsection words: {current_section_words}\n"
                                f"Target subsection words: about {target_section_words}\n\n"
                                "Return the full rewritten Markdown subsection and obey all rules below:\n"
                                "1. The first line must keep the current `###` heading.\n"
                                "2. Only use citation ids already present in this subsection.\n"
                                "3. Every substantive paragraph must keep at least one [CH-xxxxxx] citation.\n"
                                "4. Expand with grounded comparison, applicability, assumptions, costs, and limits, not empty filler.\n"
                                "5. Do not introduce planning-level extrapolation, reliability metrics, extreme-weather details, or solver claims unless the subsection already directly supports them.\n"
                                "6. Keep the tone cautious and human-sounding.\n\n"
                                f"Current subsection:\n{subsection_text}"
                            ),
                        },
                    ],
                    temperature=0.1,
                    max_tokens=3200,
                ).strip()
            except Exception:
                continue

            expanded = self._clean_report_meta_phrasing(expanded)
            expanded_citations = set(collect_citation_ids(expanded))
            expanded_words = report_word_count(expanded)
            if not expanded.startswith("### "):
                continue
            if not expanded_citations or not expanded_citations.issubset(subsection_citations):
                continue
            if not expanded_citations.issubset(global_citations):
                continue
            if uncited_substantive_paragraphs(expanded):
                continue
            if expanded_words <= current_section_words:
                continue
            updated_blocks[idx] = {**block, "text": expanded}

        expanded_report = self._join_markdown_blocks(updated_blocks)
        return expanded_report if report_word_count(expanded_report) > current_words else markdown_text

    def _fit_report_length_strict(
        self,
        manifest: ProjectManifest,
        markdown_text: str,
    ) -> str:
        target_words = manifest.target_words
        actual_words = report_word_count(markdown_text)
        min_words, max_words = target_word_range(target_words)
        if min_words <= actual_words <= max_words:
            return markdown_text

        original_citations = set(collect_citation_ids(markdown_text))
        if not original_citations:
            return markdown_text

        if actual_words < min_words:
            previous_words = -1
            for _ in range(3):
                if actual_words >= min_words or actual_words <= previous_words:
                    break
                previous_words = actual_words
                markdown_text = self._expand_short_report_sections_strict(
                    manifest,
                    markdown_text,
                    min_words=min_words,
                    max_words=max_words,
                )
                actual_words = report_word_count(markdown_text)
            if min_words <= actual_words <= max_words:
                return markdown_text

        try:
            adjusted = self.client.chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are a report length calibration assistant. "
                            "Adjust a full Markdown report into the target word range without introducing external facts. "
                            "Keep the existing heading hierarchy and citation system. "
                            "If the report is short, expand grounded comparisons, assumptions, applicability, costs, and limitations in natural prose. "
                            "If the report is long, compress repetition without deleting key evidence. "
                            "Do not use scaffold phrases such as 'direct mechanism', 'key prerequisite', 'evidence boundary', or 'logical connection to the context'."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"User request:\n{manifest.user_request}\n\n"
                            f"Target words: {target_words}\n"
                            f"Allowed range: {min_words}-{max_words}\n"
                            f"Current estimated words: {actual_words}\n\n"
                            "Return the full adjusted Markdown report and obey all rules below:\n"
                            "1. Keep the current Markdown heading hierarchy and overall report logic.\n"
                            "2. Only use citation ids that already appear in the report.\n"
                            "3. Every substantive paragraph must keep at least one [CH-xxxxxx] citation.\n"
                            "4. Do not add planning-level extrapolation, reliability metrics, extreme-weather details, or solver claims unless the current report already directly supports them.\n"
                            "5. If the report is short, expand grounded comparison, scope, assumptions, tradeoffs, and limitations instead of padding with meta commentary.\n"
                            "6. Keep the tone cautious and human-sounding.\n\n"
                            f"Current report:\n{markdown_text}"
                        ),
                    },
                ],
                temperature=0.1,
                max_tokens=9000,
            ).strip()
        except Exception:
            return markdown_text

        adjusted = self._clean_report_meta_phrasing(adjusted)
        adjusted_citations = set(collect_citation_ids(adjusted))
        adjusted_words = report_word_count(adjusted)
        if not adjusted.startswith("# "):
            return markdown_text
        if not adjusted_citations or not adjusted_citations.issubset(original_citations):
            return markdown_text
        if not (min_words <= adjusted_words <= max_words):
            return markdown_text
        adjusted = self._repair_unsupported_report_paragraphs(manifest, adjusted, "{}")
        return self._repair_uncited_report_paragraphs(manifest, adjusted)

    @staticmethod
    def _dedupe_messages(items: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for item in items:
            normalized = item.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            ordered.append(normalized)
        return ordered

    def _local_review_findings(
        self,
        manifest: ProjectManifest,
        markdown_text: str,
        trace_text: str,
    ) -> tuple[list[str], list[str]]:
        blocking, minor = lint_report(markdown_text, target_words=manifest.target_words)
        trace_blocking, trace_minor = lint_trace(markdown_text, trace_text)
        return (
            self._dedupe_messages(blocking + trace_blocking),
            self._dedupe_messages(minor + trace_minor),
        )

    def _review_penalty(
        self,
        manifest: ProjectManifest,
        markdown_text: str,
        trace_text: str,
    ) -> tuple[int, int, int]:
        blocking, _ = self._local_review_findings(manifest, markdown_text, trace_text)
        min_words, max_words = target_word_range(manifest.target_words)
        words = report_word_count(markdown_text)
        if min_words <= words <= max_words:
            word_gap = 0
        elif words < min_words:
            word_gap = min_words - words
        else:
            word_gap = words - max_words
        return (len(blocking), word_gap, len(uncited_substantive_paragraphs(markdown_text)))

    def _repair_report_from_review(
        self,
        manifest: ProjectManifest,
        markdown_text: str,
        trace_text: str,
        *,
        revision_request: str,
        max_rounds: int = 2,
        chunks=None,
    ) -> str:
        original_citations = set(collect_citation_ids(markdown_text))
        if not original_citations:
            return markdown_text

        best = self._rewrite_policy_sections(
            manifest,
            self._clean_report_meta_phrasing(markdown_text),
            trace_text,
            chunks=chunks,
        )
        best_penalty = self._review_penalty(manifest, best, trace_text)
        min_words, max_words = target_word_range(manifest.target_words)

        for _ in range(max_rounds):
            blocking, minor = self._local_review_findings(manifest, best, trace_text)
            if not blocking:
                break

            issues_text = "\n".join(f"- {item}" for item in (blocking + minor[:4]))
            try:
                candidate = self.client.chat(
                    [
                        {
                            "role": "system",
                            "content": (
                                "你是报告审查返修助手。"
                                "你要根据阻塞问题直接修复整篇 Markdown 报告，优先消除证据错配、无引文段落、模板化脚手架和字数不达标问题。"
                                "不得引入资料外事实，不得新增新的 [CH-xxxxxx] 编号。"
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"用户需求：\n{manifest.user_request}\n\n"
                                f"本轮修订任务：\n{revision_request}\n\n"
                                f"目标字数：{manifest.target_words}\n"
                                f"允许范围：{min_words}-{max_words}\n"
                                f"当前字数：{report_word_count(best)}\n\n"
                                "请优先修复以下阻塞问题：\n"
                                f"{issues_text}\n\n"
                                "必须遵守：\n"
                                "1. 直接输出修订后的完整 Markdown 报告。\n"
                                "2. 只能使用当前报告中已经出现过的 [CH-xxxxxx] 编号。\n"
                                "3. 每个实质段落都必须保留至少一个 [CH-xxxxxx] 引用；无法直接支撑的段落删除。\n"
                                "4. 若证据实际支撑的是 RO 而非 DRO，必须改回 RO，最多只写成对 DRO 的启示。\n"
                                "5. 删除“直接机制”“成立的关键前提”“证据边界”“与上下文的逻辑衔接”“从方法谱系上看”“从逻辑衔接上看”等模板化段落或表述。\n"
                                "6. 任何关于 KL/φ-散度 与 CVaR 的关系都只能写成受条件约束的联系，不能写成普遍等价。\n"
                                "7. 如果当前字数偏短，优先扩写证据稳定的小节中的机制解释、适用条件、限制与比较，不要补资料外细节。\n\n"
                                f"trace 摘要：\n{_trace_summary(trace_text)}\n\n"
                                f"当前报告：\n{best}"
                            ),
                        },
                    ],
                    temperature=0.05,
                    max_tokens=9000,
                ).strip()
            except Exception:
                break

            candidate_citations = set(collect_citation_ids(candidate))
            if not candidate.startswith("# "):
                continue
            if not self._preserves_section_structure(candidate, best):
                continue
            if not candidate_citations or not candidate_citations.issubset(original_citations):
                continue

            candidate = self._clean_report_meta_phrasing(candidate)
            candidate = self._rewrite_policy_sections(manifest, candidate, trace_text, chunks=chunks)
            candidate = self._repair_unsupported_report_paragraphs(manifest, candidate, trace_text)
            candidate = self._repair_uncited_report_paragraphs(manifest, candidate)
            candidate = self._fit_report_length_strict(manifest, candidate)
            candidate = self._rewrite_policy_sections(manifest, candidate, trace_text, chunks=chunks)
            candidate = self._repair_unsupported_report_paragraphs(manifest, candidate, trace_text)
            candidate = self._repair_uncited_report_paragraphs(manifest, candidate)

            candidate_penalty = self._review_penalty(manifest, candidate, trace_text)
            if candidate_penalty < best_penalty:
                best = candidate
                best_penalty = candidate_penalty

        return best

    def _save_report_artifact(
        self,
        *,
        manifest: ProjectManifest,
        title: str,
        markdown_text: str,
        chunks,
    ) -> ReportArtifact:
        cited = self._collect_cited_chunks(chunks, markdown_text)
        citations = collect_citation_ids(markdown_text)
        final_section = SectionDraft(
            node_id="final",
            title=title,
            level=1,
            target_words=manifest.target_words,
            content=markdown_text,
            evidence_ids=[chunk.chunk_id for chunk in cited],
            citations=citations,
        )
        validation = validate_report(
            manifest,
            [final_section],
            {chunk.chunk_id for chunk in chunks},
            self.client,
        )

        report_id = make_id("report")
        report_dir = self.storage.project_dir(manifest.project_id) / "reports" / report_id
        report_dir.mkdir(parents=True, exist_ok=True)
        markdown_path = export_markdown(report_dir, title, markdown_text)
        html_path = export_html(report_dir, title, markdown_text)
        docx_path = export_docx(report_dir, title, markdown_text)
        trace_path = export_trace(
            report_dir,
            title,
            markdown_text,
            cited,
            validation,
            generation={
                "provider": "siliconflow",
                "chat_model": self.settings.chat_model,
                "embedding_backend": manifest.embedding_backend,
                "embedding_model": self.settings.embedding_model,
                "rerank_model": self.settings.rerank_model,
            },
        )

        artifact = ReportArtifact(
            report_id=report_id,
            project_id=manifest.project_id,
            title=title,
            markdown_path=str(markdown_path),
            html_path=str(html_path),
            docx_path=str(docx_path),
            trace_path=str(trace_path),
            review_path=None,
            validation=validation,
            created_at=utc_now(),
        )
        manifest.latest_report_id = artifact.report_id
        self.storage.save_manifest(manifest)
        self.storage.save_artifact(manifest.project_id, artifact)
        return artifact

    def _revise_markdown_draft(
        self,
        manifest: ProjectManifest,
        latest_markdown: str,
        latest_trace: str,
        revision_request: str,
    ) -> str:
        try:
            return self.client.chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "You revise an existing Chinese Markdown report with minimal, high-value edits. "
                            "Keep the heading structure and citation ids. "
                            "Do not add facts outside the provided report and trace summary. "
                            "Prefer deleting or narrowing unsupported claims rather than inventing new detail. "
                            "Do not call mixed-integer models convex optimization problems."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"User request:\n{manifest.user_request}\n\n"
                            f"Revision request:\n{revision_request}\n\n"
                            "Rules:\n"
                            "1. Return the full revised Markdown report.\n"
                            "2. Keep existing [CH-xxxxxx] citation ids only.\n"
                            "3. Every substantive paragraph must keep at least one citation; otherwise delete or shorten it.\n"
                            "4. If the evidence supports RO rather than DRO, rewrite it as RO or as a cautious implication only.\n"
                            "5. Remove AI-sounding scaffold phrases and unsupported extrapolation.\n\n"
                            f"Trace summary:\n{_trace_summary(latest_trace)}\n\n"
                            f"Current report:\n{latest_markdown}"
                        ),
                    },
                ],
                temperature=0.15,
                max_tokens=7000,
            ).strip()
        except Exception:
            return latest_markdown

    def generate_report(
        self,
        project_id: str,
        *,
        title_hint: str | None = None,
        additional_requirements: str | None = None,
        target_words: int | None = None,
    ) -> ReportArtifact:
        logger.info(f"开始生成报告: project_id={project_id}, target_words={target_words}")
        manifest = self.storage.load_manifest(project_id)
        if target_words:
            manifest.target_words = target_words
            if manifest.outline:
                self._retarget_outline(manifest.outline, target_words)
            self.storage.save_manifest(manifest)
        if not manifest.outline:
            manifest = self.generate_outline(
                project_id,
                title_hint=title_hint,
                additional_requirements=additional_requirements,
            )

        chunks = self.storage.load_chunks(project_id)
        embeddings = self.storage.load_embeddings(project_id)
        retriever = RetrievalEngine(
            chunks,
            embeddings,
            self.client,
            embedding_backend=manifest.embedding_backend,
        )
        writer = SectionWriter(self.client, retriever)
        sections = writer.generate(manifest, manifest.outline, additional_requirements=additional_requirements)
        title = title_hint or manifest.name
        markdown_text = compose_structured_markdown(title, manifest.outline, sections)
        markdown_text = self._clean_report_meta_phrasing(markdown_text)
        markdown_text = self._rewrite_policy_sections(manifest, markdown_text, "{}", chunks=chunks)
        markdown_text = self._fit_report_length_strict(manifest, markdown_text)
        markdown_text = self._repair_report_from_review(
            manifest,
            markdown_text,
            "{}",
            revision_request=additional_requirements or "请修复本地审查发现的阻塞问题，并保持证据一致性。",
        )
        artifact = self._save_report_artifact(
            manifest=manifest,
            title=title,
            markdown_text=markdown_text,
            chunks=chunks,
        )
        logger.info(f"报告生成完成: {artifact.report_id}, 字数: {artifact.word_count}")
        return artifact

    def revise_report(self, project_id: str, revision_request: str, report_id: str | None = None) -> ReportArtifact:
        manifest = self.storage.load_manifest(project_id)
        artifacts = self.storage.load_artifacts(project_id)
        if not artifacts:
            raise ValueError("No existing report to revise.")

        latest = self._resolve_artifact(manifest, artifacts, report_id)
        latest_markdown = Path(latest.markdown_path).read_text(encoding="utf-8")
        latest_trace = Path(latest.trace_path).read_text(encoding="utf-8") if latest.trace_path else "{}"
        original_citations = set(collect_citation_ids(latest_markdown))
        chunks = self.storage.load_chunks(project_id)

        revised = self._revise_markdown_draft(
            manifest,
            latest_markdown,
            latest_trace,
            revision_request,
        )
        revised_citations = set(collect_citation_ids(revised))
        if (
            not revised.startswith("# ")
            or not self._preserves_section_structure(revised, latest_markdown)
            or not revised_citations
            or not revised_citations.issubset(original_citations)
        ):
            revised = latest_markdown
        revised = self._clean_report_meta_phrasing(revised)
        revised = self._rewrite_policy_sections(manifest, revised, latest_trace, chunks=chunks)
        revised = self._fit_report_length_strict(manifest, revised)
        revised = self._repair_unsupported_report_paragraphs(manifest, revised, latest_trace)
        revised = self._repair_uncited_report_paragraphs(manifest, revised)
        revised = self._repair_report_from_review(
            manifest,
            revised,
            latest_trace,
            revision_request=revision_request,
            chunks=chunks,
        )
        return self._save_report_artifact(
            manifest=manifest,
            title=latest.title,
            markdown_text=revised,
            chunks=chunks,
        )

        revised = self.client.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "你是报告修订助手。"
                            "请基于现有报告进行定向修改，保留 Markdown 结构与引用编号。"
                            "不要加入资料之外的新事实，不要生成空泛套话，不要使用绝对化措辞。"
                            "如果审查意见涉及 citation-trace 错配，应优先删除过强细节、改用现有且更匹配的引用，或把判断降级为审慎表述。"
                            "不要把含整数变量的 MISOCP、MILP、MINLP 直接称为凸优化问题。"
                            "每个实质段落都必须至少保留一个可核验的 [CH-xxxxxx] 引用。"
                            "不要把 KL 散度或一般 φ-散度 DRO 笼统写成“等价于 CVaR”；若证据不够，只能改写成条件性联系。"
                            "如果 trace 摘录未直接出现设备名、备用、储能、柔性负荷、可靠性指标、极端天气、尾部风险、统计一致性、规划层收益或求解复杂度等术语，就不要自行补入这些细节。"
                            "如果引文实际支持的是鲁棒优化（RO）而不是分布鲁棒优化（DRO），必须如实写成 RO，或写成‘对 DRO/IES 的启示’，不得偷换概念。"
                            "如果既有小节标题与证据范围不一致，你可以在不改变 Markdown 层级的前提下小幅重命名该小节，使标题服从证据。"
                            "不要反复使用‘直接机制、成立前提、证据边界、与上下文的逻辑衔接’这类模板化脚手架句式。"
                        ),
                    },
                {
                    "role": "user",
                    "content": (
                        f"用户需求:\n{manifest.user_request}\n\n"
                        f"修订要求:\n{revision_request}\n\n"
                        "请直接输出修订后的完整 Markdown 报告，不要输出解释。\n"
                        "修订时请特别遵守：\n"
                        "1. 优先保留被 trace 直接支撑的论断。\n"
                        "2. 若某条引用对应的摘录更像参考文献列表，而不是正文证据，就不要继续用它支撑强结论。\n"
                        "3. 引用编号必须仍然来自原报告已出现的 [CH-xxxxxx] 集合。\n"
                        "4. 每个实质段落都必须至少带一个 [CH-xxxxxx] 引用；没有证据的段落要删掉或改成边界说明。\n"
                        "5. 严禁保留“KL 散度 DRO 一般等价于 CVaR”这类泛化说法。\n\n"
                        f"现有溯源摘要:\n{_trace_summary(latest_trace)}\n\n"
                        f"现有报告:\n{latest_markdown}"
                    ),
                },
            ],
            temperature=0.15,
            max_tokens=9000,
        ).strip()

        revised_citations = set(collect_citation_ids(revised))
        if not revised.startswith("# ") or not revised_citations or not revised_citations.issubset(original_citations):
            revised = latest_markdown
        revised = self._clean_report_meta_phrasing(revised)
        revised = self._rewrite_policy_sections(manifest, revised, latest_trace, chunks=chunks)
        revised = self._fit_report_length_strict(manifest, revised)
        revised = self._repair_unsupported_report_paragraphs(manifest, revised, latest_trace)
        revised = self._repair_uncited_report_paragraphs(manifest, revised)
        revised = self._repair_report_from_review(
            manifest,
            revised,
            latest_trace,
            revision_request=revision_request,
        )
        return self._save_report_artifact(
            manifest=manifest,
            title=latest.title,
            markdown_text=revised,
            chunks=chunks,
        )

    def review_report(self, project_id: str, report_id: str | None = None):
        manifest = self.storage.load_manifest(project_id)
        artifacts = self.storage.load_artifacts(project_id)
        if not artifacts:
            raise ValueError("No report found for review.")
        artifact = self._resolve_artifact(manifest, artifacts, report_id)
        try:
            review = self.reviewer.review_report(
                manifest,
                markdown_path=artifact.markdown_path,
                trace_path=artifact.trace_path,
            )
        except Exception as exc:
            review = self.reviewer.review_report_local(
                manifest,
                markdown_path=artifact.markdown_path,
                trace_path=artifact.trace_path,
                llm_error=exc,
            )
        report_dir = Path(artifact.markdown_path).parent
        review_path = report_dir / "review.json"
        self.reviewer.save_review(review, review_path)
        artifact.review_path = str(review_path)
        self.storage.save_artifact(project_id, artifact)
        return review

    def _build_revision_request(self, review) -> str:
        items = review.blocking_issues + review.minor_issues[:3] + review.recommended_fixes[:5]
        merged = "\n".join(f"- {item}" for item in items if item)
        return (
            "请按以下审查意见修订报告，并保持引用编号有效、逻辑主线稳定：\n"
            f"{merged}\n"
            "- 禁止使用明显 AI 套话、机械过渡、空泛总结。\n"
            "- 删除或改写没有证据支撑的绝对化措辞。\n"
            "- 每个实质段落都必须至少保留一个 [CH-xxxxxx] 引用；若做不到，就删除该段或缩成审慎边界说明。\n"
            "- 不要把 KL 散度或一般 φ-散度 DRO 笼统写成“等价于 CVaR”；只能写成受定义与条件约束的联系。\n"
            "- 确保每个段落都是完整句，不能出现半截句或断裂段。\n"
            "- 如果某条引用与 trace 摘录不匹配，优先改写该句或更换为现有且更匹配的引用。"
        )

    def review_until_pass(
        self,
        project_id: str,
        *,
        report_id: str | None = None,
        title_hint: str | None = None,
        additional_requirements: str | None = None,
        target_words: int | None = None,
        max_rounds: int = 3,
    ) -> ReviewLoopResult:
        if report_id:
            manifest = self.storage.load_manifest(project_id)
            if target_words:
                manifest.target_words = target_words
                if manifest.outline:
                    self._retarget_outline(manifest.outline, target_words)
                self.storage.save_manifest(manifest)
            artifacts = self.storage.load_artifacts(project_id)
            if not artifacts:
                raise ValueError("No existing report found for review loop.")
            artifact = self._resolve_artifact(manifest, artifacts, report_id)
        else:
            artifact = self.generate_report(
                project_id,
                title_hint=title_hint,
                additional_requirements=additional_requirements,
                target_words=target_words,
            )

        latest_review = None
        for round_index in range(1, max_rounds + 1):
            latest_review = self.review_report(project_id, artifact.report_id)
            if latest_review.verdict.upper() == "PASS":
                return ReviewLoopResult(
                    project_id=project_id,
                    final_report_id=artifact.report_id,
                    rounds=round_index,
                    passed=True,
                    review=latest_review,
                    artifact=artifact,
                )
            if round_index < max_rounds:
                revision_request = self._build_revision_request(latest_review)
                if additional_requirements:
                    revision_request += f"\n- 持续满足以下额外要求：{additional_requirements}"
                artifact = self.revise_report(
                    project_id,
                    revision_request,
                    report_id=artifact.report_id,
                )

        assert latest_review is not None
        return ReviewLoopResult(
            project_id=project_id,
            final_report_id=artifact.report_id,
            rounds=max_rounds,
            passed=False,
            review=latest_review,
            artifact=artifact,
        )

    def get_project(self, project_id: str):
        return self.storage.load_summary(project_id)
