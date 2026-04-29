from __future__ import annotations

import re

from .models import ProjectManifest
from .siliconflow import SiliconFlowClient
from .utils import collect_citation_ids


_TRAILING_CITATION_RE = re.compile(r"(?:\s*\[(?:CH-\d{6})(?:\s*,\s*CH-\d{6})*\])+\s*$")
_BROKEN_CITATION_RE = re.compile(r"\[CH(?:-[0-9]{0,5})?$")
_AI_STYLE_PATTERNS = [
    re.compile(pattern)
    for pattern in [
        r"其核心在于",
        r"本质上",
        r"值得注意的是",
        r"需要指出的是",
        r"案例结果表明",
        r"综上所述",
        r"不难发现",
    ]
]


def looks_complete(markdown_text: str) -> bool:
    lines = [line.strip() for line in markdown_text.splitlines() if line.strip()]
    if not lines:
        return False

    last_line = lines[-1]
    if last_line.startswith("#"):
        return False

    stripped = _TRAILING_CITATION_RE.sub("", last_line).strip()
    if not stripped:
        return False
    if _BROKEN_CITATION_RE.search(last_line):
        return False
    return stripped[-1] in "。！？.!?;；”’\"'"


def _ai_style_hit_count(text: str) -> int:
    return sum(1 for pattern in _AI_STYLE_PATTERNS if pattern.search(text))


class ReportEditor:
    def __init__(self, client: SiliconFlowClient) -> None:
        self.client = client

    def polish_section(
        self,
        manifest: ProjectManifest,
        section_text: str,
        *,
        section_title: str,
        revision_focus: str | None = None,
    ) -> str:
        original_citations = set(collect_citation_ids(section_text))
        if not original_citations:
            return section_text

        edited = self.client.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "你是中文综述总编。"
                        "请把已有章节改写得更像成熟研究者亲自撰写的综述，而不是资料摘录或模板化拼接。"
                        "必须保留原有 Markdown 层级和引用编号，不得新增资料之外的事实。"
                        "允许调整句序、段落衔接与判断语气，但不要删除关键论点。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"用户需求：\n{manifest.user_request}\n\n"
                        f"章节标题：\n{section_title}\n\n"
                        f"修订重点：\n{revision_focus or '降低 AI 腔，增强比较判断、边界意识和自然过渡。'}\n\n"
                        "请严格遵守：\n"
                        "1. 直接输出修订后的完整 Markdown 章节，不要解释。\n"
                        "2. 保留标题行与全部引用编号，不能发明新引用。\n"
                        "3. 减少“其核心在于、本质上、值得注意的是、综上所述”等模板句。\n"
                        "4. 把相邻段落之间的关系写清楚，尽量体现比较、取舍、适用场景或局限边界。\n"
                        "5. 不要把语气写得过满，避免“显著优于、完全解决、普遍适用”等绝对化表述。\n"
                        "6. 如果涉及 MISOCP、MILP、MINLP 等含整数变量的模型，不要直接改写成“凸优化问题”。\n\n"
                        f"原始章节：\n{section_text}"
                    ),
                },
            ],
            temperature=0.1,
            max_tokens=min(7000, max(1800, int(len(section_text) * 1.35))),
        ).strip()

        edited_citations = set(collect_citation_ids(edited))
        if not edited.startswith("#"):
            return section_text
        if not edited_citations:
            return section_text
        if not edited_citations.issubset(original_citations):
            return section_text
        if not looks_complete(edited):
            return section_text
        if _ai_style_hit_count(edited) > _ai_style_hit_count(section_text) + 1:
            return section_text
        return edited
