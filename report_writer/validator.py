from __future__ import annotations

import re

from .models import ProjectManifest, SectionDraft, ValidationIssue, ValidationReport
from .siliconflow import SiliconFlowClient
from .utils import (
    collect_citation_ids,
    markdown_paragraphs,
    min_unique_citations_for_paragraphs,
    report_word_count,
    target_word_range,
    uncited_substantive_paragraphs,
)


_TRAILING_CITATION_RE = re.compile(r"(?:\s*\[(?:CH-\d{6})(?:\s*,\s*CH-\d{6})*\])+\s*$")
_BROKEN_CITATION_RE = re.compile(r"\[CH(?:-\d{0,5})?$")


def _line_has_complete_ending(line: str) -> bool:
    stripped = _TRAILING_CITATION_RE.sub("", line.strip()).strip()
    if not stripped:
        return False
    return stripped[-1] in "。！？.!?;；”’\"'"


def _check_section_content(section: SectionDraft) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    lines = section.content.splitlines()
    non_empty = [line.strip() for line in lines if line.strip()]
    if not non_empty:
        issues.append(
            ValidationIssue(
                dimension="quality",
                severity="high",
                message=f"章节《{section.title}》为空。",
                related_ids=[section.node_id],
            )
        )
        return issues

    if _BROKEN_CITATION_RE.search(section.content):
        issues.append(
            ValidationIssue(
                dimension="quality",
                severity="high",
                message=f"章节《{section.title}》存在不完整的引用编号。",
                related_ids=[section.node_id],
            )
        )

    if not section.content.startswith("#"):
        issues.append(
            ValidationIssue(
                dimension="quality",
                severity="high",
                message=f"章节《{section.title}》缺少 Markdown 标题行。",
                related_ids=[section.node_id],
            )
        )

    last_heading_level: int | None = None
    last_heading_text: str | None = None
    body_seen = False
    for line in non_empty:
        if line.startswith("#"):
            current_level = len(line) - len(line.lstrip("#"))
            heading_text = line.lstrip("#").strip()
            if len(heading_text) < 2:
                issues.append(
                    ValidationIssue(
                        dimension="quality",
                        severity="high",
                        message=f"章节《{section.title}》存在过短标题“{heading_text}”。",
                        related_ids=[section.node_id],
                    )
                )
            if last_heading_text and not body_seen and current_level <= (last_heading_level or current_level):
                issues.append(
                    ValidationIssue(
                        dimension="quality",
                        severity="high",
                        message=f"章节《{section.title}》中标题“{last_heading_text}”后缺少正文。",
                        related_ids=[section.node_id],
                    )
                )
            last_heading_level = current_level
            last_heading_text = heading_text
            body_seen = False
        else:
            body_seen = True

    if last_heading_text and not body_seen and section.node_id != "final":
        issues.append(
            ValidationIssue(
                dimension="quality",
                severity="high",
                message=f"章节《{section.title}》以标题“{last_heading_text}”结束，正文不完整。",
                related_ids=[section.node_id],
            )
        )

    if not non_empty[-1].startswith("#") and not _line_has_complete_ending(non_empty[-1]):
        issues.append(
            ValidationIssue(
                dimension="quality",
                severity="high",
                message=f"章节《{section.title}》末句缺少完整收束，疑似存在截断。",
                related_ids=[section.node_id],
            )
        )

    body_paragraphs = markdown_paragraphs(section.content)
    if not body_paragraphs:
        issues.append(
            ValidationIssue(
                dimension="quality",
                severity="high",
                message=f"章节《{section.title}》缺少可识别的正文段落。",
                related_ids=[section.node_id],
            )
        )

    uncited = uncited_substantive_paragraphs(section.content)
    if uncited:
        issues.append(
            ValidationIssue(
                dimension="factuality",
                severity="high",
                message=f"章节《{section.title}》存在 {len(uncited)} 个缺少引用的实质段落。",
                related_ids=[section.node_id],
            )
        )

    required_unique = min_unique_citations_for_paragraphs(len(body_paragraphs))
    unique_citations = set(collect_citation_ids(section.content))
    if body_paragraphs and len(unique_citations) < required_unique:
        issues.append(
            ValidationIssue(
                dimension="factuality",
                severity="high",
                message=(
                    f"章节《{section.title}》仅使用了 {len(unique_citations)} 个不同引用编号，"
                    f"低于当前篇幅至少 {required_unique} 个的要求。"
                ),
                related_ids=[section.node_id, *sorted(unique_citations)],
            )
        )

    return issues


def _check_final_report_structure(section: SectionDraft, *, target_words: int | None = None) -> list[ValidationIssue]:
    issues = _check_section_content(section)
    text = section.content
    if not text.startswith("# "):
        issues.append(
            ValidationIssue(
                dimension="quality",
                severity="high",
                message="报告缺少一级标题。",
                related_ids=[section.node_id],
            )
        )

    if "\n## " not in text:
        issues.append(
            ValidationIssue(
                dimension="logic",
                severity="high",
                message="报告缺少二级标题，整体结构不足。",
                related_ids=[section.node_id],
            )
        )

    if "\n### " not in text:
        issues.append(
            ValidationIssue(
                dimension="logic",
                severity="high",
                message="报告缺少三级标题，章节展开不足。",
                related_ids=[section.node_id],
            )
        )

    paragraph_count = len(markdown_paragraphs(text))
    if paragraph_count < 6:
        issues.append(
            ValidationIssue(
                dimension="quality",
                severity="medium",
                message="报告正文段落数量偏少，可能展开不足。",
                related_ids=[section.node_id],
            )
        )

    if target_words:
        actual_words = report_word_count(text)
        min_words, max_words = target_word_range(target_words)
        if actual_words < min_words or actual_words > max_words:
            issues.append(
                ValidationIssue(
                    dimension="quality",
                    severity="high",
                    message=(
                        f"报告字数约为 {actual_words}，不在目标 {target_words} 字允许范围 "
                        f"{min_words}-{max_words} 内。"
                    ),
                    related_ids=[section.node_id],
                )
            )

    return issues


def validate_report(
    manifest: ProjectManifest,
    sections: list[SectionDraft],
    valid_citation_ids: set[str],
    client: SiliconFlowClient,
) -> ValidationReport:
    issues: list[ValidationIssue] = []

    for section in sections:
        citations = section.citations or collect_citation_ids(section.content)
        if not citations:
            issues.append(
                ValidationIssue(
                    dimension="factuality",
                    severity="high",
                    message=f"章节《{section.title}》缺少引用编号。",
                    related_ids=[section.node_id],
                )
            )
        missing = [item for item in citations if item not in valid_citation_ids]
        if missing:
            issues.append(
                ValidationIssue(
                    dimension="factuality",
                    severity="high",
                    message=f"章节《{section.title}》包含无效引用编号：{missing}。",
                    related_ids=[section.node_id, *missing],
                )
            )

        if section.node_id == "final":
            issues.extend(_check_final_report_structure(section, target_words=manifest.target_words))
        else:
            issues.extend(_check_section_content(section))

    sample_text = ""
    if len(sections) == 1 and sections[0].node_id == "final":
        content = sections[0].content
        sample_text = content if len(content) <= 12000 else f"{content[:5000]}\n\n[...]\n\n{content[-3000:]}"
    else:
        sample_text = "\n\n".join(section.content[:1800] for section in sections[:6])

    llm_checked = False
    try:
        client.chat_json(
            system_prompt=(
                "你是报告抽样质检助手。"
                "只需快速检查文本是否存在明显截断、结构异常或引用表达混乱。"
                "不要输出主观写作风格评价。"
            ),
            user_prompt=(
                f"用户需求：\n{manifest.user_request}\n\n"
                f"报告样本：\n{sample_text}\n\n"
                "请仅做抽样检查。"
            ),
            schema_hint="""{
  "summary": "string",
  "issues": [
    {
      "dimension": "factuality|logic|quality|compliance",
      "severity": "low|medium|high",
      "message": "string"
    }
  ]
}""",
            max_tokens=700,
        )
        llm_checked = True
    except Exception:
        llm_checked = False

    passed = not any(item.severity == "high" for item in issues)
    if issues:
        summary = f"已完成规则校验，发现 {len(issues)} 个需关注问题。"
    elif llm_checked:
        summary = "已完成规则校验，并进行了抽样质检。"
    else:
        summary = "已完成规则校验。"
    return ValidationReport(passed=passed, summary=summary, issues=issues)
