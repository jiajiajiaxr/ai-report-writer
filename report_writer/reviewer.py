from __future__ import annotations

import json
import re
from pathlib import Path

from .models import ProjectManifest, ReviewReport
from .siliconflow import SiliconFlowClient, SiliconFlowHTTPStatusError
from .utils import (
    collect_citation_ids,
    dump_json,
    extract_json_block,
    markdown_paragraphs,
    min_unique_citations_for_paragraphs,
    report_word_count,
    target_word_range,
    truncate,
    uncited_substantive_paragraphs,
)


AI_STYLE_PATTERNS = [
    ("存在绝对化措辞“首次”，容易产生无依据的首创性判断。", re.compile(r"首次")),
    ("存在绝对化措辞“完全”，容易过度承诺。", re.compile(r"完全")),
    ("存在可能机械化的总结短语“这些案例共同表明”。", re.compile(r"这些案例共同表明")),
]
TEMPLATE_META_PATTERNS = [
    ("“直接机制”", re.compile(r"直接机制")),
    ("“成立的关键前提”", re.compile(r"成立的关键前提")),
    ("“证据边界”", re.compile(r"证据边界")),
    ("“与上下文的逻辑衔接”", re.compile(r"与上下文的逻辑衔接")),
    ("“从方法谱系上看”", re.compile(r"从方法谱系上看")),
]
FACTUAL_ERROR_PATTERNS = [
    (
        "将含整数变量的 MISOCP 直接表述为“凸优化问题”，属于技术性失准表述。",
        re.compile(r"MISOCP[^。！？\n]{0,40}凸优化问题", re.IGNORECASE),
    ),
    (
        "将 KL 散度下的 DRO 一般性表述为“等价于 CVaR”，属于过度泛化，需改写为更审慎的条件性表述。",
        re.compile(r"KL散度[^。！？\n]{0,40}等价于[^。！？\n]{0,20}CVaR", re.IGNORECASE),
    ),
    (
        "将一般 φ-散度 DRO 笼统表述为与 CVaR 存在等价关系，仍然过宽，需收窄为更具体的条件性表述。",
        re.compile(
            r"(?:φ-散度|phi-divergence)[^。！？\n]{0,80}(?:等价|等价关系)[^。！？\n]{0,30}CVaR|"
            r"CVaR[^。！？\n]{0,30}(?:等价|等价关系)[^。！？\n]{0,80}(?:φ-散度|KL散度)",
            re.IGNORECASE,
        ),
    ),
]
REFERENCE_LINE_RE = re.compile(r"(?:^|\n)\s*(?:\[\d+\]|\d+\s*[.)])\s+[A-Z\u4e00-\u9fff]")
REFERENCE_CUE_RE = re.compile(r"\b(?:doi|transactions on|ieee|vol\.|no\.|pp\.|et al\.)\b", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
FRONT_MATTER_CUE_RE = re.compile(
    r"\b(?:received\s+\d|digital object identifier|abstract|keywords:|corresponding author|date of publication)\b",
    re.IGNORECASE,
)
TERM_SUPPORT_RULES = [
    (
        "φ/KL 理论表述",
        re.compile(
            r"(?:φ-散度|phi-divergence|KL散度|Kullback|entropic risk|相对熵|coherent risk)",
            re.IGNORECASE,
        ),
        [
            re.compile(
                r"(?:phi-divergence|𝜙-divergence|kullback|entropic risk|coherent risk|cvar|value-at-risk|entropy function|csisz[aá]r)",
                re.IGNORECASE,
            )
        ],
    ),
    (
        "DPMM",
        re.compile(r"(?:\bDPMM\b|Dirichlet过程混合)", re.IGNORECASE),
        [re.compile(r"(?:\bdpmm\b|dirichlet\s+process)", re.IGNORECASE)],
    ),
    (
        "Copula/Markov 时空相关",
        re.compile(r"(?:Copula|马尔可夫|Markov)", re.IGNORECASE),
        [re.compile(r"(?:copula|markov)", re.IGNORECASE)],
    ),
    (
        "可靠性/规划结论",
        re.compile(r"(?:SAIFI|扩容规划|扩展规划|长期规划|长期投资|投资决策|规划层面|投资成本|过度投资)", re.IGNORECASE),
        [re.compile(r"(?:SAIFI|reliability|planning|planner|expansion|investment|规划|投资|扩展)", re.IGNORECASE)],
    ),
    (
        "LearnAMR",
        re.compile(r"LearnAMR", re.IGNORECASE),
        [re.compile(r"learnamr", re.IGNORECASE)],
    ),
    (
        "RL-MPC / safe DRL 机制",
        re.compile(r"(?:RL-MPC|safe\s*DRL|安全过滤器|MPC作为安全过滤器|MPC 作为安全过滤器)", re.IGNORECASE),
        [re.compile(r"(?:safe\s*drl|reinforcement learning|model predictive control|\bMPC\b)", re.IGNORECASE)],
    ),
    (
        "DRRL 成熟案例否定判断",
        re.compile(r"(?:尚未|未见|暂无)[^。！？\n]{0,30}(?:DRRL|分布鲁棒强化学习)[^。！？\n]{0,20}(?:成熟案例|直接应用|应用案例)", re.IGNORECASE),
        [re.compile(r"(?:DRRL|distributionally robust reinforcement learning)", re.IGNORECASE)],
    ),
    (
        "两阶段调度设备/备用细节",
        re.compile(
            r"(?:燃气轮机|储能(?:系统)?|电制冷机|柔性负荷|快速响应设备|备用(?:容量)?|基线出力|需求响应合同|日前阶段|实时阶段)",
            re.IGNORECASE,
        ),
        [
            re.compile(
                r"(?:gas turbine|energy storage|electric chiller|flexible load|demand response|reserve|day-ahead|real-time|first stage|second stage|re-dispatch|intra-day)",
                re.IGNORECASE,
            )
        ],
    ),
    (
        "相关性/可靠性外推",
        re.compile(
            r"(?:时空相关性直接决定|储能等灵活性资源配置|结构性变化|可靠性约束|扩容规划|扩展规划|长期规划|过度投资)",
            re.IGNORECASE,
        ),
        [
            re.compile(
                r"(?:correlation|multi-interval|copula|markov|fr[eé]chet|comonotonic|reliability|planning|expansion|storage|flexible)",
                re.IGNORECASE,
            )
        ],
    ),
    (
        "规划层外推语句",
        re.compile(r"(?:合理推断|由此可推断|可外推到|具有相似的潜力|类似的潜力)", re.IGNORECASE),
        [],
    ),
]
BOUNDARY_CUE_RE = re.compile(
    r"(?:现有证据|现有文献|现有证据块|当前资料|当前检索样本|本文检索样本|当前一手(?:pdf|资料)|当前一手pdf|当前提供的一手(?:pdf|资料|证据))"
    r"[^。！？\n]{0,40}(?:不足|有限|未直接支持|未见|缺乏|未提供|不支持|无法)|"
    r"缺乏可直接核验的一手证据|缺乏直接的一手材料支持|证据块[^。！？\n]{0,20}缺乏|"
    r"不宜据此断言|不展开更细结论|只能说明|仅保留框架性判断|仅作方向性讨论|未提供细节",
    re.IGNORECASE,
)
DRO_TEXT_RE = re.compile(r"(?:\bDRO\b|分布鲁棒|distributionally robust)", re.IGNORECASE)
RO_SOURCE_ONLY_RE = re.compile(
    r"(?:\brobust optimization\b|adaptive robust|two-stage robust optimization|鲁棒优化|输电扩展规划)",
    re.IGNORECASE,
)
DRO_SOURCE_RE = re.compile(r"(?:distributionally robust|分布鲁棒)", re.IGNORECASE)


def lint_report(markdown_text: str, *, target_words: int | None = None) -> tuple[list[str], list[str]]:
    blocking: list[str] = []
    minor: list[str] = []
    stripped = markdown_text.strip()

    if not stripped:
        return ["报告为空。"], []

    if stripped[-1] not in "。！？.!?】]'\"":
        blocking.append("报告结尾缺少完整句末标点，可能存在截断。")

    if re.search(r"(得到了|表明了|说明了)\s*$", stripped):
        blocking.append("报告末尾存在未完成句子。")

    paragraphs = markdown_paragraphs(stripped)
    short_count = sum(1 for item in paragraphs if len(item) < 40)
    if short_count >= 2:
        minor.append("存在多个过短段落，结构可能偏碎。")

    uncited_substantive = [truncate(item, 120) for item in uncited_substantive_paragraphs(stripped)]
    if uncited_substantive:
        blocking.append(f"存在 {len(uncited_substantive)} 个缺少引用的实质性段落。")

    unique_citations = set(collect_citation_ids(stripped))
    required_unique = min_unique_citations_for_paragraphs(len(paragraphs))
    if paragraphs and len(unique_citations) < required_unique:
        blocking.append(
            f"全文仅使用了 {len(unique_citations)} 个不同引用编号，低于当前篇幅至少 {required_unique} 个的覆盖要求。"
        )

    if target_words:
        actual_words = report_word_count(stripped)
        min_words, max_words = target_word_range(target_words)
        if actual_words < min_words or actual_words > max_words:
            blocking.append(
                f"报告字数约为 {actual_words}，不在目标 {target_words} 字允许范围 {min_words}-{max_words} 内。"
            )

    for message, pattern in AI_STYLE_PATTERNS:
        if pattern.search(stripped):
            minor.append(message)

    for message, pattern in FACTUAL_ERROR_PATTERNS:
        if pattern.search(stripped):
            blocking.append(message)

    template_hits = [(label, len(pattern.findall(stripped))) for label, pattern in TEMPLATE_META_PATTERNS]
    template_hits = [(label, count) for label, count in template_hits if count]
    template_total = sum(count for _, count in template_hits)
    if template_total >= 6 or len(template_hits) >= 4:
        labels = "、".join(label for label, _ in template_hits[:5])
        blocking.append(f"模板化脚手架句式反复出现（如 {labels}），会明显增加 AI 生成痕迹。")
    elif template_hits:
        labels = "、".join(label for label, _ in template_hits[:4])
        minor.append(f"存在一些模板化脚手架句式（如 {labels}），建议改写为更自然的综述表达。")

    return blocking, minor


def _looks_reference_excerpt(text: str) -> bool:
    normalized = text.strip()
    if not normalized:
        return False
    ref_lines = len(REFERENCE_LINE_RE.findall(normalized))
    cue_hits = len(REFERENCE_CUE_RE.findall(normalized))
    year_hits = len(YEAR_RE.findall(normalized))
    return ref_lines >= 2 or (ref_lines >= 1 and cue_hits >= 2 and year_hits >= 3)


def _looks_digest_excerpt(text: str) -> bool:
    normalized = text.strip()
    if not normalized:
        return False
    return "文献：" in normalized and ("核心创新点" in normalized or "应用与结论" in normalized or "见上节" in normalized)


def _looks_front_matter_excerpt(text: str) -> bool:
    normalized = text.strip()
    if not normalized:
        return False
    return len(FRONT_MATTER_CUE_RE.findall(normalized)) >= 2


def _context_matches_markdown(context: str, markdown_text: str) -> bool:
    normalized = context.strip()
    if not normalized:
        return True
    if normalized in markdown_text:
        return True
    fragments = [fragment.strip() for fragment in normalized.split("…") if len(fragment.strip()) >= 24]
    if not fragments:
        compact = re.sub(r"\s+", "", normalized.replace("…", ""))
        return compact and compact[:80] in re.sub(r"\s+", "", markdown_text)
    return all(fragment in markdown_text for fragment in fragments[:2])


def _entry_support_blob(entry: dict) -> str:
    return "\n".join(
        str(entry.get(field, ""))
        for field in ("file_name", "section_title", "excerpt")
    )


def _term_support_issues(markdown_text: str, entries: list[dict]) -> list[str]:
    entry_map: dict[str, list[dict]] = {}
    for entry in entries:
        citation_id = str(entry.get("citation_id", "")).strip()
        if not citation_id:
            continue
        entry_map.setdefault(citation_id, []).append(entry)

    issues: list[str] = []
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
                issues.append(
                    f"正文写到了“{label}”，但对应引文的源文摘录未直接出现相关术语或机制，疑似超出证据：{truncate(paragraph, 120)}"
                )
                break
    return issues


def _ro_dro_mismatch_issues(markdown_text: str, entries: list[dict]) -> list[str]:
    entry_map: dict[str, list[dict]] = {}
    for entry in entries:
        citation_id = str(entry.get("citation_id", "")).strip()
        if citation_id:
            entry_map.setdefault(citation_id, []).append(entry)

    issues: list[str] = []
    for paragraph in markdown_paragraphs(markdown_text):
        if not DRO_TEXT_RE.search(paragraph):
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
        if RO_SOURCE_ONLY_RE.search(support_blob) and not DRO_SOURCE_RE.search(support_blob):
            issues.append(
                f"正文将主要支撑为 RO 的证据写成了 DRO，疑似概念错配：{truncate(paragraph, 140)}"
            )
    return issues


def lint_trace(markdown_text: str, trace_text: str) -> tuple[list[str], list[str]]:
    blocking: list[str] = []
    minor: list[str] = []
    try:
        payload = json.loads(trace_text)
    except json.JSONDecodeError:
        return ["溯源 JSON 无法解析，无法完成引用一致性审查。"], []

    if not isinstance(payload, dict):
        return ["溯源文件结构异常。"], []

    generation = payload.get("generation")
    if not isinstance(generation, dict) or not generation.get("chat_model"):
        minor.append("溯源文件缺少生成模型元信息，无法独立核验写作模型。")

    entries = payload.get("trace")
    if not isinstance(entries, list):
        return ["溯源文件缺少 trace 条目。"], minor

    paragraph_count = len(markdown_paragraphs(markdown_text))
    required_trace_entries = min_unique_citations_for_paragraphs(paragraph_count)
    if paragraph_count and len(entries) < required_trace_entries:
        blocking.append(
            f"trace 仅包含 {len(entries)} 个条目，低于当前篇幅至少 {required_trace_entries} 个的最低溯源覆盖要求。"
        )

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        citation_id = str(entry.get("citation_id", "")).strip() or "UNKNOWN"
        contexts = [item for item in entry.get("report_contexts", []) if isinstance(item, str) and item.strip()]
        excerpt = str(entry.get("excerpt", "")).strip()

        if not contexts:
            minor.append(f"引用 {citation_id} 缺少对应的正文上下文。")
            continue

        stale_contexts = [ctx for ctx in contexts if not _context_matches_markdown(ctx, markdown_text)]
        if stale_contexts:
            blocking.append(f"引用 {citation_id} 的溯源上下文与当前正文不一致，疑似旧版残留。")

        if not excerpt:
            blocking.append(f"引用 {citation_id} 缺少可核验的溯源摘录。")
            continue

        if _looks_reference_excerpt(excerpt):
            blocking.append(f"引用 {citation_id} 的溯源摘录更像参考文献列表而非正文证据，难以直接支撑对应论断。")
        elif _looks_digest_excerpt(excerpt):
            blocking.append(f"引用 {citation_id} 的溯源摘录属于二手文献摘要块，不适合作为严格审查下的主证据。")
        elif _looks_front_matter_excerpt(excerpt):
            blocking.append(f"引用 {citation_id} 的溯源摘录更像题名页或摘要前言信息，信号过弱，难以支撑具体论断。")

    context_total = sum(
        len([item for item in entry.get("report_contexts", []) if isinstance(item, str) and item.strip()])
        for entry in entries
        if isinstance(entry, dict)
    )
    required_contexts = max(4, min(paragraph_count, max(1, paragraph_count - 2)))
    if paragraph_count and context_total < required_contexts:
        blocking.append(
            f"trace 只覆盖了 {context_total} 处正文上下文，低于当前报告至少 {required_contexts} 处的可核验覆盖度。"
        )

    blocking.extend(_term_support_issues(markdown_text, [entry for entry in entries if isinstance(entry, dict)]))
    blocking.extend(_ro_dro_mismatch_issues(markdown_text, [entry for entry in entries if isinstance(entry, dict)]))

    return blocking, minor


def _trace_summary(trace_text: str, *, max_entries: int = 24) -> str:
    try:
        payload = json.loads(trace_text)
    except json.JSONDecodeError:
        return truncate(trace_text, 2000)
    if not isinstance(payload, dict):
        return truncate(trace_text, 2000)

    header_parts: list[str] = []
    generation = payload.get("generation")
    if isinstance(generation, dict):
        header_parts.append(
            "generation="
            + json.dumps(
                {
                    "provider": generation.get("provider"),
                    "chat_model": generation.get("chat_model"),
                    "embedding_backend": generation.get("embedding_backend"),
                },
                ensure_ascii=False,
            )
        )

    entries = payload.get("trace")
    if not isinstance(entries, list):
        return "\n".join(header_parts + [truncate(trace_text, 2000)])

    rendered: list[str] = []
    for entry in entries[:max_entries]:
        if not isinstance(entry, dict):
            continue
        citation_id = entry.get("citation_id", "UNKNOWN")
        source_kind = entry.get("source_kind", "")
        file_name = truncate(str(entry.get("file_name", "")), 120)
        context = truncate(" | ".join(entry.get("report_contexts", [])[:2]), 260)
        excerpt = truncate(str(entry.get("excerpt", "")), 260)
        rendered.append(
            f"[{citation_id}] source={source_kind} file={file_name}\n"
            f"context={context}\n"
            f"excerpt={excerpt}"
        )

    return "\n\n".join(header_parts + rendered)


def _dedupe_keep_order(items: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _local_review_recommended_fixes(
    blocking: list[str],
    minor: list[str],
    *,
    target_words: int,
    actual_words: int,
) -> list[str]:
    fixes: list[str] = []
    joined = "\n".join(blocking + minor)

    if "缺少引用" in joined:
        fixes.append("为每个实质段落补齐现有 [CH-xxxxxx] 引用；无法直接支撑的段落删除或改写为谨慎边界说明。")
    if "字数" in joined:
        fixes.append(
            f"将全文字数控制在目标 {target_words} 字的允许范围内，当前约 {actual_words} 字。"
        )
    if "RO" in joined and "DRO" in joined:
        fixes.append("把被 RO 证据支撑的段落改回 RO 表述，最多只保留为对 DRO 或 IES 的启示。")
    if "trace" in joined or "摘录" in joined or "溯源" in joined:
        fixes.append("逐段核对 trace 摘录，只保留能被源文直接支持的术语、机制和结论。")
    if "模板" in joined or "AI" in joined:
        fixes.append("删除固定脚手架和模板化元话语，改成自然的综述句式。")
    if "截断" in joined or "句子不完整" in joined or "结构" in joined:
        fixes.append("补齐残句并压实章节结构，避免标题后无正文或段落过碎。")
    if not fixes and minor:
        fixes.append("根据轻微问题做编辑级改写，优先消除生硬过渡、重复表达和格式噪声。")
    return _dedupe_keep_order(fixes)


def _local_review_summary(
    *,
    blocking: list[str],
    minor: list[str],
    actual_words: int,
    min_words: int,
    max_words: int,
    llm_error: Exception | None = None,
) -> str:
    parts = [
        f"已完成本地规则审查，当前字数约 {actual_words}，目标允许范围为 {min_words}-{max_words}。"
    ]
    if blocking:
        parts.append(f"发现 {len(blocking)} 个阻塞问题。")
    elif minor:
        parts.append(f"未发现阻塞问题，但发现 {len(minor)} 个轻微问题。")
    else:
        parts.append("未发现明显规则性问题。")

    if isinstance(llm_error, SiliconFlowHTTPStatusError):
        if llm_error.is_balance_insufficient:
            parts.append("由于硅基流动账户余额不足，未执行 LLM 补充审查。")
        elif llm_error.is_rate_limited:
            parts.append("由于硅基流动触发限流，未执行 LLM 补充审查。")
        else:
            parts.append("由于硅基流动暂时不可用，未执行 LLM 补充审查。")
    elif llm_error is not None:
        parts.append("由于审查模型调用失败，本次结论基于本地规则生成。")

    return "".join(parts)


class ReportReviewer:
    def __init__(self, client: SiliconFlowClient) -> None:
        self.client = client

    def review_report(
        self,
        manifest: ProjectManifest,
        *,
        markdown_path: str,
        trace_path: str,
    ) -> ReviewReport:
        markdown_file = Path(markdown_path)
        trace_file = Path(trace_path)
        markdown_text = markdown_file.read_text(encoding="utf-8")
        trace_text = trace_file.read_text(encoding="utf-8") if trace_file.exists() else "{}"

        actual_words = report_word_count(markdown_text)
        min_words, max_words = target_word_range(manifest.target_words)
        blocking, minor = lint_report(markdown_text, target_words=manifest.target_words)
        trace_blocking, trace_minor = lint_trace(markdown_text, trace_text)
        blocking.extend(trace_blocking)
        minor.extend(trace_minor)
        system_prompt = (
            "你是严格但校准过阈值的终审编辑。"
            "目标是判断这篇综述是否存在明显 AI 写作缺陷。"
            "只有当问题已经明显影响专业读者观感，或足以让读者清楚怀疑这是 AI 拼接文本时，才给 FAIL。"
            "个别常见学术句式、轻微工整感、或可通过常规编辑微调的小问题，不足以单独判定为 FAIL。"
        )
        user_prompt = (
            f"用户需求:\n{manifest.user_request}\n\n"
            f"目标字数: {manifest.target_words}\n"
            f"允许范围: {min_words}-{max_words}\n"
            f"当前估算字数: {actual_words}\n\n"
            f"报告正文片段:\n{truncate(markdown_text, 12000)}\n\n"
            f"溯源摘要:\n{_trace_summary(trace_text)}\n\n"
            "请重点审查：\n"
            "1. 是否有明显 AI 腔、模板话、空泛套话、机械过渡、重复论述；\n"
            "2. 是否有逻辑跳跃、结构松散、证据与结论贴合度不足；\n"
            "3. 是否有绝对化措辞、无依据评价；\n"
            "4. 是否有截断、句子不完整、格式异常；\n"
            "5. 是否存在技术性硬伤，例如把混合整数模型直接说成凸优化问题；\n"
            "6. 是否存在 citation-trace 错配，尤其是把参考文献列表样式的摘录当作正文证据；\n"
            "7. 是否满足目标字数范围要求。\n"
            "只有当这些问题已经明显影响专业读者观感时，才给 FAIL。"
        )
        schema_hint = """{
  "verdict": "PASS or FAIL",
  "summary": "string",
  "blocking_issues": ["string"],
  "minor_issues": ["string"],
  "recommended_fixes": ["string"]
}"""
        try:
            raw = self.client.chat_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema_hint=schema_hint,
                max_tokens=1800,
            )
        except Exception:
            fallback = self.client.chat(
                [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": (
                            f"{user_prompt}\n\n"
                            "请严格输出 JSON，不要输出解释。\n"
                            f"JSON 结构要求:\n{schema_hint}"
                        ),
                    },
                ],
                temperature=0,
                max_tokens=1800,
            )
            raw = json.loads(extract_json_block(fallback))

        review = ReviewReport.model_validate(
            {
                "verdict": raw["verdict"],
                "summary": raw["summary"],
                "blocking_issues": blocking + raw.get("blocking_issues", []),
                "minor_issues": minor + raw.get("minor_issues", []),
                "recommended_fixes": raw.get("recommended_fixes", []),
                "files_reviewed": [str(markdown_file), str(trace_file)],
            }
        )
        if review.blocking_issues:
            review.verdict = "FAIL"
        return review

    def review_report_local(
        self,
        manifest: ProjectManifest,
        *,
        markdown_path: str,
        trace_path: str,
        llm_error: Exception | None = None,
    ) -> ReviewReport:
        markdown_file = Path(markdown_path)
        trace_file = Path(trace_path)
        markdown_text = markdown_file.read_text(encoding="utf-8")
        trace_text = trace_file.read_text(encoding="utf-8") if trace_file.exists() else "{}"

        actual_words = report_word_count(markdown_text)
        min_words, max_words = target_word_range(manifest.target_words)
        blocking, minor = lint_report(markdown_text, target_words=manifest.target_words)
        trace_blocking, trace_minor = lint_trace(markdown_text, trace_text)
        blocking.extend(trace_blocking)
        minor.extend(trace_minor)

        review = ReviewReport(
            verdict="FAIL" if blocking else "PASS",
            summary=_local_review_summary(
                blocking=blocking,
                minor=minor,
                actual_words=actual_words,
                min_words=min_words,
                max_words=max_words,
                llm_error=llm_error,
            ),
            blocking_issues=_dedupe_keep_order(blocking),
            minor_issues=_dedupe_keep_order(minor),
            recommended_fixes=_local_review_recommended_fixes(
                blocking,
                minor,
                target_words=manifest.target_words,
                actual_words=actual_words,
            ),
            files_reviewed=[str(markdown_file), str(trace_file)],
        )
        if review.blocking_issues:
            review.verdict = "FAIL"
        return review

    @staticmethod
    def save_review(review: ReviewReport, path: str | Path) -> None:
        dump_json(review.model_dump(mode="json"), Path(path))
