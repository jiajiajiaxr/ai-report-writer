from __future__ import annotations

import logging
import re
import warnings
from typing import Iterable

from .editor import ReportEditor, looks_complete
from .models import OutlineNode, ProjectManifest, SectionDraft
from .rag import RetrievalEngine, RetrievalResult, format_evidence
from .siliconflow import SiliconFlowClient
from .utils import (
    collect_citation_ids,
    flatten_outline,
    markdown_paragraphs,
    min_unique_citations_for_paragraphs,
    truncate,
    uncited_substantive_paragraphs,
)

logger = logging.getLogger(__name__)

# AI腔高频模板句式检测模式
AI_SPEAK_PATTERNS = [
    (r"然而[，,]?(?:该|此|这些|这些?)(?:方法|模型|技术|方案|策略|框架|理论|优化| Approach| Method| Model)[^。]{0,30}(?:局限性|限制|挑战|不足|缺陷|短板|问题)", "AI模板: 然而该方法存在局限性..."),
    (r"(?:该|此|这些?)(?:方法|模型|技术|方案|策略|框架|理论|优化)[^。]{0,20}(?:优势|优点|特点|特征|效益)[^。]*在于", "AI模板: 这种方法的优势在于..."),
    (r"(?:其?|本方法?|该模型?|本模型?)[^。]{0,15}(?:适用边界|成立前提|前提条件|应用范围|使用范围)", "AI模板: ...适用边界/成立前提..."),
    (r"(?:这构成了?|构成?|形成了?|导致了?)[^。]{0,20}(?:主要|核心|关键|根本)[^。]{0,15}(?:限制|约束|挑战|困难|瓶颈|短板)", "AI模板: 这构成了...的主要限制..."),
    (r"(?:从而|因此|故|据此|基于此)[^。]{0,20}(?:可以|能够|有助|有助于|有利于)", "AI模板: 从而可以/有助于..."),
    (r"(?:本质上|从根本上|归根结底而言)[^。]{0,30}", "AI模板: 本质上..."),
    (r"(?:值得注意的是|需要指出的是|应当指出的是|必须强调的是)[^。]*", "AI模板: 值得注意的是..."),
    (r"(?:综上所述|总之|总而言之|简而言之)[^。]{0,50}", "AI模板: 综上所述..."),
    (r"(?:首先|其次|再次|最后|此外|另外|与此同时)[，,][^。]*[，。][^。]*[，。][^。]*", "AI模板: 首先...其次...再次..."),
    (r"(?:一方面|另一方面|此外|与此同时)[，,]", "AI模板: 一方面...另一方面..."),
    (r"(?:经过?|在此基础上?|在此基础上进一步?|在此基础上)", "AI模板: 在此基础上..."),
    (r"(?:在.*?方面|在.*?角度上?|从.*?来看?|从.*?而言)", "AI模板: 在...方面..."),
    (r"经济性[与和]鲁棒性[^。]*?(?:平衡|权衡|取舍|折中| trade-off)", "AI模板: 经济性与鲁棒性的平衡/权衡..."),
    (r"(?:过于保守|过于乐观|介于两者之间|优于|劣于)[^。]{0,30}(?:经济性|鲁棒性|保守性)", "AI模板: 过于保守...经济性..."),
    (r"(?:模糊集|不确定集)[^。]{0,20}(?:太宽|过大|过小|太窄|收紧|扩大)", "AI模板: 模糊集太宽/太窄..."),
    (r"(?:计算复杂度|计算成本|计算负担)[^。]{0,20}(?:高|增加|增大|膨胀)", "AI模板: 计算复杂度高..."),
    (r"(?:双层|min-max|极大极小|minimax)[^。]{0,20}(?:优化|问题|模型)", "AI模板: 双层min-max优化..."),
    (r"(?:对偶转化|对偶理论|拉格朗日对偶)[^。]{0,20}(?:转化|转换|重构)", "AI模板: 对偶转化..."),
    (r"(?:数据有限|样本有限|历史数据不足|数据不足)[^。]{0,30}(?:难以|无法|不足以|不适合)", "AI模板: 数据有限...难以..."),
    (r"(?:分布偏移|分布漂移|分布不确定)[^。]{0,20}(?:风险|问题|挑战)", "AI模板: 分布偏移风险..."),
]

# 需要削减权重的关键词组合（这些观点不该反复强调）
REDUNDANT_CLAIMS = [
    "经济性和鲁棒性的平衡",
    "保守性与经济性的权衡",
    "计算复杂度高",
    "双层优化问题",
    "模糊集半径",
    "数据有限",
    "分布偏移",
]


def _detect_ai_speak(text: str) -> list[tuple[str, str]]:
    """检测AI腔句式，返回 (匹配文本, 模板类型) 列表"""
    issues = []
    paragraphs = markdown_paragraphs(text)
    for para in paragraphs:
        # 检测AI腔句式
        for pattern, description in AI_SPEAK_PATTERNS:
            if re.search(pattern, para):
                issues.append((truncate(para, 80), description))
    return issues


def _count_redundant_claims(text: str) -> int:
    """统计重复提及的冗余观点数量"""
    count = 0
    paragraphs = markdown_paragraphs(text)
    for para in paragraphs:
        for claim in REDUNDANT_CLAIMS:
            if claim in para:
                count += 1
    return count


def _has_excessive_ai_speak(text: str, threshold: int = 3) -> bool:
    """检测是否AI腔过多（超过阈值段落有AI腔句式）"""
    ai_paragraph_count = 0
    paragraphs = markdown_paragraphs(text)
    for para in paragraphs:
        for pattern, _ in AI_SPEAK_PATTERNS:
            if re.search(pattern, para):
                ai_paragraph_count += 1
                break
    return ai_paragraph_count >= threshold


def _check_memory_overlap(new_text: str, memory: str) -> list[str]:
    """检查新章节与memory的重叠内容，避免重复已论述观点"""
    if not memory or memory == "尚无已生成章节。":
        return []

    issues = []
    new_lower = new_text.lower()
    memory_lower = memory.lower()

    # 检测新内容是否与memory中的已有论点重复
    redundant_phrases = [
        ("经济性与鲁棒性", "经济性与鲁棒性平衡已在其他章节论述过"),
        ("保守性.*代价", "保守性与代价的权衡已在其他章节详述"),
        ("计算复杂度", "计算复杂度问题已在其他章节讨论"),
        ("数据有限", "数据局限性已在其他章节提及"),
        ("分布偏移", "分布偏移风险已在其他章节说明"),
    ]

    for pattern, message in redundant_phrases:
        if re.search(pattern, new_lower) and re.search(pattern, memory_lower):
            issues.append(message)

    return issues[:3]  # 最多返回3个重叠警告


def _leaf_nodes(nodes: Iterable[OutlineNode]) -> list[tuple[int, OutlineNode]]:
    collected: list[tuple[int, OutlineNode]] = []

    def visit(node: OutlineNode, depth: int) -> None:
        if not node.children:
            collected.append((depth, node))
            return
        for child in node.children:
            visit(child, depth + 1)

    for node in nodes:
        visit(node, 1)
    return collected


SECTION_TERM_RE = re.compile(r"[A-Za-z][A-Za-z0-9-]{1,}|[\u4e00-\u9fff]{2,6}")
GENERIC_SECTION_TERMS = {
    "研究",
    "方法",
    "问题",
    "系统",
    "场景",
    "应用",
    "未来",
    "展望",
    "启示",
    "控制",
    "优化",
    "综合",
    "能源",
    "调度",
    "分布",
    "鲁棒",
    "阶段",
    "模型",
    "框架",
}
SECTION_SUPPORT_RULES = [
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


def _is_frontier_rl_section(node: OutlineNode) -> bool:
    upper = node.title.upper()
    return node.node_id.startswith("S5") and ("强化学习" in node.title or "DRRL" in upper or "RL" in upper)


def _safe_goal(node: OutlineNode) -> str:
    if node.node_id == "S2.2":
        return (
            "只讨论一手 PDF 直接支持的 φ-散度 / KL 相关理论。"
            "如果证据不能明确支持某种等价关系，就只能写成'在特定构造或条件下可与某类经典风险度量建立联系'，"
            "不要把一般 φ-散度或 KL-DRO 笼统写成等价于 CVaR。"
        )
    if node.node_id == "S3.1":
        return (
            "聚焦一手 PDF 直接支持的两阶段 DRO 调度框架本身，例如日前/实时两阶段、能量与备用协同、"
            "以及鲁棒性与经济性的权衡。若当前资料未直接支持 DPMM 等高级相关性模型，就不要写 DPMM。"
        )
    if node.node_id == "S3.2":
        return (
            "把本节写成'时空相关性与可靠性约束的扩展方向'。"
            "只有当一手 PDF 正文直接出现 Copula、Markov、SAIFI 或扩容规划等术语时，才讨论这些具体机制；"
            "否则仅保留更一般的数据驱动相关性建模与约束扩展，并明确证据边界，"
            "不要从有限证据继续外推出规划或可靠性收益。"
        )
    if node.node_id == "S4.2":
        return (
            "只总结当前一手 PDF 直接支持的长期规划 / 投资启示。"
            "如果现有资料未直接支持 SAIFI、扩容规划或'避免过度投资'这类具体结论，"
            "就只写证据边界，不要从调度层证据进一步推断规划层潜力或收益。"
        )
    if _is_frontier_rl_section(node):
        return (
            "只在一手 PDF 证据直接支持时讨论 DRRL。"
            "如果当前资料只直接支持 LearnAMR、RL-MPC 等学习型控制案例，而不直接支持 DRRL 本身，"
            "就必须明确写出证据边界，不得把 LearnAMR、建筑能源控制、ADMM 或一般 DRO 算法写成 DRRL 已被验证的应用。"
        )
    return node.goal


def _section_terms(node: OutlineNode) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for token in SECTION_TERM_RE.findall(f"{node.title} {_safe_goal(node)}"):
        normalized = token.lower()
        if normalized in seen or normalized in GENERIC_SECTION_TERMS or len(normalized) <= 1:
            continue
        seen.add(normalized)
        terms.append(token)
    return terms[:14]


def _section_keywords(node: OutlineNode) -> list[str]:
    title = node.title.lower()
    goal = _safe_goal(node).lower()
    if _is_frontier_rl_section(node):
        return [
            "reinforcement learning",
            "q-learning",
            "distributionally robust q-learning",
            "learnamr",
            "model predictive control",
            "rl-mpc",
            "distribution shift",
            "tv",
            "total variation",
            "interactive data collection",
            "building energy",
            "hvac",
        ]
    if "wasserstein" in title:
        return ["wasserstein", "kantorovich", "rubinstein", "lipschitz", "optimal transport"]
    if "φ" in node.title or "散度" in node.title or "kl" in title or "cvar" in title:
        return [
            "phi-divergence",
            "kullback-leibler",
            "kl divergence",
            "entropic risk measure",
            "coherent risk measure",
            "duality",
            "ambiguity set",
        ]
    if "矩" in node.title or "二阶锥" in node.title or "misocp" in title:
        return ["moment", "mean", "covariance", "affine", "second-order cone", "soc", "misocp"]
    if node.node_id == "S3.1":
        return [
            "two-stage robust optimization",
            "first stage",
            "second stage",
            "day-ahead",
            "real-time",
            "re-dispatch",
            "reserve",
            "virtual power plant",
            "demand response",
        ]
    if node.node_id == "S3.2":
        return [
            "correlation structure",
            "multi-interval uncertainty",
            "marginal ambiguity set",
            "fréchet",
            "copula",
            "comonotonicity",
            "reliability",
            "planning",
        ]
    if node.node_id == "S4.2":
        return [
            "transmission expansion planning",
            "investment planning",
            "long-term planning",
            "adaptive robust planning",
            "network expansion",
            "planning under uncertainty",
        ]
    if "规划" in node.title or "可靠性" in node.title or "扩容" in goal or "投资" in goal:
        return [
            "expansion planning",
            "reliability",
            "investment",
            "distribution system",
            "gas",
            "demand response",
            "saifi",
            "load loss",
            "n-1",
            "resilience",
        ]
    if "算法" in node.title or "c&cg" in title or "列与约束生成" in node.title:
        return ["column-and-constraint generation", "c&cg", "benders", "decomposition"]
    return []


def _chunk_blob(result: RetrievalResult) -> str:
    chunk = result.chunk
    return f"{chunk.file_name}\n{chunk.section_title}\n{chunk.text}".lower()


def _dedupe_results(results: list[RetrievalResult]) -> list[RetrievalResult]:
    deduped: list[RetrievalResult] = []
    seen: set[str] = set()
    for item in results:
        if item.chunk.chunk_id in seen:
            continue
        seen.add(item.chunk.chunk_id)
        deduped.append(item)
    return deduped


def _section_specific_results(node: OutlineNode, results: list[RetrievalResult]) -> list[RetrievalResult]:
    if not results:
        return results

    if node.node_id == "S2.2":
        theory = [
            item
            for item in results
            if "distributionally-robust-optimization.pdf" in item.chunk.file_name.lower()
            and re.search(
                r"(?:phi-divergence|𝜙-divergence|kullback|entropic risk|coherent risk|entropy function|cvar|value-at-risk)",
                _chunk_blob(item),
                re.IGNORECASE,
            )
        ]
        if theory:
            return _dedupe_results(theory)[:6]

    if node.node_id == "S3.1":
        strong = [
            item
            for item in results
            if re.search(
                r"(?:first stage|second stage|day-ahead|real-time|re-dispatch|redispatch|reserve|recourse|demand response)",
                _chunk_blob(item),
                re.IGNORECASE,
            )
        ]
        scenario = [
            item
            for item in results
            if re.search(r"(?:virtual power plant|energy community|photovoltaic output|load power)", _chunk_blob(item), re.IGNORECASE)
        ]
        if strong:
            return _dedupe_results(strong + scenario)[:6]

    if node.node_id == "S3.2":
        correlation = [
            item
            for item in results
            if "distributionally-robust-optimization.pdf" in item.chunk.file_name.lower()
            and re.search(
                r"(?:marginal ambiguity|fr[ée]chet|copula|comonotonic|correlation)",
                _chunk_blob(item),
                re.IGNORECASE,
            )
        ]
        extensions = [
            item
            for item in results
            if re.search(
                r"(?:multi-interval uncertainty|nonanticipativity|reliability|planning)",
                _chunk_blob(item),
                re.IGNORECASE,
            )
        ]
        if correlation or extensions:
            return _dedupe_results(correlation + extensions)[:6]

    return results


def _filter_retrieved_results(node: OutlineNode, results: list[RetrievalResult]) -> list[RetrievalResult]:
    if not results:
        return results

    section_specific = _section_specific_results(node, results)
    if section_specific is not results:
        return section_specific

    profile_keywords = _section_keywords(node)
    if profile_keywords:
        primary = [item for item in results if item.chunk.file_name.lower().endswith(".pdf")]
        matched = [
            item
            for item in primary
            if any(
                keyword in item.chunk.text.lower() or keyword in item.chunk.section_title.lower()
                for keyword in profile_keywords
            )
        ]
        if matched:
            return matched[: min(len(matched), 6)]
        if primary:
            return primary[: min(len(primary), 6)]
        return results[: min(len(results), 6)]

    terms = _section_terms(node)
    if not terms:
        return results
    matched = [
        item
        for item in results
        if any(term.lower() in item.chunk.text.lower() or term.lower() in item.chunk.section_title.lower() for term in terms)
    ]
    return matched if len(matched) >= max(3, min(5, max(1, len(results) // 2))) else results


def _section_guidance(node: OutlineNode) -> str:
    rules = [
        "不要把资料逐条罗列成文献摘要，要把材料组织成有判断的综述叙述。",
        "每段只承担一个明确功能：界定问题、解释方法、比较方案、说明适用场景或指出限制。",
        "至少写出一处'为什么这样建模/求解'的原因，避免只描述'做了什么'。",
        "至少写出一处比较或取舍，说明该方法相对替代方案的优势与代价。",
        "引用只使用证据块中出现的编号，并让引用紧跟相关句子。",
        "每个实质段落都必须出现至少一个可核验的 [CH-xxxxxx] 引用；如果某段没有直接证据，就删掉或改成更审慎的表述。",
        "若证据块同时包含原始论文 PDF 和综述性 DOCX，优先使用原始论文 PDF 作为主要依据。",
        "不要把参考文献列表、作者信息或 bibliography 段落当成事实证据来写结论。",
        "不要把 KL 散度或更一般的 φ-散度 DRO 笼统写成'等价于 CVaR'；如需提及，只能写成在特定构造、风险度量或对偶条件下可建立联系。",
        "不要仅凭文件名、题名页或二手摘要判断事实；只有被引证据正文直接出现的术语和结论，才允许写进报告。",
    ]

    if node.node_id.startswith("S2"):
        rules.extend(
            [
                "理论章节要写清楚模糊集的构造依据、对偶转化路径，以及成立条件或计算代价。",
                "不要把不同模糊集说成普遍更优，而要说明它们分别适合什么数据条件与风险偏好。",
                "如果涉及 MISOCP、MILP、MINLP 等含整数变量的模型，不要直接称其为'凸优化问题'；最多说明其连续部分保留凸锥结构，或说明它可被重构为求解器可处理的确定性模型。",
            ]
        )
    if node.node_id.startswith("S3"):
        rules.extend(
            [
                "模型章节必须写出建模对象、关键不确定性、求解框架，以及为什么该框架适合综合能源系统。",
                "至少指出一种边界条件、代价或潜在失败模式，例如数据质量、计算规模或相关性估计误差。",
                "像 DPMM、Copula、Markov、SAIFI 这类专名，只有在被引一手 PDF 正文里直接出现时才能写；否则改写为更一般的数据驱动相关性模型或可靠性约束扩展。",
                "像燃气轮机、储能、电制冷机、柔性负荷、备用这些设备或机制细节，也只有在被引 PDF 正文直接出现时才能写；否则只保留两阶段结构与不确定性框架。",
                "不要从一般相关性建模直接外推出电气热匹配、灵活性资源配置或可靠性收益；证据不足时要明确写成扩展方向与边界。",
            ]
        )
    if node.node_id.startswith("S4"):
        rules.extend(
            [
                "应用章节不能只说'效果更好'，必须说明比较对象、改善方向，以及代价是否增加。",
                "至少明确一个场景匹配关系，例如更适合日前调度、扩容规划、强相关风电场等。",
                "若没有一手 PDF 直接支持 SAIFI、扩容规划、避免过度投资等专门结论，就只能保留一般性的'可靠性与投资权衡'表述。",
                "不要写'可以合理推断其在规划层面具有相似潜力'、'可进一步外推到长期投资'之类的延伸句。",
            ]
        )
    if node.node_id.startswith("S5"):
        rules.extend(
            [
                "前沿展望要基于现有资料延伸，不要虚构路线图。",
                "结尾应保留审慎语气，指出真正仍待解决的问题。",
                "如果讨论 DRRL，只能使用明确提到强化学习、分布漂移、TV 距离、样本效率或 support shift 的证据。",
                "不要把综合能源系统里的 ADMM、分布式交易或一般 DRO 求解算法，直接挪用为 DRRL 的证据。",
                "如果把 DRRL 联系到 LearnAMR、建筑能源控制或其他具体工程场景，必须明确写成'方法启示'或'可能的借鉴'，不能写成已经被直接验证的事实。",
                "如果一手 PDF 证据并未直接覆盖 DRRL，就应明确说明证据不足，并把本节写成'潜在交叉方向'而不是具体方法评述。",
                "本节不得把 LearnAMR、建筑能源控制、RL-MPC 或 ADMM 写成 DRRL 已经采用或已经验证的技术路线。",
                "如果被引 PDF 没有直接写出 RL-MPC、safe DRL、安全过滤器等表述，就不要把这些具体机制写成事实。",
            ]
        )

    return "\n".join(f"- {rule}" for rule in rules)


class SectionWriter:
    def __init__(self, client: SiliconFlowClient, retriever: RetrievalEngine) -> None:
        self.client = client
        self.retriever = retriever
        self.editor = ReportEditor(client)

    def _model_review_section(
        self,
        current_section: str,
        previous_summaries: str,
    ) -> dict:
        """
        用大模型审查当前章节与已生成章节的内容重复。
        返回结构化结果: {
            "repeat": bool,
            "severity": "low" | "medium" | "high",
            "overlaps": [
                {
                    "current_paragraph": "重复内容所在的当前段落摘要",
                    "overlaps_with": "对应的已发布章节",
                    "overlapping_content": "重复的具体内容",
                    "suggested_fix": "修改建议"
                }
            ]
        }
        """
        system_prompt = "你是资深学术审稿人，擅长检测学术综述中的内容重复，特别是同一项目内不同章节间的重复论点、数据和结论。"

        user_prompt = f"""已发布章节摘要：
{previous_summaries if previous_summaries else "尚无已发布章节"}

待审章节：
{current_section}

请仔细比对"待审章节"与"已发布章节摘要"的内容，检测是否存在：
1. 具体观点、结论重复（如"经济性与鲁棒性的平衡"、"计算复杂度高"等在多处反复出现）
2. 相同数据或案例重复使用
3. 相似论述结构或论证路径
4. 同一文献引用被用于支撑相同的通用结论

只输出JSON格式结果。
JSON 结构要求:
{{
    "repeat": true或false,
    "severity": "low"或"medium"或"high",  // low=无重复或轻微，medium=有可接受的重复，high=严重重复需修改
    "overlaps": [
        {{
            "current_paragraph": "重复内容所在的当前段落摘要（不超过80字）",
            "overlaps_with": "对应的已发布章节标题",
            "overlapping_content": "重复的具体内容（不超过50字）",
            "suggested_fix": "修改建议，说明如何改写此段以避免重复（不超过100字）"
        }}
    ]
}}"""

        schema_hint = """{
    "repeat": true或false,
    "severity": "low"或"medium"或"high",
    "overlaps": [
        {
            "current_paragraph": "重复内容所在的当前段落摘要（不超过80字）",
            "overlaps_with": "对应的已发布章节标题",
            "overlapping_content": "重复的具体内容（不超过50字）",
            "suggested_fix": "修改建议（不超过100字）"
        }
    ]
}"""

        try:
            result = self.client.chat_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema_hint=schema_hint,
                max_tokens=2000,
            )
            return result
        except Exception:
            return {"repeat": False, "severity": "low", "overlaps": []}

    def _manual_retrieval(self, node: OutlineNode) -> list[RetrievalResult]:
        candidates: list[RetrievalResult] = []
        if node.node_id == "S2.2":
            for chunk in self.retriever.chunks:
                if "distributionally-robust-optimization.pdf" not in chunk.file_name.lower():
                    continue
                blob = f"{chunk.section_title}\n{chunk.text}".lower()
                score = 0.0
                if re.search(r"(?:phi-divergence|𝜙-divergence|kullback|kl divergence|entropy function)", blob, re.IGNORECASE):
                    score += 4.0
                if re.search(r"(?:entropic risk|coherent risk|value-at-risk|cvar)", blob, re.IGNORECASE):
                    score += 4.0
                if re.search(r"(?:duality|dual representation|conjugate)", blob, re.IGNORECASE):
                    score += 2.0
                if chunk.page_start in {22, 23, 24, 79, 80, 81, 82, 83}:
                    score += 1.5
                if score <= 0:
                    continue
                candidates.append(RetrievalResult(chunk=chunk, score=score))

        if node.node_id == "S4.2":
            for chunk in self.retriever.chunks:
                if "输电扩展规划" not in chunk.file_name and "transmission expansion" not in chunk.file_name.lower():
                    continue
                blob = f"{chunk.section_title}\n{chunk.text}".lower()
                score = 0.0
                if re.search(r"(?:planning|investment|transmission expansion|uncertainty|stochastic programming|robust optimization)", blob, re.IGNORECASE):
                    score += 4.0
                if re.search(r"(?:correlation|covariance|scenario|random variable|normal distribution)", blob, re.IGNORECASE):
                    score += 2.5
                if re.search(r"(?:reliability|expansion|cost)", blob, re.IGNORECASE):
                    score += 1.5
                if chunk.page_start in {3, 4, 5, 6, 7}:
                    score += 1.0
                if score <= 0:
                    continue
                candidates.append(RetrievalResult(chunk=chunk, score=score))

        if node.node_id == "S3.2":
            for chunk in self.retriever.chunks:
                file_name = chunk.file_name.lower()
                blob = f"{chunk.section_title}\n{chunk.text}".lower()
                score = 0.0
                if "distributionally-robust-optimization.pdf" in file_name:
                    if re.search(r"(?:marginal ambiguity|fr[ée]chet|copula|comonotonic|correlation)", blob, re.IGNORECASE):
                        score += 4.0
                    if chunk.page_start in {42, 43, 44, 170, 171, 172}:
                        score += 1.5
                elif "multi-energy" in file_name or "robustoptimizationfor integrated energy systems" in file_name:
                    if re.search(r"(?:multi-interval uncertainty|nonanticipativity|uncertainty set|reliability)", blob, re.IGNORECASE):
                        score += 4.0
                    if chunk.page_start in {4, 13}:
                        score += 1.0
                elif "虚拟发电厂" in chunk.file_name or "two-stage robust optimization considering the uncertainty of sources and loads" in file_name:
                    if re.search(r"(?:first stage|second stage|comparative schemes|uncertainty of sources and loads)", blob, re.IGNORECASE):
                        score += 2.5
                    if chunk.page_start in {8, 12}:
                        score += 1.0
                if score <= 0:
                    continue
                candidates.append(RetrievalResult(chunk=chunk, score=score))

        if not candidates:
            return []

        candidates.sort(key=lambda item: (item.score, -item.chunk.page_start, -item.chunk.position), reverse=True)

        selected: list[RetrievalResult] = []
        seen_pages: set[int] = set()
        for item in candidates:
            if item.chunk.page_start in seen_pages:
                continue
            seen_pages.add(item.chunk.page_start)
            selected.append(item)
            if len(selected) >= 6:
                break
        return selected

    def _repair_section(
        self,
        manifest: ProjectManifest,
        *,
        depth: int,
        node: OutlineNode,
        evidence: str,
        memory: str,
        additional_requirements: str | None,
    ) -> str:
        heading_marks = "#" * min(depth + 1, 4)
        section_goal = _safe_goal(node)
        min_unique_citations = min_unique_citations_for_paragraphs(max(1, round(node.target_words / 220)))
        return self.client.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "你是报告修复助手。"
                        "任务是把一个存在结构缺陷或截断风险的章节修复为完整、可发布的 Markdown 小节。"
                        "必须只依据给定证据写作，保留学术综述语气，不得补造资料外事实。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"用户需求：\n{manifest.user_request}\n\n"
                        f"额外要求：\n{additional_requirements or '无'}\n\n"
                        f"全局记忆：\n{memory}\n\n"
                        f"章节标题：{node.title}\n"
                        f"章节目标：{section_goal}\n"
                        f"目标字数：{node.target_words}\n\n"
                        "请直接输出完整章节，并严格满足：\n"
                        f"1. 第一行必须是 `{heading_marks} {node.title}`。\n"
                        "2. 章节必须完整收束，最后一句不能截断。\n"
                        "3. 至少包含一处比较或取舍分析。\n"
                        "4. 至少包含一处适用场景、限制或边界条件说明。\n"
                        "5. 每一段都要落到具体事实，并在相关句后附引用编号。\n"
                        "6. 每个实质段落都必须至少带一个 [CH-xxxxxx] 引用；没有证据的细节要删掉。\n"
                        f"7. 全章至少使用 {min_unique_citations} 个不同的 [CH-xxxxxx] 引用编号。\n\n"
                        f"证据块：\n{evidence}"
                    ),
                },
            ],
            temperature=0.05,
            max_tokens=min(7000, max(2200, int(node.target_words * 2.5))),
        ).strip()

    def _force_citation_repair(
        self,
        manifest: ProjectManifest,
        *,
        node: OutlineNode,
        draft_text: str,
        evidence: str,
        repair_notes: str | None = None,
    ) -> str:
        paragraph_count = max(1, len(markdown_paragraphs(draft_text)))
        min_unique_citations = min_unique_citations_for_paragraphs(paragraph_count)
        return self.client.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "你是引用修复助手。"
                        "请把章节中的非 CH 引用改成可追溯的 [CH-xxxxxx] 编号，并保持原文事实与结构。"
                        "不得新增资料外内容，不得使用 [1]、[2] 之类的论文内部编号。"
                        "每个实质段落都必须至少保留一个可核验的 [CH-xxxxxx]。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"用户需求：\n{manifest.user_request}\n\n"
                        f"章节标题：{node.title}\n\n"
                        "请直接输出修复后的完整 Markdown 章节，并严格满足：\n"
                        "1. 只能使用证据块中出现的 [CH-xxxxxx] 编号。\n"
                        "2. 删除或替换所有 [1]、[2]、[9] 之类的数字引用。\n"
                        "3. 不改变标题层级，不删掉关键分析。\n"
                        "4. 每个实质段落都必须至少带一个 [CH-xxxxxx] 引用；如果某段没有证据，就删掉或改短，结尾总结段也不能例外。\n"
                        f"5. 全章至少使用 {min_unique_citations} 个不同的 [CH-xxxxxx] 编号。\n"
                        f"6. 额外修复要求：\n{repair_notes or '无'}\n\n"
                        f"待修复章节：\n{draft_text}\n\n"
                        f"证据块：\n{evidence}"
                    ),
                },
            ],
            temperature=0,
            max_tokens=min(7000, max(2200, int(len(draft_text) * 1.5))),
        ).strip()

    def _write_boundary_section(
        self,
        manifest: ProjectManifest,
        *,
        depth: int,
        node: OutlineNode,
        evidence: str,
        memory: str,
        additional_requirements: str | None,
        repair_notes: list[str],
    ) -> str:
        heading_marks = "#" * min(depth + 1, 4)
        return self.client.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "你是证据边界修复助手。"
                        "当现有一手证据不足以支撑章节中的某些具体机制、指标或强结论时，"
                        "你要把章节改写成保守但完整的综述版本。"
                        "只能保留证据正文直接支持的事实；若关键机制缺乏一手 PDF 证据，"
                        "必须明确写出'当前一手 PDF 的直接证据有限，本节仅保留框架性判断'。"
                        "不要硬写不被证据直接支持的 DPMM、Copula、Markov、SAIFI、扩容规划细节或 DRRL 成熟案例。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"用户需求：\n{manifest.user_request}\n\n"
                        f"额外要求：\n{additional_requirements or '无'}\n\n"
                        f"全局记忆：\n{memory}\n\n"
                        f"章节标题：{node.title}\n"
                        f"章节目标：{_safe_goal(node)}\n"
                        f"目标字数：{node.target_words}\n\n"
                        "请直接输出一个更保守的完整 Markdown 章节，并严格满足：\n"
                        f"1. 第一行必须是 `{heading_marks} {node.title}`。\n"
                        "2. 每个实质段落都必须至少带一个 [CH-xxxxxx] 引用，开头第一段也不能例外。\n"
                        "3. 可以比原目标字数更短，但必须完整收束，并保留综述结构。\n"
                        "4. 如果某个具体机制或指标缺乏一手 PDF 直接支持，就明确写成证据边界，不要继续给出细节性判断；即便是边界说明段，也要带引用。\n"
                        "5. 不要根据文件名、题名页或二手摘要扩写事实。\n"
                        "6. 禁止写'由此可推断其在规划层也有类似潜力'、'可合理外推到长期投资'之类的延伸判断。\n\n"
                        f"当前需要回避或降级的内容：\n" + "\n".join(f"- {item}" for item in repair_notes) + "\n\n"
                        f"证据块：\n{evidence}"
                    ),
                },
            ],
            temperature=0,
            max_tokens=min(5000, max(1600, int(node.target_words * 1.6))),
        ).strip()

    @staticmethod
    def _is_valid_section(text: str) -> bool:
        if not text.startswith("#") or not looks_complete(text):
            return False
        paragraphs = markdown_paragraphs(text)
        if not paragraphs:
            return False
        if uncited_substantive_paragraphs(text):
            return False
        citations = collect_citation_ids(text)
        if not citations:
            return False
        return len(citations) >= min_unique_citations_for_paragraphs(len(paragraphs))

    @staticmethod
    def _unsupported_support_issues(text: str, retrieved: list[RetrievalResult]) -> list[str]:
        support_map = {
            item.chunk.chunk_id: "\n".join(
                [item.chunk.file_name, item.chunk.section_title, item.chunk.text]
            )
            for item in retrieved
        }
        issues: list[str] = []
        for paragraph in markdown_paragraphs(text):
            if BOUNDARY_CUE_RE.search(paragraph):
                continue
            citations = collect_citation_ids(paragraph)
            if not citations:
                continue
            support_blob = "\n".join(support_map.get(citation_id, "") for citation_id in citations)
            if not support_blob:
                continue
            for label, paragraph_pattern, source_patterns in SECTION_SUPPORT_RULES:
                if paragraph_pattern.search(paragraph) and not any(pattern.search(support_blob) for pattern in source_patterns):
                    issues.append(f"{label}: {truncate(paragraph, 120)}")
                    break
        return issues

    def _is_acceptable_section(self, text: str, retrieved: list[RetrievalResult]) -> bool:
        return self._is_valid_section(text) and not self._unsupported_support_issues(text, retrieved)

    def generate(
        self,
        manifest: ProjectManifest,
        outline: list[OutlineNode],
        *,
        additional_requirements: str | None = None,
    ) -> list[SectionDraft]:
        memory = "尚无已生成章节。"
        glossary_text = "\n".join(f"- {item.term}: {item.definition}" for item in manifest.glossary[:20])
        full_outline = "\n".join(
            f"- {node.node_id} {node.title}: {_safe_goal(node)}（{node.target_words}字）"
            for node in flatten_outline(outline)
        )
        drafts: list[SectionDraft] = []

        for depth, node in _leaf_nodes(outline):
            logger.info(f"[{node.node_id}] 开始生成章节: {node.title}")
            heading_marks = "#" * min(depth + 1, 4)
            section_goal = _safe_goal(node)
            keyword_hint = ", ".join(_section_keywords(node))
            min_unique_citations = min_unique_citations_for_paragraphs(max(1, round(node.target_words / 220)))
            retrieval_top_k = 20 if node.node_id in {"S2.2", "S3.1", "S3.2", "S4.2"} else 12 if _is_frontier_rl_section(node) else 8
            query = (
                f"{manifest.user_request}\n章节标题：{node.title}\n章节目标：{section_goal}"
                + (f"\n英文检索锚词：{keyword_hint}" if keyword_hint else "")
            )
            retrieved = self._manual_retrieval(node)
            if not retrieved:
                retrieved = self.retriever.search(query, top_k=retrieval_top_k)
                retrieved = _filter_retrieved_results(node, retrieved)
            evidence = format_evidence(retrieved)
            logger.info(f"[{node.node_id}] 检索到 {len(retrieved)} 个证据块")
            section_guidance = _section_guidance(node)

            response = self.client.chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "你是资深中文综述作者。"
                            "你要基于检索证据写出可追溯、像人写的专业章节，而不是资料堆叠或模板作文。"
                            "判断、比较和边界意识与事实同样重要。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"用户需求：\n{manifest.user_request}\n\n"
                            f"额外要求：\n{additional_requirements or '无'}\n\n"
                            f"用户画像：\n{manifest.user_profile.model_dump_json(indent=2)}\n\n"
                            f"术语库：\n{glossary_text or '无'}\n\n"
                            f"完整大纲：\n{full_outline}\n\n"
                            f"已生成内容记忆：\n{memory}\n\n"
                            f"当前章节标题：{node.title}\n"
                            f"当前章节目标：{section_goal}\n"
                            f"目标字数：{node.target_words}\n\n"
                            "请严格遵守以下写作规则：\n"
                            f"1. 第一行必须输出 `{heading_marks} {node.title}`。\n"
                            "2. 直接写完整章节，不要解释你的写作过程。\n"
                            "3. 每段都要围绕明确论点展开，不要把多篇资料逐条拼接成流水账。\n"
                            "4. 尽量少用\"其核心在于、本质上、值得注意的是、综上所述、案例结果表明\"等套话。\n"
                            "5. 不要夸大证据强度，不要写'完全解决、显著优于一切、普遍适用'等绝对化表达。\n"
                            "6. 至少给出一处比较判断，说明不同方法、场景或求解框架之间的取舍。\n"
                            "7. 与应用有关的章节，要明确写出方法更适合什么场景、不适合什么场景，或代价在哪里。\n"
                            "8. 引用编号只能来自证据块，并紧跟在相应句后。\n"
                            "9. 严禁使用 [1]、[2]、[9] 这类原论文内部参考文献编号，必须使用 [CH-xxxxxx]。\n"
                            "10. 若证据块同时出现 PDF 原始论文与 DOCX 综述材料，优先让正文判断建立在 PDF 原始论文证据上，DOCX 只作补充。\n"
                            "11. 句子里若出现专名或专术语（如 DPMM、SAIFI、TV 距离、ADMM、LearnAMR），引用的证据必须直接出现该专名或同义表述。\n"
                            "12. 若证据不能直接支撑某个细节，就删去该细节，不要硬写。\n"
                            "13. 最后一段要自然收束，不能突然停止。\n"
                            "14. 每个实质段落（不少于 60 字）都必须至少包含一个 [CH-xxxxxx] 引用；没有直接证据的句子就删掉或降级成审慎边界说明。\n"
                            f"15. 全章至少使用 {min_unique_citations} 个不同的 [CH-xxxxxx] 引用编号，不要整章只反复依赖一两个编号。\n"
                            "16. 不要根据文件名、题名页或综述摘要臆断正文事实；若被引证据正文没有直接出现某个专名或结论，就不要写出来。\n"
                            "17. 如果证据块主要支持的是鲁棒优化（RO）而不是分布鲁棒优化（DRO），就必须如实写成 RO；最多只能说明其对 DRO/IES 的启示，不能偷换成 DRO。\n"
                            "18. 如果当前小节标题与证据范围不一致，可以在不改变标题层级的前提下小幅重命名，使标题服从证据。\n"
                            "19. 不要用'直接机制、成立前提、证据边界、与上下文的逻辑衔接'这类固定脚手架反复组织段落；要把判断自然融入综述叙述。\n"
                            "20. 本章不要重复已有章节的核心观点，如'经济性与鲁棒性平衡'、'计算复杂度高'、'数据有限'、'模糊集半径选择'等；如需提及，只需一句话带过，不要展开论述。\n\n"
                            f"当前章节额外提示：\n{section_guidance}\n\n"
                            f"证据块：\n{evidence}"
                        ),
                    },
                ],
                temperature=0.2,
                max_tokens=min(7000, max(2000, int(node.target_words * 2.3))),
            ).strip()

            polished = self.editor.polish_section(
                manifest,
                response,
                section_title=node.title,
                revision_focus=(
                    "减少 AI 腔和机械过渡，增强比较判断、场景适配、代价与边界说明。"
                    "如果段落只是顺排文献，请改成综合评述。"
                ),
            )

            # 大模型语义审查：检测跨章节内容重复
            logger.info(f"[{node.node_id}] 进行大模型语义审查...")
            review_result = self._model_review_section(polished, memory)

            if review_result.get("repeat") and review_result.get("severity") == "high":
                overlaps = review_result.get("overlaps", [])
                logger.info(f"[{node.node_id}] 检测到跨章节重复，进行针对性重写...")
                if overlaps:
                    overlap_summary = "\n".join(
                        f"- 段落'{o['current_paragraph']}'与章节'{o['overlaps_with']}'重复：{o['overlapping_content']}。修改建议：{o['suggested_fix']}"
                        for o in overlaps
                    )
                    targeted_rewrite_prompt = (
                        "重写此章节中的重复内容，要求：\n"
                        "1. 针对以下列出的重复段落进行针对性修改，不要全章重写\n"
                        "2. 换用新的角度、句式和论述方式，避免与已发布章节重复\n"
                        "3. 保持引用编号不变，只修改句式表达\n"
                        "4. 保持章节整体结构和字数基本不变\n\n"
                        f"原章节：\n{polished}\n\n"
                        f"需要修改的重复内容：\n{overlap_summary}"
                    )
                    try:
                        rewritten = self.client.chat(
                            [
                                {"role": "system", "content": "你是资深中文综述作者，擅长避免内容重复，专注于针对性修改而非全章重写。"},
                                {"role": "user", "content": targeted_rewrite_prompt},
                            ],
                            temperature=0.1,
                            max_tokens=min(7000, max(2000, int(node.target_words * 2.3))),
                        ).strip()
                        if rewritten.startswith("#"):
                            polished = rewritten
                    except Exception:
                        pass  # 保持原polished不变

            candidate = polished
            if not self._is_acceptable_section(candidate, retrieved):
                if self._is_acceptable_section(response, retrieved):
                    candidate = response
                for _ in range(3):
                    if self._is_acceptable_section(candidate, retrieved):
                        break
                    repaired = self._repair_section(
                        manifest,
                        depth=depth,
                        node=node,
                        evidence=evidence,
                        memory=memory,
                        additional_requirements=additional_requirements,
                    )
                    repaired_polished = self.editor.polish_section(
                        manifest,
                        repaired,
                        section_title=node.title,
                        revision_focus="优先修复截断、段落碎裂与生硬衔接，同时保留引用编号。",
                    )
                    if self._is_acceptable_section(repaired_polished, retrieved):
                        candidate = repaired_polished
                    elif self._is_acceptable_section(repaired, retrieved):
                        candidate = repaired

            if not self._is_acceptable_section(candidate, retrieved):
                support_issues = self._unsupported_support_issues(candidate, retrieved)
                uncited_issues = [truncate(item, 120) for item in uncited_substantive_paragraphs(candidate)]
                repair_notes = support_issues[:]
                if uncited_issues:
                    repair_notes.append("以下实质段落缺少引用，必须补齐 [CH-xxxxxx] 或删除：")
                    repair_notes.extend(uncited_issues)
                forced = self._force_citation_repair(
                    manifest,
                    node=node,
                    draft_text=candidate,
                    evidence=evidence,
                    repair_notes="\n".join(f"- {item}" for item in repair_notes) if repair_notes else None,
                )
                if self._is_acceptable_section(forced, retrieved):
                    candidate = forced

            if not self._is_acceptable_section(candidate, retrieved):
                support_issues = self._unsupported_support_issues(candidate, retrieved)
                boundary = self._write_boundary_section(
                    manifest,
                    depth=depth,
                    node=node,
                    evidence=evidence,
                    memory=memory,
                    additional_requirements=additional_requirements,
                    repair_notes=support_issues or ["仅保留被一手 PDF 直接支持的框架性判断。"],
                )
                if not self._is_acceptable_section(boundary, retrieved):
                    uncited_issues = [truncate(item, 120) for item in uncited_substantive_paragraphs(boundary)]
                    boundary_notes = ["优先给边界说明段补齐 [CH-xxxxxx] 引用，并删除仍然超出证据的细节。"]
                    if uncited_issues:
                        boundary_notes.append("以下实质段落缺少引用，必须补齐 [CH-xxxxxx] 或删除：")
                        boundary_notes.extend(uncited_issues)
                    boundary = self._force_citation_repair(
                        manifest,
                        node=node,
                        draft_text=boundary,
                        evidence=evidence,
                        repair_notes="\n".join(f"- {item}" for item in boundary_notes),
                    )
                if self._is_acceptable_section(boundary, retrieved):
                    candidate = boundary

            if not self._is_acceptable_section(candidate, retrieved):
                # Fallback: use raw response if it's not empty
                if response.strip() and self._is_valid_section(response):
                    candidate = response
                # If still not acceptable, use whatever we have as last resort
                if not self._is_acceptable_section(candidate, retrieved):
                    if candidate.strip():
                        # Log warning but continue instead of failing
                        import warnings
                        warnings.warn(f"Section {node.node_id} {node.title} has issues but continuing with generated content")
                    else:
                        raise ValueError(f"Failed to generate any content for {node.node_id} {node.title}")

            polished = candidate
            citations = collect_citation_ids(polished)
            drafts.append(
                SectionDraft(
                    node_id=node.node_id,
                    title=node.title,
                    level=depth + 1,
                    target_words=node.target_words,
                    content=polished,
                    evidence_ids=[item.chunk.chunk_id for item in retrieved],
                    citations=citations,
                )
            )

            logger.info(f"[{node.node_id}] 章节生成完成，字数约 {len(polished)} 字")
            memory = self.client.chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "你是写作记忆整理助手。"
                            "请把已生成章节压缩成后续写作可复用的全局记忆，避免重复论点。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"完整大纲：\n{full_outline}\n\n"
                            f"已有记忆：\n{memory}\n\n"
                            f"新增章节：\n{polished}\n\n"
                            "请输出 4 到 6 条简洁记忆，内容包括："
                            "已经确立的主线、已写过的比较判断、后文仍需展开的线索。"
                        ),
                    },
                ],
                temperature=0,
                max_tokens=700,
            ).strip()

        return drafts
