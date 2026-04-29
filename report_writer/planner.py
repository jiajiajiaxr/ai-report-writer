from __future__ import annotations

from .models import OutlineNode, ProjectManifest
from .rag import RetrievalResult, format_evidence
from .siliconflow import SiliconFlowClient


def _normalize_outline(items: list[dict], prefix: str = "S") -> list[OutlineNode]:
    nodes: list[OutlineNode] = []
    for index, item in enumerate(items, start=1):
        node_id = f"{prefix}{index}"
        children = _normalize_outline(item.get("children", []), prefix=f"{node_id}.")
        nodes.append(
            OutlineNode(
                node_id=node_id,
                title=item["title"],
                goal=item.get("goal", ""),
                target_words=int(item.get("target_words", 800)),
                evidence_ids=item.get("evidence_ids", []),
                children=children,
            )
        )
    return nodes


class OutlinePlanner:
    def __init__(self, client: SiliconFlowClient) -> None:
        self.client = client

    def plan(
        self,
        manifest: ProjectManifest,
        evidence: list[RetrievalResult],
        *,
        title_hint: str | None = None,
        additional_requirements: str | None = None,
    ) -> list[OutlineNode]:
        glossary_text = "\n".join(f"- {item.term}: {item.definition}" for item in manifest.glossary[:15])
        evidence_text = format_evidence(evidence[:8])
        schema = """[
  {
    "title": "string",
    "goal": "string",
    "target_words": 1200,
    "evidence_ids": ["CH-000001"],
    "children": [
      {
        "title": "string",
        "goal": "string",
        "target_words": 800,
        "evidence_ids": ["CH-000002"],
        "children": []
      }
    ]
  }
]"""
        raw = self.client.chat_json(
            system_prompt=(
                "你是专业报告的总规划器。"
                "要根据资料证据先锁定全局逻辑，再生成多级大纲。"
                "不能超出已给资料。"
            ),
            user_prompt=(
                f"用户需求:\n{manifest.user_request}\n\n"
                f"标题提示:\n{title_hint or manifest.name}\n\n"
                f"附加要求:\n{additional_requirements or '无'}\n\n"
                f"用户画像:\n{manifest.user_profile.model_dump_json(indent=2)}\n\n"
                f"术语库:\n{glossary_text}\n\n"
                f"关键证据:\n{evidence_text}\n\n"
                f"目标总字数约 {manifest.target_words} 字。请输出多级大纲。"
            ),
            schema_hint=schema,
            max_tokens=8000,
        )
        return _normalize_outline(raw)
