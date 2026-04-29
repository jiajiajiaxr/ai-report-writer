from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _read_key_file(path: Path) -> str | None:
    if not path.exists():
        return None
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return None
    lines = [line.strip() for line in raw.splitlines() if line.strip() and not line.strip().startswith("#")]
    if not lines:
        return None
    first = lines[0]
    if "=" in first:
        _, value = first.split("=", 1)
        return value.strip().strip("'\"")
    return first.strip().strip("'\"")


@dataclass(slots=True)
class Settings:
    workspace_dir: Path
    data_dir: Path
    api_key: str
    base_url: str = "https://api.siliconflow.cn/v1"
    chat_model: str = "deepseek-ai/DeepSeek-V3.2"
    embedding_model: str = "BAAI/bge-m3"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    request_timeout: int = 300
    default_target_words: int = 12000
    max_retrieval_chunks: int = 10
    embedding_batch_size: int = 6


def load_settings(workspace_dir: str | Path | None = None) -> Settings:
    workspace = Path(workspace_dir or Path.cwd()).resolve()
    # Support Anthropic (Claude), SiliconFlow, and MiniMax API keys
    env_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("SILICONFLOW_API_KEY") or os.getenv("SILICONFLOW_KEY")
    file_key = _read_key_file(workspace / "key.env")
    api_key = (env_key or file_key or "").strip()
    if not api_key:
        raise RuntimeError("Missing API key. Set ANTHROPIC_API_KEY or SILICONFLOW_API_KEY or provide key.env.")
    data_dir = workspace / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Detect provider based on key prefix, env var, or base URL
    minimax_base_url = os.getenv("MINIMAX_BASE_URL", "").strip()
    minimax_key = os.getenv("MINIMAX_API_KEY") or os.getenv("MINIMAX_KEY")
    is_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    # MiniMax uses sk-cp- prefix, also detect via base URL or explicit env var
    is_minimax = (
        bool(minimax_key)
        or api_key.startswith("sk-cp-")
        or bool(minimax_base_url)
    )

    if is_minimax:
        base_url = minimax_base_url or "https://api.minimaxi.com/anthropic/v1"
        chat_model = os.getenv("MINIMAX_CHAT_MODEL", "MiniMax-M2.7")
    elif is_anthropic:
        base_url = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1").rstrip("/")
        chat_model = os.getenv("ANTHROPIC_CHAT_MODEL", "claude-sonnet-4-20250514")
    else:
        base_url = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1").rstrip("/")
        chat_model = os.getenv("SILICONFLOW_CHAT_MODEL", "deepseek-ai/DeepSeek-V3.2")

    return Settings(
        workspace_dir=workspace,
        data_dir=data_dir,
        api_key=api_key,
        base_url=base_url,
        chat_model=chat_model,
        embedding_model="BAAI/bge-m3",
        rerank_model="BAAI/bge-reranker-v2-m3",
        request_timeout=int(os.getenv("SILICONFLOW_TIMEOUT", "300")),
        default_target_words=int(os.getenv("REPORT_TARGET_WORDS", "12000")),
        max_retrieval_chunks=int(os.getenv("REPORT_TOPK", "10")),
        embedding_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "6")),
    )
