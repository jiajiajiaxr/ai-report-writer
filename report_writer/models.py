from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    doc_id: str
    file_name: str
    source_path: str
    file_type: str
    title: str
    checksum: str
    char_count: int
    page_count: int = 0
    tags: list[str] = Field(default_factory=list)
    created_at: datetime


class DocumentRecord(BaseModel):
    metadata: DocumentMetadata
    text: str
    page_texts: list[str] = Field(default_factory=list)


class ChunkRecord(BaseModel):
    chunk_id: str
    doc_id: str
    chunk_type: str
    text: str
    token_estimate: int
    page_start: int = 1
    page_end: int = 1
    section_title: str = ""
    position: int = 0
    file_name: str = ""


class GlossaryTerm(BaseModel):
    term: str
    definition: str
    aliases: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


class UserProfile(BaseModel):
    domain: str = "通用专业报告"
    objectives: list[str] = Field(default_factory=list)
    focus_topics: list[str] = Field(default_factory=list)
    writing_preferences: list[str] = Field(default_factory=list)
    inferred_from: list[str] = Field(default_factory=list)


class OutlineNode(BaseModel):
    node_id: str
    title: str
    goal: str
    target_words: int
    evidence_ids: list[str] = Field(default_factory=list)
    children: list["OutlineNode"] = Field(default_factory=list)


OutlineNode.model_rebuild()


class SectionDraft(BaseModel):
    node_id: str
    title: str
    level: int
    target_words: int
    content: str
    evidence_ids: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)


class ValidationIssue(BaseModel):
    dimension: str
    severity: str
    message: str
    related_ids: list[str] = Field(default_factory=list)


class ValidationReport(BaseModel):
    passed: bool
    summary: str
    issues: list[ValidationIssue] = Field(default_factory=list)


class ReviewReport(BaseModel):
    verdict: str
    summary: str
    blocking_issues: list[str] = Field(default_factory=list)
    minor_issues: list[str] = Field(default_factory=list)
    recommended_fixes: list[str] = Field(default_factory=list)
    files_reviewed: list[str] = Field(default_factory=list)


class TraceEntry(BaseModel):
    citation_id: str
    doc_id: str
    file_name: str
    page_start: int
    page_end: int
    excerpt: str


class ReportArtifact(BaseModel):
    report_id: str
    project_id: str
    title: str
    markdown_path: str
    html_path: str
    docx_path: str
    trace_path: str
    review_path: str | None = None
    validation: ValidationReport
    created_at: datetime


class ProjectManifest(BaseModel):
    project_id: str
    name: str
    source_dir: str
    user_request: str
    target_words: int
    embedding_backend: str = "siliconflow"
    created_at: datetime
    user_profile: UserProfile
    glossary: list[GlossaryTerm] = Field(default_factory=list)
    document_ids: list[str] = Field(default_factory=list)
    chunk_count: int = 0
    outline: list[OutlineNode] = Field(default_factory=list)
    latest_report_id: str | None = None


class IngestFolderRequest(BaseModel):
    project_name: str
    source_dir: str
    user_request: str
    target_words: int = 12000


class OutlineRequest(BaseModel):
    title_hint: str | None = None
    additional_requirements: str | None = None


class GenerateReportRequest(BaseModel):
    title_hint: str | None = None
    additional_requirements: str | None = None
    target_words: int | None = None


class ReviseReportRequest(BaseModel):
    report_id: str | None = None
    revision_request: str


class ReviewLoopRequest(BaseModel):
    report_id: str | None = None
    title_hint: str | None = None
    additional_requirements: str | None = None
    target_words: int | None = None
    max_rounds: int = 3


class ProjectSummary(BaseModel):
    manifest: ProjectManifest
    documents: list[DocumentMetadata] = Field(default_factory=list)
    artifacts: list[ReportArtifact] = Field(default_factory=list)


class LlmReviewResult(BaseModel):
    summary: str
    issues: list[dict[str, Any]] = Field(default_factory=list)


class ReviewLoopResult(BaseModel):
    project_id: str
    final_report_id: str
    rounds: int
    passed: bool
    review: ReviewReport
    artifact: ReportArtifact
