from __future__ import annotations

from fastapi import FastAPI

from report_writer.config import load_settings
from report_writer.models import (
    GenerateReportRequest,
    IngestFolderRequest,
    OutlineRequest,
    ReviseReportRequest,
    ReviewLoopRequest,
)
from report_writer.service import ReportWriterService


settings = load_settings()
service = ReportWriterService(settings)
app = FastAPI(title="AI Report Writer", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": settings.chat_model}


@app.on_event("shutdown")
def shutdown_event() -> None:
    service.close()


@app.post("/projects/ingest-folder")
def ingest_folder(payload: IngestFolderRequest):
    return service.ingest_folder(
        project_name=payload.project_name,
        source_dir=payload.source_dir,
        user_request=payload.user_request,
        target_words=payload.target_words,
    )


@app.post("/projects/{project_id}/outline")
def generate_outline(project_id: str, payload: OutlineRequest):
    return service.generate_outline(
        project_id,
        title_hint=payload.title_hint,
        additional_requirements=payload.additional_requirements,
    )


@app.post("/projects/{project_id}/generate")
def generate_report(project_id: str, payload: GenerateReportRequest):
    return service.generate_report(
        project_id,
        title_hint=payload.title_hint,
        additional_requirements=payload.additional_requirements,
        target_words=payload.target_words,
    )


@app.post("/projects/{project_id}/revise")
def revise_report(project_id: str, payload: ReviseReportRequest):
    return service.revise_report(
        project_id,
        payload.revision_request,
        report_id=payload.report_id,
    )


@app.get("/projects/{project_id}/review")
def review_report(project_id: str, report_id: str | None = None):
    return service.review_report(project_id, report_id)


@app.post("/projects/{project_id}/review-loop")
def review_loop(project_id: str, payload: ReviewLoopRequest):
    return service.review_until_pass(
        project_id,
        report_id=payload.report_id,
        title_hint=payload.title_hint,
        additional_requirements=payload.additional_requirements,
        target_words=payload.target_words,
        max_rounds=payload.max_rounds,
    )


@app.get("/projects/{project_id}")
def get_project(project_id: str):
    return service.get_project(project_id)
