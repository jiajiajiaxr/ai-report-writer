from __future__ import annotations

from pathlib import Path

import numpy as np

from .models import ChunkRecord, DocumentMetadata, DocumentRecord, ProjectManifest, ProjectSummary, ReportArtifact
from .utils import dump_json, load_json


class ProjectStorage:
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.projects_dir = root_dir / "projects"
        self.projects_dir.mkdir(parents=True, exist_ok=True)

    def project_dir(self, project_id: str) -> Path:
        path = self.projects_dir / project_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _path(self, project_id: str, name: str) -> Path:
        return self.project_dir(project_id) / name

    def save_manifest(self, manifest: ProjectManifest) -> None:
        dump_json(manifest.model_dump(mode="json"), self._path(manifest.project_id, "manifest.json"))

    def load_manifest(self, project_id: str) -> ProjectManifest:
        return ProjectManifest.model_validate(load_json(self._path(project_id, "manifest.json")))

    def save_documents(self, project_id: str, documents: list[DocumentRecord]) -> None:
        dump_json([doc.model_dump(mode="json") for doc in documents], self._path(project_id, "documents.json"))

    def load_documents(self, project_id: str) -> list[DocumentRecord]:
        raw = load_json(self._path(project_id, "documents.json"))
        return [DocumentRecord.model_validate(item) for item in raw]

    def load_document_metadata(self, project_id: str) -> list[DocumentMetadata]:
        return [doc.metadata for doc in self.load_documents(project_id)]

    def save_chunks(self, project_id: str, chunks: list[ChunkRecord]) -> None:
        dump_json([chunk.model_dump(mode="json") for chunk in chunks], self._path(project_id, "chunks.json"))

    def load_chunks(self, project_id: str) -> list[ChunkRecord]:
        raw = load_json(self._path(project_id, "chunks.json"))
        return [ChunkRecord.model_validate(item) for item in raw]

    def save_embeddings(self, project_id: str, matrix: np.ndarray) -> None:
        target = self._path(project_id, "embeddings.npy")
        target.parent.mkdir(parents=True, exist_ok=True)
        np.save(target, matrix)

    def load_embeddings(self, project_id: str) -> np.ndarray:
        return np.load(self._path(project_id, "embeddings.npy"))

    def save_artifact(self, project_id: str, artifact: ReportArtifact) -> None:
        artifacts = self.load_artifacts(project_id)
        artifacts = [item for item in artifacts if item.report_id != artifact.report_id]
        artifacts.append(artifact)
        artifacts.sort(key=lambda item: item.created_at)
        dump_json([item.model_dump(mode="json") for item in artifacts], self._path(project_id, "artifacts.json"))

    def load_artifacts(self, project_id: str) -> list[ReportArtifact]:
        path = self._path(project_id, "artifacts.json")
        if not path.exists():
            return []
        raw = load_json(path)
        artifacts = [ReportArtifact.model_validate(item) for item in raw]
        return sorted(artifacts, key=lambda item: item.created_at)

    def load_summary(self, project_id: str) -> ProjectSummary:
        return ProjectSummary(
            manifest=self.load_manifest(project_id),
            documents=self.load_document_metadata(project_id),
            artifacts=self.load_artifacts(project_id),
        )
