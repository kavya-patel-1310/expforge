"""Pydantic request/response schemas for the REST API."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ORM(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class ExperimentCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = ""
    tags: dict[str, Any] = {}


class ExperimentOut(ORM):
    id: int
    name: str
    description: str
    tags: dict[str, Any]
    created_at: datetime
    run_count: int = 0
    best: dict[str, Any] | None = None


class RunCreate(BaseModel):
    experiment: str = Field(description="Experiment name; created if it does not exist")
    name: str | None = None
    params: dict[str, Any] = {}
    tags: dict[str, Any] = {}
    system_info: dict[str, Any] = {}
    sweep_id: int | None = None
    notes: str = ""


class RunUpdate(BaseModel):
    status: Literal["running", "finished", "failed", "killed"] | None = None
    params: dict[str, Any] | None = None
    tags: dict[str, Any] | None = None
    summary: dict[str, Any] | None = None
    evaluation: dict[str, Any] | None = None
    notes: str | None = None
    error: str | None = None


class MetricPoint(BaseModel):
    key: str
    value: float
    step: int = 0


class MetricBatch(BaseModel):
    metrics: list[MetricPoint]


class ArtifactOut(ORM):
    id: int
    run_id: int
    name: str
    kind: str
    size_bytes: int
    sha256: str
    created_at: datetime


class RunOut(ORM):
    id: int
    experiment_id: int
    experiment_name: str = ""
    sweep_id: int | None
    name: str
    status: str
    params: dict[str, Any]
    tags: dict[str, Any]
    summary: dict[str, Any]
    system_info: dict[str, Any]
    notes: str
    error: str
    started_at: datetime
    ended_at: datetime | None
    duration_s: float | None = None


class RunDetail(RunOut):
    evaluation: dict[str, Any]
    artifacts: list[ArtifactOut] = []
    metric_keys: list[str] = []


class CompareRequest(BaseModel):
    run_ids: list[int] = Field(min_length=2, max_length=20)


class SweepOut(ORM):
    id: int
    experiment_id: int
    name: str
    strategy: str
    objective: str
    direction: str
    search_space: dict[str, Any]
    status: str
    best_run_id: int | None
    created_at: datetime


class RegisterModelRequest(BaseModel):
    name: str
    run_id: int
    artifact_name: str = "model.joblib"
    description: str = ""


class StageUpdate(BaseModel):
    stage: Literal["None", "Staging", "Production", "Archived"]
    archive_existing: bool = True


class ModelVersionOut(ORM):
    id: int
    version: int
    run_id: int
    artifact_id: int
    stage: str
    metrics_snapshot: dict[str, Any]
    created_at: datetime


class RegisteredModelOut(ORM):
    id: int
    name: str
    description: str
    created_at: datetime
    versions: list[ModelVersionOut] = []


class PredictRequest(BaseModel):
    instances: list[list[float]] = Field(min_length=1)
    version: int | None = Field(default=None, description="Defaults to the Production version")
    return_proba: bool = False


class JobCreate(BaseModel):
    config: dict[str, Any]


class JobOut(ORM):
    id: int
    config: dict[str, Any]
    status: str
    run_ids: list[int]
    sweep_id: int | None
    error: str
    created_at: datetime
    finished_at: datetime | None
