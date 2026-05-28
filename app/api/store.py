"""In-memory store of recent compute results for the test/dev read API.

The DataForge platform owns the source of truth for users, dataset registry,
job records and audit. The Python compute plane is a worker. To make the
``GET`` test endpoints required by TASK-058 hermetic and side-effect-free,
this module keeps a small in-memory snapshot of the last result of each
``POST`` (analyze/preview/execute) plus an explicit slot for Version
Compare reports populated by tests or by the apply workflow when it ships
a compare artifact.

The store never holds raw PII or raw row payloads — only contract-shaped
artifacts and their stable identifiers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock

from app.domain import (
    ActionPlan,
    ComputeRunStatus,
    DataForgeReport,
    DecisionReport,
    ReviewQueue,
    VersionCompareReport,
    WorkflowType,
)


@dataclass(frozen=True)
class JobResultRecord:
    """Recorded result of an analyze or apply job execution."""

    job_id: str
    workflow_type: WorkflowType
    status: ComputeRunStatus
    status_url: str
    expected_outputs: tuple[str, ...]
    materialized_assets: tuple[str, ...]
    idempotency_key: str
    mutates_dataset: bool
    action_plan_id: str | None = None
    action_plan_hash: str | None = None
    candidate_artifact_uri: str | None = None
    candidate_artifact_hash: str | None = None
    synthetic_artifact_uri: str | None = None
    synthetic_status: str | None = None
    model_impact_artifact_uri: str | None = None
    export_package_artifact_uri: str | None = None


@dataclass(frozen=True)
class ActionPlanRecord:
    """Recorded ActionPlan with its execution metadata, if any."""

    action_plan: ActionPlan
    job_id: str | None = None
    action_plan_hash: str | None = None
    accepted_step_ids: tuple[str, ...] = ()
    workflow_type: WorkflowType | None = None
    status_url: str | None = None


@dataclass
class ComputeResultStore:
    """In-memory store of recent compute results.

    The store is thread-safe through an :class:`RLock` so the FastAPI
    handlers can mutate it under uvicorn's threaded worker.
    """

    _lock: RLock = field(default_factory=RLock)
    _jobs: dict[str, JobResultRecord] = field(default_factory=dict)
    _action_plans: dict[str, ActionPlanRecord] = field(default_factory=dict)
    _dataforge_reports: dict[str, DataForgeReport] = field(default_factory=dict)
    _decision_reports: dict[str, DecisionReport] = field(default_factory=dict)
    _review_queues: dict[str, tuple[ReviewQueue, ...]] = field(default_factory=dict)
    _version_compare: dict[str, VersionCompareReport] = field(default_factory=dict)

    def record_job(self, record: JobResultRecord) -> None:
        with self._lock:
            self._jobs[record.job_id] = record

    def get_job(self, job_id: str) -> JobResultRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def record_action_plan(self, record: ActionPlanRecord) -> None:
        with self._lock:
            self._action_plans[record.action_plan.action_plan_id] = record

    def get_action_plan(self, action_plan_id: str) -> ActionPlanRecord | None:
        with self._lock:
            return self._action_plans.get(action_plan_id)

    def record_dataforge_report(self, report: DataForgeReport) -> None:
        with self._lock:
            self._dataforge_reports[report.report_id] = report

    def get_dataforge_report(self, report_id: str) -> DataForgeReport | None:
        with self._lock:
            return self._dataforge_reports.get(report_id)

    def record_decision_report(self, report: DecisionReport) -> None:
        with self._lock:
            self._decision_reports[report.decision_report_id] = report

    def get_decision_report(self, report_id: str) -> DecisionReport | None:
        with self._lock:
            return self._decision_reports.get(report_id)

    def record_review_queues(
        self, *, report_id: str, queues: tuple[ReviewQueue, ...]
    ) -> None:
        with self._lock:
            self._review_queues[report_id] = queues

    def get_review_queues(self, report_id: str) -> tuple[ReviewQueue, ...] | None:
        with self._lock:
            return self._review_queues.get(report_id)

    def record_version_compare(self, report: VersionCompareReport) -> None:
        with self._lock:
            self._version_compare[report.report_id] = report

    def get_version_compare(self, compare_id: str) -> VersionCompareReport | None:
        with self._lock:
            return self._version_compare.get(compare_id)


__all__ = [
    "ActionPlanRecord",
    "ComputeResultStore",
    "JobResultRecord",
]
