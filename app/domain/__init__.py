"""Contract-compatible domain models shared by kernel, plugins, and adapters."""

from app.domain.artifact import ArtifactLineage, ArtifactRef
from app.domain.compute import ComputeRun, ComputeRunStatus, WorkflowType
from app.domain.context import (
    DatasetVersionContext,
    PlatformJobContext,
    PlatformJobType,
    RiskProfile,
)
from app.domain.errors import ErrorBody, ErrorCode, ErrorResponse
from app.domain.evidence import (
    DuplicateSignals,
    EvidenceBundle,
    EvidenceRef,
    EvidenceSignals,
    LearningValueSignals,
    NormalizedSignal,
    ObjectAnalyticalPassport,
    ObjectDecisionBlock,
    ObjectIdentity,
    PredictionBlock,
    PrivacyBlock,
    SignalStatus,
    TechnicalQualityBlock,
)
from app.domain.manifest import (
    DataModality,
    DataSplit,
    ManifestLineage,
    ManifestRow,
    PredictionArtifactRef,
    PredictionManifest,
    PredictionRow,
)

__all__ = [
    "ArtifactLineage",
    "ArtifactRef",
    "ComputeRun",
    "ComputeRunStatus",
    "DatasetVersionContext",
    "DataModality",
    "DataSplit",
    "DuplicateSignals",
    "ErrorBody",
    "ErrorCode",
    "ErrorResponse",
    "EvidenceBundle",
    "EvidenceRef",
    "EvidenceSignals",
    "LearningValueSignals",
    "ManifestLineage",
    "ManifestRow",
    "NormalizedSignal",
    "ObjectAnalyticalPassport",
    "ObjectDecisionBlock",
    "ObjectIdentity",
    "PlatformJobContext",
    "PlatformJobType",
    "PredictionArtifactRef",
    "PredictionBlock",
    "PredictionManifest",
    "PredictionRow",
    "PrivacyBlock",
    "RiskProfile",
    "SignalStatus",
    "TechnicalQualityBlock",
    "WorkflowType",
]
