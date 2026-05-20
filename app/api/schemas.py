"""API response schemas."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from app.domain.common import NonEmptyStr


class HealthResponse(BaseModel):
    """Health response for service and contract compatibility checks."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: NonEmptyStr
    service_version: NonEmptyStr
    contract_pack_version: NonEmptyStr
