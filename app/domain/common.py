"""Shared domain value constraints."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

NonEmptyStr = Annotated[str, Field(min_length=1)]
Sha256Digest = Annotated[str, Field(pattern=r"^sha256:[a-f0-9]{64}$")]
S3Uri = Annotated[str, Field(pattern=r"^s3://[^\s]+$")]
