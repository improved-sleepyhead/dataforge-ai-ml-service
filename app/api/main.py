"""FastAPI application entrypoint for the DataForge AI compute plane."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from starlette.middleware.base import RequestResponseEndpoint

from app.adapters import FakePlatformMetadataClient
from app.api.schemas import (
    AnalyzeDatasetAcceptedResponse,
    AnalyzeDatasetRequest,
    HealthResponse,
)
from app.api.security import PlatformIdentityDep, ServiceSignatureError
from app.domain import ComputeRunStatus, ErrorBody, ErrorCode, ErrorResponse
from app.kernel.config import ServiceConfig, load_config
from app.orchestration.analyze_workflow import launch_analyze_dataset_workflow
from app.plugin_sdk import CapabilitiesResponse
from app.plugins import build_static_plugin_manager
from app.validation.contracts import load_contract_pack

SERVICE_PACKAGE_NAME = "dataforgeai-ml-service"
API_PREFIX = "/api/v1"


def create_app(
    *,
    include_test_error_route: bool = False,
    include_test_protected_route: bool = False,
    config: ServiceConfig | None = None,
) -> FastAPI:
    """Create the FastAPI app without requiring production secrets."""
    application = FastAPI(
        title="DataForge AI ML Service",
        version=service_version(),
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )
    application.state.service_config = config
    application.state.fake_platform_client = FakePlatformMetadataClient()

    @application.middleware("http")
    async def safe_unhandled_error_middleware(
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        try:
            return await call_next(request)
        except Exception:  # noqa: BLE001 - boundary middleware must sanitize all unhandled errors
            return error_json_response(
                status_code=500,
                code=ErrorCode.PLUGIN_EXECUTION_FAILED,
                message="Internal compute service error.",
                recoverable=False,
                stage="api",
            )

    @application.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        return error_json_response(
            status_code=exc.status_code,
            code=_http_error_code(exc.status_code),
            message="Request could not be processed.",
            recoverable=exc.status_code < 500,
            stage="api",
            details={"http_status": exc.status_code},
        )

    @application.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        return error_json_response(
            status_code=422,
            code=ErrorCode.INVALID_JOB_PAYLOAD,
            message="Request validation failed.",
            recoverable=True,
            stage="api",
            details={"error_count": len(exc.errors())},
        )

    @application.exception_handler(ServiceSignatureError)
    async def service_signature_exception_handler(
        request: Request,
        exc: ServiceSignatureError,
    ) -> JSONResponse:
        return error_json_response(
            status_code=exc.status_code,
            code=exc.code,
            message="Service signature validation failed.",
            recoverable=True,
            stage="api.security",
            details={"reason_code": exc.reason_code},
        )

    @application.get(
        f"{API_PREFIX}/health",
        response_model=HealthResponse,
        responses={500: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
        tags=["system"],
    )
    async def health() -> HealthResponse:
        contract_pack = load_contract_pack()
        return HealthResponse(
            status="ok",
            service_version=service_version(),
            contract_pack_version=contract_pack.version,
        )

    @application.get(
        f"{API_PREFIX}/capabilities",
        response_model=CapabilitiesResponse,
        responses={500: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
        tags=["system"],
    )
    async def capabilities() -> CapabilitiesResponse:
        return build_static_plugin_manager().capabilities()

    @application.post(
        f"{API_PREFIX}/jobs/analyze-dataset",
        status_code=202,
        response_model=AnalyzeDatasetAcceptedResponse,
        responses={401: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
        tags=["jobs"],
    )
    async def analyze_dataset(
        payload: AnalyzeDatasetRequest,
        identity: PlatformIdentityDep,
        request: Request,
    ) -> AnalyzeDatasetAcceptedResponse:
        del identity
        result = launch_analyze_dataset_workflow(
            request=payload,
            config=_resolve_service_config(request),
            fake_platform=request.app.state.fake_platform_client,
        )
        return AnalyzeDatasetAcceptedResponse(
            status=ComputeRunStatus.ACCEPTED,
            job_id=result.job_id,
            status_url=result.status_url,
            expected_outputs=result.expected_outputs,
            materialized_assets=result.materialized_assets,
            mutates_dataset=False,
        )

    if include_test_error_route:
        _add_test_error_routes(application)
    if include_test_protected_route:
        _add_test_protected_routes(application)

    return application


def error_json_response(
    *,
    status_code: int,
    code: ErrorCode,
    message: str,
    recoverable: bool,
    stage: str,
    details: dict[str, object] | None = None,
) -> JSONResponse:
    """Build a safe ErrorResponse JSON body without raw exception details."""
    response = ErrorResponse(
        error=ErrorBody(
            code=code,
            message=message,
            recoverable=recoverable,
            stage=stage,
            details={} if details is None else details,
        )
    )
    return JSONResponse(status_code=status_code, content=response.model_dump(mode="json"))


def service_version() -> str:
    """Return installed package version with a local fallback for editable checkouts."""
    try:
        return version(SERVICE_PACKAGE_NAME)
    except PackageNotFoundError:
        return "0.1.0"


def _resolve_service_config(request: Request) -> ServiceConfig:
    config = getattr(request.app.state, "service_config", None)
    if isinstance(config, ServiceConfig):
        return config
    return load_config()


def _http_error_code(status_code: int) -> ErrorCode:
    if status_code == 404:
        return ErrorCode.ARTIFACT_NOT_FOUND
    if status_code in {400, 401, 403, 405, 409, 422}:
        return ErrorCode.INVALID_JOB_PAYLOAD
    return ErrorCode.PLUGIN_EXECUTION_FAILED


def _add_test_error_routes(application: FastAPI) -> None:
    @application.get("/__test__/unhandled-error", include_in_schema=False)
    async def unhandled_error_fixture() -> None:
        raise RuntimeError("raw secret token and demo@example.com must not leak")

    @application.get("/__test__/validation-error", include_in_schema=False)
    async def validation_error_fixture(
        limit: Annotated[int, Query(ge=1)],
    ) -> dict[str, int]:
        return {"limit": limit}


def _add_test_protected_routes(application: FastAPI) -> None:
    @application.post("/__test__/protected-compute-request", include_in_schema=False)
    async def protected_compute_request_fixture(
        identity: PlatformIdentityDep,
    ) -> dict[str, str]:
        return {
            "status": "accepted",
            "service_identity": identity.service_identity,
            "organization_id": identity.organization_id,
            "project_id": identity.project_id,
        }


app = create_app()
