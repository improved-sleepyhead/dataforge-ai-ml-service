"""Pytest compatibility shims for the local FastAPI boundary tests."""

from __future__ import annotations

import asyncio
from types import TracebackType
from typing import Any

import fastapi.testclient
import httpx


class StableASGITestClient:
    """Small synchronous ASGI client used to avoid TestClient portal hangs.

    The project tests only need the usual ``get`` / ``post`` request helpers.
    Using ``httpx.AsyncClient`` with ``ASGITransport`` avoids the Starlette
    blocking-portal path that currently hangs in this Python 3.13 environment.
    """

    __test__ = False

    def __init__(
        self,
        app: Any,
        *,
        base_url: str = "http://testserver",
        raise_server_exceptions: bool = True,
        **_: Any,
    ) -> None:
        self._app = app
        self._base_url = base_url
        self._raise_server_exceptions = raise_server_exceptions

    def __enter__(self) -> StableASGITestClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None

    def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        async def _send() -> httpx.Response:
            transport = httpx.ASGITransport(
                app=self._app,
                raise_app_exceptions=self._raise_server_exceptions,
            )
            async with httpx.AsyncClient(
                transport=transport,
                base_url=self._base_url,
            ) as client:
                return await client.request(method, url, **kwargs)

        return asyncio.run(_send())

    def get(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("POST", url, **kwargs)

    def put(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("PUT", url, **kwargs)

    def delete(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("DELETE", url, **kwargs)

    def patch(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("PATCH", url, **kwargs)


fastapi.testclient.TestClient = StableASGITestClient  # type: ignore[misc, assignment]
