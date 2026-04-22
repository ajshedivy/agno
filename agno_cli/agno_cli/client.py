"""Standalone HTTP client for AgentOS API.

Uses httpx directly — no imports from the agno library.
"""

import json
import sys
from typing import Any, Dict, Iterator, Optional, Tuple

import httpx


class AgnoClientError(Exception):
    """Base exception for AgnoClient errors."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class AgnoClient:
    """Synchronous HTTP client for AgentOS API."""

    def __init__(self, base_url: str, timeout: float = 60.0, security_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.headers: Dict[str, str] = {}
        if security_key:
            self.headers["Authorization"] = f"Bearer {security_key}"
        self._client = httpx.Client(timeout=self.timeout, headers=self.headers)

    def _handle_error(self, e: Exception) -> None:
        """Convert httpx exceptions to user-friendly errors."""
        if isinstance(e, (httpx.ConnectError, httpx.ConnectTimeout)):
            raise AgnoClientError(f"Cannot connect to AgentOS at {self.base_url}. Is the server running?")
        if isinstance(e, httpx.TimeoutException):
            raise AgnoClientError(f"Request timed out after {self.timeout}s.")
        raise AgnoClientError(str(e))

    def _handle_response(self, response: httpx.Response) -> Any:
        """Check response status and return parsed JSON or None."""
        if response.status_code >= 400:
            detail = ""
            try:
                body = response.json()
                detail = body.get("detail", json.dumps(body))
            except Exception:
                detail = response.text or f"HTTP {response.status_code}"
            if response.status_code in (401, 403):
                raise AgnoClientError(
                    f"Authentication failed ({response.status_code}): {detail}. "
                    "Check your security key with 'agno-os config show'.",
                    status_code=response.status_code,
                )
            if response.status_code == 404:
                raise AgnoClientError(f"Not found: {detail}", status_code=404)
            raise AgnoClientError(f"HTTP {response.status_code}: {detail}", status_code=response.status_code)
        if not response.content:
            return None
        return response.json()

    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Execute GET request."""
        try:
            response = self._client.get(f"{self.base_url}{endpoint}", params=_clean_params(params))
            return self._handle_response(response)
        except AgnoClientError:
            raise
        except Exception as e:
            self._handle_error(e)

    def post(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        as_form: bool = False,
    ) -> Any:
        """Execute POST request."""
        try:
            kwargs: Dict[str, Any] = {"params": _clean_params(params)}
            if data is not None:
                if as_form:
                    kwargs["data"] = data
                else:
                    kwargs["json"] = data
            response = self._client.post(f"{self.base_url}{endpoint}", **kwargs)
            return self._handle_response(response)
        except AgnoClientError:
            raise
        except Exception as e:
            self._handle_error(e)

    def patch(
        self, endpoint: str, data: Optional[Dict[str, Any]] = None, params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Execute PATCH request."""
        try:
            kwargs: Dict[str, Any] = {"params": _clean_params(params)}
            if data is not None:
                kwargs["json"] = data
            response = self._client.patch(f"{self.base_url}{endpoint}", **kwargs)
            return self._handle_response(response)
        except AgnoClientError:
            raise
        except Exception as e:
            self._handle_error(e)

    def delete(
        self, endpoint: str, data: Optional[Dict[str, Any]] = None, params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Execute DELETE request."""
        try:
            kwargs: Dict[str, Any] = {"params": _clean_params(params)}
            if data is not None:
                kwargs["json"] = data
            response = self._client.delete(f"{self.base_url}{endpoint}", **kwargs)
            return self._handle_response(response)
        except AgnoClientError:
            raise
        except Exception as e:
            self._handle_error(e)

    def post_multipart(
        self,
        endpoint: str,
        files: Dict[str, Tuple[str, Any, str]],
        data: Optional[Dict[str, str]] = None,
    ) -> Any:
        """Execute multipart POST request for file uploads."""
        try:
            response = self._client.post(f"{self.base_url}{endpoint}", files=files, data=data or {})
            return self._handle_response(response)
        except AgnoClientError:
            raise
        except Exception as e:
            self._handle_error(e)

    def stream_post(
        self,
        endpoint: str,
        data: Dict[str, Any],
        as_form: bool = True,
    ) -> Iterator[str]:
        """Stream POST request, yielding SSE data lines."""
        try:
            kwargs: Dict[str, Any] = {}
            if as_form:
                kwargs["data"] = data
            else:
                kwargs["json"] = data
            with self._client.stream("POST", f"{self.base_url}{endpoint}", **kwargs) as response:
                if response.status_code >= 400:
                    response.read()
                    self._handle_response(response)
                for line in response.iter_lines():
                    yield line
        except AgnoClientError:
            raise
        except Exception as e:
            self._handle_error(e)

    def close(self) -> None:
        """Close the underlying httpx client."""
        self._client.close()


def parse_sse_events(lines: Iterator[str]) -> Iterator[Dict[str, Any]]:
    """Parse SSE stream lines into event dicts."""
    for line in lines:
        if not line or line.startswith(":"):
            continue
        if line.startswith("data: "):
            try:
                yield json.loads(line[6:])
            except json.JSONDecodeError:
                print(f"Warning: failed to parse SSE event: {line[:100]}", file=sys.stderr)
                continue


def _clean_params(params: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Remove None values from query params."""
    if params is None:
        return None
    return {k: v for k, v in params.items() if v is not None}
