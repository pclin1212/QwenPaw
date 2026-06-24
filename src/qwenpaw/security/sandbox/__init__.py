# -*- coding: utf-8 -*-
"""QwenPaw sandbox subsystem -- Pure HTTP Client.

Architecture (v3, minimal-config):
    Sandbox container is an EXTERNAL service launched independently (see
    docker/sandbox-standalone/). QwenPaw is a pure HTTP client -- it does
    NOT manage container lifecycle.

User-facing config (security.sandbox):
    * enabled  -- master switch
    * endpoint -- where to connect (default http://localhost:8765)

Everything else (timeouts, host-bound tool list, workspace mapping) is
intentionally NOT in config:
    * Timeouts live as module constants below. Tune in source if you have
      a real reason; 99% of users never need to.
    * HOST_BOUND_TOOLS is a code constant in proxy_factory.py -- it is a
      property of the tool implementation (touches UI / host filesystem),
      not a per-deployment knob.
    * workspace_root is derived from the AgentRunner's working dir.

When enabled=True:
    build_sandbox_client() does a fast /health probe. On success it
    returns (SandboxClient, PathTranslator); the proxy layer rewrites
    every non-host-bound tool into an HTTP proxy.

    On failure it raises SandboxUnavailableError. NO silent fallback --
    if the user asked for a sandbox, they get one or they get a clear
    error.

Public API:
    build_sandbox_client(cfg, workspace_root) -> (SandboxClient, PathTranslator) | None
    SandboxClient                              -- shim: .transport / .started / .container_workspace / .stop()
    SandboxUnavailableError                    -- enabled=True but endpoint dead
    HttpTransport / PathTranslator / proxify_tool_dict / HOST_BOUND_TOOLS
"""
from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

from .transport import HttpTransport, ToolTransport
from .proxy_factory import (
    HOST_BOUND_TOOLS,
    PathTranslator,
    make_sandbox_proxy,
    proxify_tool_dict,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level timeouts. Sandbox is expected to be already running and on a
# fast local network (same host or same docker bridge), so values are tight.
# ---------------------------------------------------------------------------
HTTP_CONNECT_TIMEOUT_SEC: float = 5.0
HTTP_CALL_TIMEOUT_SEC: float = 120.0
HEALTH_PROBE_TIMEOUT_SEC: float = 10.0
HEALTH_PROBE_INTERVAL_SEC: float = 0.5


__all__ = [
    "build_sandbox_client",
    "SandboxClient",
    "SandboxUnavailableError",
    "HttpTransport",
    "ToolTransport",
    "HOST_BOUND_TOOLS",
    "PathTranslator",
    "make_sandbox_proxy",
    "proxify_tool_dict",
    # exposed for tests / power users that want to override
    "HTTP_CONNECT_TIMEOUT_SEC",
    "HTTP_CALL_TIMEOUT_SEC",
    "HEALTH_PROBE_TIMEOUT_SEC",
]


# ---------------------------------------------------------------------------
# Logging fallback: ensure sandbox INFO logs are visible even when this module
# is imported outside the regular CLI entrypoint. CLI's setup_logger() wins
# when present (we never overwrite an already-configured `qwenpaw` logger).
# ---------------------------------------------------------------------------
_root = logging.getLogger("qwenpaw")
if not _root.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"),
    )
    _root.addHandler(_h)
    _root.setLevel(logging.INFO)


class SandboxUnavailableError(RuntimeError):
    """Raised when sandbox.enabled=True but the endpoint is unreachable."""


class SandboxClient:
    """Thin shim pairing an HttpTransport with the attributes consumed by
    `react_agent` and `proxy_factory` (.transport / .started / .container_workspace).

    Replaces the old DockerSandboxExecutor. No container lifecycle here:
    `stop()` only closes the HTTP client; the sandbox container is owned
    by docker compose on the operator side.
    """

    # Hard-coded -- matches docker/sandbox/server.py mount point. Not a
    # config knob because the sandbox image fixes this path at build time.
    DEFAULT_CONTAINER_WORKSPACE = "/workspace"

    def __init__(
        self,
        transport: HttpTransport,
        container_workspace: str = DEFAULT_CONTAINER_WORKSPACE,
    ) -> None:
        self._transport = transport
        self._container_workspace = container_workspace
        self._started = True  # True once readiness probe has passed

    @property
    def transport(self) -> HttpTransport:
        return self._transport

    @property
    def started(self) -> bool:
        return self._started

    @property
    def container_workspace(self) -> str:
        return self._container_workspace

    async def stop(self) -> None:
        """Close the HTTP transport. The container itself keeps running --
        it is owned by an external docker compose stack."""
        if self._started:
            try:
                await self._transport.close()
            finally:
                self._started = False


async def build_sandbox_client(
    sandbox_cfg: Any,
    workspace_root: Optional[str] = None,
) -> Optional[Tuple[SandboxClient, PathTranslator]]:
    """Connect to an externally-launched sandbox service and return a client.

    Parameters
    ----------
    sandbox_cfg:
        The SandboxConfig pydantic object. Only `enabled` and `endpoint`
        are read. Everything else is module-level / derived.
    workspace_root:
        Host-side absolute path that should map to /workspace inside the
        sandbox. Used by PathTranslator to rewrite tool arguments. Pass
        None if you don't need path translation.

    Returns
    -------
    None
        When `sandbox_cfg` is None or `enabled` is False. The agent
        proceeds with in-process tool execution.
    (SandboxClient, PathTranslator)
        When the readiness probe succeeds.

    Raises
    ------
    SandboxUnavailableError
        When `enabled=True` and the endpoint does not become healthy
        within HEALTH_PROBE_TIMEOUT_SEC. No silent fallback: enabling the
        sandbox is an explicit safety choice; failing loudly is correct.
    """
    if sandbox_cfg is None or not getattr(sandbox_cfg, "enabled", False):
        return None

    endpoint = getattr(sandbox_cfg, "endpoint", "http://localhost:8765")

    logger.info(
        "[sandbox-client] connecting endpoint=%s probe_timeout=%.1fs",
        endpoint, HEALTH_PROBE_TIMEOUT_SEC,
    )

    transport = HttpTransport(
        base_url=endpoint,
        timeout=HTTP_CALL_TIMEOUT_SEC,
        connect_timeout=HTTP_CONNECT_TIMEOUT_SEC,
    )

    ready = await transport.wait_ready(
        max_wait_seconds=HEALTH_PROBE_TIMEOUT_SEC,
        interval=HEALTH_PROBE_INTERVAL_SEC,
    )
    if not ready:
        await transport.close()
        msg = (
            f"Sandbox service at {endpoint} did not become healthy within "
            f"{HEALTH_PROBE_TIMEOUT_SEC:.1f}s. Start it with "
            f"`docker compose up -d` under docker/sandbox-standalone/, "
            f"then retry. If the agent is itself containerised, make sure "
            f"both containers share a docker network and the endpoint uses "
            f"the sandbox container's DNS name."
        )
        logger.error("[sandbox-client] %s", msg)
        raise SandboxUnavailableError(msg)

    logger.info("[sandbox-client] ready at %s", endpoint)
    client = SandboxClient(transport=transport)
    translator = PathTranslator(
        host_workspace=str(workspace_root) if workspace_root else "",
        container_workspace=client.container_workspace,
    )
    return client, translator
