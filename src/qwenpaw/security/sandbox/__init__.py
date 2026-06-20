# -*- coding: utf-8 -*-
"""QwenPaw sandbox subsystem — Tool Service Pattern.

Architecture:
  * Host process runs only orchestration + non-sandboxable tools (browser,
    multi-agent IPC, etc.).
  * A Docker container runs the FULL QwenPaw tool source as a service
    (qwenpaw-sandbox image, FastAPI server on port 8765).
  * Host tools are replaced at registration time with thin RPC proxies
    that forward (name, arguments) over HTTP to the container.

Adding a new tool: zero work in this directory.  Define the tool the
normal way under qwenpaw.agents.tools.*, rebuild the sandbox image, done.

Public API:
  DockerSandboxExecutor — manages container lifecycle, exposes transport
  HttpTransport         — RPC client (host -> container)
  PathTranslator        — rewrites host paths to /workspace paths in args
  proxify_tool_dict()   — converts a dict of tool fns into RPC proxies
  HOST_BOUND_TOOLS      — set of tool names that must NOT be sandboxed
"""
from .executor import DockerSandboxExecutor, DEFAULT_SANDBOXED_TOOLS
from .transport import HttpTransport, ToolTransport
from .manager import SandboxManager
from .proxy_factory import (
    HOST_BOUND_TOOLS,
    PathTranslator,
    make_sandbox_proxy,
    proxify_tool_dict,
)

__all__ = [
    "SandboxManager",
    "DockerSandboxExecutor",
    "DEFAULT_SANDBOXED_TOOLS",
    "HttpTransport",
    "ToolTransport",
    "HOST_BOUND_TOOLS",
    "PathTranslator",
    "make_sandbox_proxy",
    "proxify_tool_dict",
]


# ---------------------------------------------------------------------------
# Logging fallback: ensure sandbox INFO logs are visible even when this module
# is imported outside the regular CLI entrypoint (e.g. ad-hoc scripts, pytest,
# `python -m qwenpaw.something`). The CLI's app_cmd.setup_logger() wins when
# present (we never overwrite an already-configured `qwenpaw` logger).
# ---------------------------------------------------------------------------
def _bootstrap_sandbox_logging() -> None:
    import logging
    import os as _os
    import sys as _sys

    pkg_logger = logging.getLogger("qwenpaw")
    # If something already configured the package logger, leave it alone.
    if pkg_logger.handlers:
        return

    # Honor an env override; default INFO so [path-xlat] / [sandbox-call] /
    # [sandbox-rpc] lines surface without extra setup.
    level_name = _os.environ.get("QWENPAW_SANDBOX_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    handler = logging.StreamHandler(_sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    pkg_logger.addHandler(handler)
    pkg_logger.setLevel(level)
    pkg_logger.propagate = False


_bootstrap_sandbox_logging()
