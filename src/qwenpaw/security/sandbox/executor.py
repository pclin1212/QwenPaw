# -*- coding: utf-8 -*-
"""DockerSandboxExecutor — container lifecycle manager.

Purpose: bring up a Docker container that runs the QwenPaw tool server,
publish its port to the host, and expose an HttpTransport pointing at it.
Nothing else.  The host process never executes commands inside the
container directly — all communication goes through the tool server's
RPC endpoints.

This is a pure orchestration layer.  Compare to the previous design
where this file contained exec_command/exec_python/read_file/write_file
methods that issued docker exec calls — all of that is gone.  The
container runs a real service; we just talk to it.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import subprocess
import uuid
from pathlib import Path
from typing import Optional

from .transport import HttpTransport


logger = logging.getLogger(__name__)


# Default tool manifest — host tells the container which tools to expose.
# Tools that are host-bound (browser, screenshot, agent-management) are
# explicitly excluded so the sandbox doesn't waste startup time loading them.
DEFAULT_SANDBOXED_TOOLS = (
    "execute_shell_command",
    "read_file",
    "write_file",
    "edit_file",
    "append_file",
    "grep_search",
    "glob_search",
    "run_tool_batch",
)


def _find_free_port() -> int:
    """Find an unused TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class DockerSandboxExecutor:
    """Manages one persistent container per agent session."""

    def __init__(
        self,
        host_workspace: str,
        image: str = "qwenpaw-sandbox:latest",
        memory_limit: Optional[str] = "1g",
        cpu_quota: Optional[float] = None,
        network_enabled: bool = False,
        extra_volumes: Optional[dict] = None,
        env_vars: Optional[dict] = None,
        sandboxed_tools=None,
        container_name_prefix: str = "qwenpaw-sandbox",
        ready_timeout_seconds: int = 60,
    ) -> None:
        self.host_workspace = os.path.abspath(host_workspace)
        self.image = image
        self.memory_limit = memory_limit
        self.cpu_quota = cpu_quota
        self.network_enabled = network_enabled
        self.extra_volumes = extra_volumes or {} or {}
        self.env_vars = env_vars or {} or {}
        self.sandboxed_tools = sandboxed_tools if sandboxed_tools is not None else DEFAULT_SANDBOXED_TOOLS
        self.ready_timeout_seconds = ready_timeout_seconds

        self.container_name = f"{container_name_prefix}-{uuid.uuid4().hex[:8]}"
        self.host_port: Optional[int] = None
        self.transport: Optional[HttpTransport] = None
        self._started = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, ready_timeout_seconds: Optional[float] = None) -> None:
        """Launch the container and wait for the tool server to be healthy."""
        if self._started:
            return

        if ready_timeout_seconds is None:
            ready_timeout_seconds = float(self.ready_timeout_seconds)

        # Pre-flight: docker available?
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError("docker CLI returned non-zero")
        except FileNotFoundError as e:
            raise RuntimeError("docker not installed or not in PATH") from e

        # Write tool manifest to a temp file, mount it into the container
        manifest_path = self._write_tool_manifest()

        self.host_port = _find_free_port()
        cmd = self._build_run_command(manifest_path)
        logger.info("Starting sandbox container: %s", self.container_name)
        logger.debug("docker cmd: %s", " ".join(cmd))

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"docker run failed: {stderr.decode('utf-8', errors='replace')}"
            )

        # Connect transport and wait for /health
        self.transport = HttpTransport(f"http://127.0.0.1:{self.host_port}")
        ready = await self.transport.wait_ready(max_wait_seconds=ready_timeout_seconds)
        if not ready:
            # Capture container logs for debugging
            logs = await self._capture_logs()
            await self.stop()
            raise RuntimeError(
                f"sandbox tool server did not become ready within "
                f"{ready_timeout_seconds}s.  Container logs:\n{logs}"
            )

        # Confirm tools registered
        try:
            tools = await self.transport.list_tools()
            logger.info("Sandbox ready with %d tools: %s", len(tools), ", ".join(tools))
        except Exception as e:
            logger.warning("Could not list tools after startup: %s", e)

        self._started = True

    async def stop(self) -> None:
        """Stop and remove the container."""
        if self.transport:
            await self.transport.close()
            self.transport = None

        # Best-effort container cleanup
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "stop", "-t", "5", self.container_name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.communicate()
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("Failed to stop container %s: %s", self.container_name, e)

        self._started = False

    @property
    def started(self) -> bool:
        return self._started

    @property
    def container_workspace(self) -> str:
        return "/workspace"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _write_tool_manifest(self) -> str:
        """Serialize the tool list to a host-side file we'll bind-mount."""
        manifests_dir = Path.home() / ".qwenpaw" / "sandbox" / "manifests"
        manifests_dir.mkdir(parents=True, exist_ok=True)
        path = manifests_dir / f"{self.container_name}.json"
        path.write_text(
            json.dumps(list(self.sandboxed_tools), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return str(path.resolve())

    def _build_run_command(self, manifest_host_path: str) -> list:
        cmd = [
            "docker", "run",
            "-d",
            "--rm",
            "--name", self.container_name,
            "-p", f"127.0.0.1:{self.host_port}:8765",
            "-v", f"{self.host_workspace}:/workspace",
            "-v", f"{manifest_host_path}:/app/tool_manifest.json:ro",
            "--security-opt", "no-new-privileges",
            "-w", "/workspace",
        ]
        if self.memory_limit:
            cmd.extend(["--memory", str(self.memory_limit)])
        if self.cpu_quota is not None:
            # Accept either a fraction of cores (float, e.g. 1.5) or raw
            # microseconds (int >= 1000).  Default Docker --cpu-period is 100000.
            try:
                q = float(self.cpu_quota)
                if q <= 16:  # treat as cores
                    micro = int(q * 100000)
                else:        # treat as raw microseconds
                    micro = int(q)
                cmd.extend(["--cpu-quota", str(micro)])
            except (TypeError, ValueError):
                pass

        if not self.network_enabled:
            # Note: --network none would block /tools/call too.  We need
            # localhost connectivity for the published port to work.
            # Use a custom network with no outbound instead, OR rely on
            # the published port + iptables.  For simplicity we use the
            # default bridge and document that the sandbox CAN reach
            # outbound; real isolation requires an internal network.
            # TODO: add --network mode=bridge with internal=true once tested.
            pass

        for host_path, container_path in self.extra_volumes.items():
            cmd.extend(["-v", f"{host_path}:{container_path}"])

        for k, v in self.env_vars.items():
            cmd.extend(["-e", f"{k}={v}"])

        cmd.append(self.image)
        return cmd

    async def _capture_logs(self, tail: int = 50) -> str:
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "logs", "--tail", str(tail), self.container_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            stdout, _ = await proc.communicate()
            return stdout.decode("utf-8", errors="replace")
        except Exception as e:  # pylint: disable=broad-except
            return f"(could not capture logs: {e})"
