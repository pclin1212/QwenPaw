# -*- coding: utf-8 -*-
"""SandboxManager -- multi-scope container lifecycle manager.

Supports three scopes controlled by ``security.sandbox.scope`` in config:

  ``"session"`` -- one container per (agent_id, session_id).
      Production / multi-user default.  Perfect isolation between
      concurrent sessions; resources reclaimed on session end or idle TTL.

  ``"agent"`` -- one container per agent_id, shared across all sessions
      that talk to the same agent.  Low overhead; preserves environment
      state across sessions.  Good for single-dev local work.

  ``"shared"`` -- one global container for every agent and every session
      on the platform.  Minimal overhead, zero isolation.  Not recommended
      for production / multi-tenant.

Lifecycle:
  * ``acquire()``         -- get-or-create a handle (executor, translator).
  * ``release_session()`` -- stop one session's sandbox (scope="session").
  * ``shutdown()``        -- stop ALL sandboxes owned by this manager.
  * Idle session sandboxes are auto-reaped after ``SESSION_IDLE_TTL``.
  * Beyond ``MAX_SESSION_SANDBOXES``, the least-recently-used is evicted.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)

# (executor, path_translator)
SandboxHandle = Tuple[Any, Any]

# Module-level singleton for "shared" scope -- truly global across all
# AgentRunner instances / all workspaces.
_shared_handle: Optional[SandboxHandle] = None
_shared_lock: asyncio.Lock = asyncio.Lock()


class SandboxManager:
    """Per-workspace sandbox container manager with configurable scope.

    Each ``AgentRunner`` owns one instance.  The ``"shared"`` scope
    delegates to a module-level singleton so that containers are shared
    across all workspaces.
    """

    # Safety cap on concurrent session-scoped containers per workspace.
    MAX_SESSION_SANDBOXES = 16

    # Idle session sandboxes are reaped after this many seconds.
    SESSION_IDLE_TTL = 1800  # 30 min

    def __init__(self, agent_id: str, workspace_dir: str) -> None:
        self.agent_id = agent_id
        self.workspace_dir = workspace_dir

        # "agent" scope: single cached handle.
        self._agent_handle: Optional[SandboxHandle] = None

        # "session" scope: session_id -> [handle, last_used_monotonic]
        self._session_handles: dict[str, list] = {}

        # Per-key locks allow parallel starts for different sessions
        # while deduping starts for the same key.
        self._key_locks: dict[str, asyncio.Lock] = {}
        self._struct_lock = asyncio.Lock()  # guards dict mutations

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def acquire(
        self,
        scope: str,
        session_id: str,
        sandbox_cfg: Any,
    ) -> Optional[SandboxHandle]:
        """Get or create a sandbox handle for *scope*.

        Returns ``(executor, path_translator)`` or ``None`` if sandbox is
        disabled or failed in non-strict mode.  Raises ``RuntimeError``
        in strict mode on failure.
        """
        if not sandbox_cfg or not getattr(sandbox_cfg, "enabled", False):
            return None

        if scope == "shared":
            return await self._acquire_shared(sandbox_cfg)
        elif scope == "session":
            return await self._acquire_session(
                session_id or "_default", sandbox_cfg,
            )
        else:
            return await self._acquire_agent(sandbox_cfg)

    async def release_session(self, session_id: str) -> None:
        """Stop and remove the sandbox for *session_id* (scope="session")."""
        sid = session_id or "_default"
        async with self._struct_lock:
            entry = self._session_handles.pop(sid, None)
        if entry is not None:
            handle = entry[0]
            await self._stop_handle(handle, f"session:{sid[:8]}")

    async def shutdown(self) -> None:
        """Stop ALL sandboxes managed by this instance.

        Does NOT stop the module-level shared sandbox -- call
        ``shutdown_shared()`` separately on application exit.
        """
        async with self._struct_lock:
            agent_h = self._agent_handle
            session_items = [
                (sid, entry[0])
                for sid, entry in self._session_handles.items()
            ]
            self._agent_handle = None
            self._session_handles.clear()

        if agent_h is not None:
            await self._stop_handle(agent_h, "agent")
        for sid, handle in session_items:
            await self._stop_handle(handle, f"session:{sid[:8]}")

    # ------------------------------------------------------------------
    # Scope: agent
    # ------------------------------------------------------------------

    async def _acquire_agent(
        self, sandbox_cfg: Any,
    ) -> Optional[SandboxHandle]:
        lock = await self._get_key_lock(f"agent:{self.agent_id}")
        async with lock:
            if self._agent_handle is not None:
                executor, _ = self._agent_handle
                if executor.started:
                    return self._agent_handle
                self._agent_handle = None  # stale

            try:
                self._agent_handle = await self._create_and_start(
                    sandbox_cfg, "agent",
                )
                return self._agent_handle
            except Exception as exc:
                if self._agent_handle is not None:
                    ex, _ = self._agent_handle
                    try:
                        await ex.stop()
                    except Exception:  # pylint: disable=broad-except
                        pass
                self._agent_handle = None
                return self._on_failure(exc, sandbox_cfg, "agent")

    # ------------------------------------------------------------------
    # Scope: session
    # ------------------------------------------------------------------

    async def _acquire_session(
        self,
        session_id: str,
        sandbox_cfg: Any,
    ) -> Optional[SandboxHandle]:
        # Reap idle / dead session sandboxes first (best-effort).
        await self._reap_idle_sessions()

        lock = await self._get_key_lock(
            f"session:{self.agent_id}:{session_id}",
        )
        async with lock:
            entry = self._session_handles.get(session_id)
            if entry is not None:
                handle = entry[0]
                executor, _ = handle
                if executor.started:
                    entry[1] = time.monotonic()  # touch LRU
                    return handle
                self._session_handles.pop(session_id, None)

            # Enforce max concurrent session sandboxes (LRU eviction).
            await self._maybe_evict_session()

            try:
                handle = await self._create_and_start(
                    sandbox_cfg, f"session:{session_id[:8]}",
                )
                async with self._struct_lock:
                    self._session_handles[session_id] = [
                        handle, time.monotonic(),
                    ]
                return handle
            except Exception as exc:
                return self._on_failure(
                    exc, sandbox_cfg, f"session:{session_id[:8]}",
                )

    async def _maybe_evict_session(self) -> None:
        """If at capacity, evict the least-recently-used session sandbox."""
        if len(self._session_handles) < self.MAX_SESSION_SANDBOXES:
            return
        oldest_sid = min(
            self._session_handles,
            key=lambda k: self._session_handles[k][1],
        )
        entry = self._session_handles.pop(oldest_sid)
        handle = entry[0]
        await self._stop_handle(
            handle, f"session:{oldest_sid[:8]} (LRU evicted)",
        )

    async def _reap_idle_sessions(self) -> None:
        """Remove session sandboxes that are idle or dead."""
        now = time.monotonic()
        to_reap: list[tuple[str, SandboxHandle]] = []
        async with self._struct_lock:
            for sid, entry in list(self._session_handles.items()):
                handle = entry[0]
                executor, _ = handle
                age = now - entry[1]
                if age > self.SESSION_IDLE_TTL or not executor.started:
                    to_reap.append((sid, handle))
                    del self._session_handles[sid]
        for sid, handle in to_reap:
            await self._stop_handle(
                handle, f"session:{sid[:8]} (idle reaped)",
            )

    # ------------------------------------------------------------------
    # Scope: shared (module-level singleton)
    # ------------------------------------------------------------------

    async def _acquire_shared(
        self, sandbox_cfg: Any,
    ) -> Optional[SandboxHandle]:
        global _shared_handle
        async with _shared_lock:
            if _shared_handle is not None:
                executor, _ = _shared_handle
                if executor.started:
                    return _shared_handle
                _shared_handle = None

            try:
                _shared_handle = await self._create_and_start(
                    sandbox_cfg, "shared",
                )
                return _shared_handle
            except Exception as exc:
                if _shared_handle is not None:
                    ex, _ = _shared_handle
                    try:
                        await ex.stop()
                    except Exception:  # pylint: disable=broad-except
                        pass
                _shared_handle = None
                strict = bool(getattr(sandbox_cfg, "strict", False))
                if strict:
                    logger.error(
                        "Shared sandbox STRICT failure (%s); "
                        "refusing host fallback", exc,
                    )
                    raise RuntimeError(
                        f"Sandbox is required (strict=true) but "
                        f"failed: {exc}. Fix the sandbox or set "
                        f"strict=false."
                    ) from exc
                logger.warning(
                    "Shared sandbox failed (%s); local fallback", exc,
                )
                return None

    @staticmethod
    async def shutdown_shared() -> None:
        """Stop the module-level shared sandbox (app shutdown)."""
        global _shared_handle
        async with _shared_lock:
            handle = _shared_handle
            _shared_handle = None
        if handle is not None:
            executor, _ = handle
            try:
                await executor.stop()
                logger.info("Shared sandbox stopped (global)")
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("Shared sandbox stop failed: %s", exc)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _get_key_lock(self, key: str) -> asyncio.Lock:
        async with self._struct_lock:
            if key not in self._key_locks:
                self._key_locks[key] = asyncio.Lock()
            return self._key_locks[key]

    async def _create_and_start(
        self,
        sandbox_cfg: Any,
        label: str,
    ) -> SandboxHandle:
        """Build executor + translator, start, return handle."""
        from .executor import DockerSandboxExecutor
        from .proxy_factory import PathTranslator

        executor = DockerSandboxExecutor(
            host_workspace=self.workspace_dir,
            image=sandbox_cfg.image,
            memory_limit=sandbox_cfg.memory_limit,
            cpu_quota=sandbox_cfg.cpu_quota,
            network_enabled=sandbox_cfg.network_enabled,
            extra_volumes=dict(sandbox_cfg.extra_volumes) or None,
            env_vars=dict(sandbox_cfg.env_vars) or None,
            sandboxed_tools=sandbox_cfg.sandboxed_tools,
            ready_timeout_seconds=sandbox_cfg.ready_timeout_seconds,
        )
        ready_timeout = getattr(sandbox_cfg, "ready_timeout_seconds", 60)
        await executor.start(ready_timeout_seconds=ready_timeout)
        translator = PathTranslator(
            host_workspace=self.workspace_dir,
            container_workspace=executor.container_workspace,
        )
        from datetime import datetime as _sb_dt
        _sb_ts = _sb_dt.now().isoformat(timespec="milliseconds")
        logger.info(
            "Sandbox started [%s] agent=%s: %s ts=%s",
            label, self.agent_id, executor.container_name, _sb_ts,
        )
        return executor, translator

    @staticmethod
    def _on_failure(
        exc: Exception,
        sandbox_cfg: Any,
        label: str,
    ) -> Optional[SandboxHandle]:
        """Strict -> raise; non-strict -> log + return None."""
        strict = bool(getattr(sandbox_cfg, "strict", False))
        if strict:
            logger.error(
                "Sandbox [%s] STRICT failure (%s); "
                "refusing host fallback",
                label, exc,
            )
            raise RuntimeError(
                f"Sandbox is required (strict=true) but "
                f"failed: {exc}. Fix the sandbox or set "
                f"strict=false."
            ) from exc
        logger.warning(
            "Sandbox [%s] failed (%s); local fallback",
            label, exc,
        )
        return None

    @staticmethod
    async def _stop_handle(
        handle: Optional[SandboxHandle], label: str,
    ) -> None:
        """Best-effort stop of a sandbox handle."""
        if handle is None:
            return
        executor, _ = handle
        try:
            await executor.stop()
            logger.info("Sandbox stopped [%s]", label)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Sandbox stop failed [%s]: %s", label, exc)
