from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Type, Union

from agno.agent.agent import Agent
from agno.media import Audio, File, Image, Video
from agno.models.base import Model
from agno.models.message import Message
from agno.models.response import ToolExecution
from agno.run.messages import RunMessages
from agno.run.agent import RunOutput, RunOutputEvent
from agno.session import AgentSession
from agno.tools import Toolkit
from agno.tools.function import Function
from agno.utils.log import log_debug, log_warning

from pydantic import BaseModel

try:  # pragma: no cover - MCP optional dependency
    from agno.tools.mcp import MCPTools, MultiMCPTools
except (ImportError, ModuleNotFoundError):  # pragma: no cover - MCP not installed
    MCPToolkitTypes: Tuple[type, ...] = ()
else:  # pragma: no cover - import succeeds when MCP extras installed
    MCPToolkitTypes = (MCPTools, MultiMCPTools)

try:  # Python 3.11+
    BaseExceptionGroupType = BaseExceptionGroup  # type: ignore[name-defined]
except NameError:  # pragma: no cover - Python <3.11 (unused)
    BaseExceptionGroupType = tuple()  # type: ignore[assignment]


class MCPAgent(Agent):
    """Agent subclass with automatic Model Context Protocol tool management."""

    def __init__(
        self,
        *args: Any,
        auto_cleanup_mcp: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._auto_cleanup_mcp: bool = auto_cleanup_mcp
        self._pending_mcp_toolkits: List[Toolkit] = []
        self._managed_mcp_toolkits: List[Toolkit] = []
        self._failed_mcp_toolkits: Set[int] = set()
        self._mcp_context: Optional[Dict[str, Any]] = None
        self._mcp_rebuilding: bool = False

    # ------------------------------------------------------------------
    # Tool lifecycle hooks
    # ------------------------------------------------------------------
    def get_tools(
        self,
        run_response: RunOutput,
        session: AgentSession,
        async_mode: bool = False,
        user_id: Optional[str] = None,
        knowledge_filters: Optional[Dict[str, Any]] = None,
    ) -> Optional[List[Union[Toolkit, Callable[..., Any], Function, Dict[str, Any]]]]:
        tools = super().get_tools(
            run_response=run_response,
            session=session,
            async_mode=async_mode,
            user_id=user_id,
            knowledge_filters=knowledge_filters,
        )

        if not tools:
            return tools

        for tool in tools:
            if not self._is_mcp_toolkit(tool):
                continue

            if async_mode:
                if not getattr(tool, "_initialized", False):
                    self._add_pending_mcp_toolkit(tool)
            else:
                self._connect_mcp_toolkit_blocking(tool)

            if getattr(tool, "_initialized", False):
                self._register_connected_toolkit(tool)

        return tools

    def _determine_tools_for_model(
        self,
        model: Model,
        run_response: RunOutput,
        session: AgentSession,
        session_state: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        async_mode: bool = False,
        knowledge_filters: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self._mcp_rebuilding and async_mode:
            self._pending_mcp_toolkits = []
            self._failed_mcp_toolkits.clear()
            self._mcp_context = None

        super()._determine_tools_for_model(
            model=model,
            run_response=run_response,
            session=session,
            session_state=session_state,
            user_id=user_id,
            async_mode=async_mode,
            knowledge_filters=knowledge_filters,
        )

        self._mcp_context = {
            "model": model,
            "run_response": run_response,
            "session": session,
            "session_state": session_state,
            "user_id": user_id,
            "knowledge_filters": knowledge_filters,
        }

    # ------------------------------------------------------------------
    # Overridden execution methods (async)
    # ------------------------------------------------------------------
    async def _arun(
        self,
        run_response: RunOutput,
        input: Union[str, List, Dict, Message, BaseModel, List[Message]],
        session: AgentSession,
        session_state: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        images: Optional[Sequence[Image]] = None,
        videos: Optional[Sequence[Video]] = None,
        audio: Optional[Sequence[Audio]] = None,
        files: Optional[Sequence[File]] = None,
        knowledge_filters: Optional[Dict[str, Any]] = None,
        add_history_to_context: Optional[bool] = None,
        add_dependencies_to_context: Optional[bool] = None,
        add_session_state_to_context: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
        response_format: Optional[Union[Dict[str, Any], Type[BaseModel]]] = None,
        dependencies: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> RunOutput:
        await self._ensure_async_mcp_toolkits_ready(
            model=self.model,
            run_response=run_response,
            session=session,
            session_state=session_state,
            user_id=user_id,
            knowledge_filters=knowledge_filters,
        )

        try:
            return await super()._arun(
                run_response=run_response,
                input=input,
                session=session,
                session_state=session_state,
                user_id=user_id,
                images=images,
                videos=videos,
                audio=audio,
                files=files,
                knowledge_filters=knowledge_filters,
                add_history_to_context=add_history_to_context,
                add_dependencies_to_context=add_dependencies_to_context,
                add_session_state_to_context=add_session_state_to_context,
                metadata=metadata,
                response_format=response_format,
                dependencies=dependencies,
                **kwargs,
            )
        finally:
            if self._auto_cleanup_mcp:
                await self._cleanup_managed_mcp_toolkits()

    async def _arun_stream(
        self,
        run_response: RunOutput,
        session: AgentSession,
        input: Union[str, List, Dict, Message, BaseModel, List[Message]],
        session_state: Optional[Dict[str, Any]] = None,
        audio: Optional[Sequence[Audio]] = None,
        images: Optional[Sequence[Image]] = None,
        videos: Optional[Sequence[Video]] = None,
        files: Optional[Sequence[File]] = None,
        knowledge_filters: Optional[Dict[str, Any]] = None,
        add_history_to_context: Optional[bool] = None,
        add_dependencies_to_context: Optional[bool] = None,
        add_session_state_to_context: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
        dependencies: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        response_format: Optional[Union[Dict[str, Any], Type[BaseModel]]] = None,
        stream_intermediate_steps: bool = False,
        workflow_context: Optional[Dict[str, Any]] = None,
        yield_run_response: Optional[bool] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Union[RunOutputEvent, RunOutput]]:
        await self._ensure_async_mcp_toolkits_ready(
            model=self.model,
            run_response=run_response,
            session=session,
            session_state=session_state,
            user_id=user_id,
            knowledge_filters=knowledge_filters,
        )

        try:
            async for event in super()._arun_stream(
                run_response=run_response,
                session=session,
                input=input,
                session_state=session_state,
                audio=audio,
                images=images,
                videos=videos,
                files=files,
                knowledge_filters=knowledge_filters,
                add_history_to_context=add_history_to_context,
                add_dependencies_to_context=add_dependencies_to_context,
                add_session_state_to_context=add_session_state_to_context,
                metadata=metadata,
                dependencies=dependencies,
                user_id=user_id,
                response_format=response_format,
                stream_intermediate_steps=stream_intermediate_steps,
                workflow_context=workflow_context,
                yield_run_response=yield_run_response,
                **kwargs,
            ):
                yield event
        finally:
            if self._auto_cleanup_mcp:
                await self._cleanup_managed_mcp_toolkits()

    async def _acontinue_run(
        self,
        run_response: RunOutput,
        run_messages: RunMessages,
        session: AgentSession,
        user_id: Optional[str] = None,
        response_format: Optional[Union[Dict[str, Any], Type[BaseModel]]] = None,
        dependencies: Optional[Dict[str, Any]] = None,
    ) -> RunOutput:
        await self._ensure_async_mcp_toolkits_ready(
            model=self.model,
            run_response=run_response,
            session=session,
            session_state=None,
            user_id=user_id,
            knowledge_filters=None,
        )

        try:
            return await super()._acontinue_run(
                run_response=run_response,
                run_messages=run_messages,
                session=session,
                user_id=user_id,
                response_format=response_format,
                dependencies=dependencies,
            )
        finally:
            if self._auto_cleanup_mcp:
                await self._cleanup_managed_mcp_toolkits()

    async def _acontinue_run_stream(
        self,
        run_response: RunOutput,
        run_messages: RunMessages,
        session: AgentSession,
        user_id: Optional[str] = None,
        response_format: Optional[Union[Dict[str, Any], Type[BaseModel]]] = None,
        stream_intermediate_steps: bool = False,
        dependencies: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[Union[RunOutputEvent, RunOutput]]:
        await self._ensure_async_mcp_toolkits_ready(
            model=self.model,
            run_response=run_response,
            session=session,
            session_state=None,
            user_id=user_id,
            knowledge_filters=None,
        )

        try:
            async for event in super()._acontinue_run_stream(
                run_response=run_response,
                run_messages=run_messages,
                session=session,
                user_id=user_id,
                response_format=response_format,
                stream_intermediate_steps=stream_intermediate_steps,
                dependencies=dependencies,
            ):
                yield event
        finally:
            if self._auto_cleanup_mcp:
                await self._cleanup_managed_mcp_toolkits()

    # ------------------------------------------------------------------
    # Overridden execution methods (sync)
    # ------------------------------------------------------------------
    def _run(
        self,
        run_response: RunOutput,
        run_messages: RunMessages,
        session: AgentSession,
        user_id: Optional[str] = None,
        response_format: Optional[Union[Dict[str, Any], Type[BaseModel]]] = None,
    ) -> RunOutput:
        self._ensure_sync_mcp_toolkits_ready(
            model=self.model,
            run_response=run_response,
            session=session,
            session_state=None,
            user_id=user_id,
            knowledge_filters=None,
        )

        try:
            return super()._run(
                run_response=run_response,
                run_messages=run_messages,
                session=session,
                user_id=user_id,
                response_format=response_format,
            )
        finally:
            if self._auto_cleanup_mcp:
                self._cleanup_managed_mcp_toolkits_sync()

    def _run_stream(
        self,
        run_response: RunOutput,
        run_messages: RunMessages,
        session: AgentSession,
        user_id: Optional[str] = None,
        response_format: Optional[Union[Dict[str, Any], Type[BaseModel]]] = None,
        stream_intermediate_steps: bool = False,
        workflow_context: Optional[Dict[str, Any]] = None,
        yield_run_response: bool = False,
    ) -> Iterator[Union[RunOutputEvent, RunOutput]]:
        self._ensure_sync_mcp_toolkits_ready(
            model=self.model,
            run_response=run_response,
            session=session,
            session_state=None,
            user_id=user_id,
            knowledge_filters=None,
        )

        iterator = super()._run_stream(
            run_response=run_response,
            run_messages=run_messages,
            session=session,
            user_id=user_id,
            response_format=response_format,
            stream_intermediate_steps=stream_intermediate_steps,
            workflow_context=workflow_context,
            yield_run_response=yield_run_response,
        )

        try:
            for event in iterator:
                yield event
        finally:
            if self._auto_cleanup_mcp:
                self._cleanup_managed_mcp_toolkits_sync()

    def _continue_run(
        self,
        run_response: RunOutput,
        run_messages: RunMessages,
        session: AgentSession,
        user_id: Optional[str] = None,
        response_format: Optional[Union[Dict[str, Any], Type[BaseModel]]] = None,
        updated_tools: Optional[List[ToolExecution]] = None,
    ) -> RunOutput:
        self._ensure_sync_mcp_toolkits_ready(
            model=self.model,
            run_response=run_response,
            session=session,
            session_state=None,
            user_id=user_id,
            knowledge_filters=None,
        )

        try:
            return super()._continue_run(
                run_response=run_response,
                run_messages=run_messages,
                session=session,
                user_id=user_id,
                response_format=response_format,
                updated_tools=updated_tools,
            )
        finally:
            if self._auto_cleanup_mcp:
                self._cleanup_managed_mcp_toolkits_sync()

    def _continue_run_stream(
        self,
        run_response: RunOutput,
        run_messages: RunMessages,
        session: AgentSession,
        user_id: Optional[str] = None,
        response_format: Optional[Union[Dict[str, Any], Type[BaseModel]]] = None,
        stream_intermediate_steps: bool = False,
        updated_tools: Optional[List[ToolExecution]] = None,
    ) -> Iterator[Union[RunOutputEvent, RunOutput]]:
        self._ensure_sync_mcp_toolkits_ready(
            model=self.model,
            run_response=run_response,
            session=session,
            session_state=None,
            user_id=user_id,
            knowledge_filters=None,
        )

        iterator = super()._continue_run_stream(
            run_response=run_response,
            run_messages=run_messages,
            session=session,
            user_id=user_id,
            response_format=response_format,
            stream_intermediate_steps=stream_intermediate_steps,
            updated_tools=updated_tools,
        )

        try:
            for event in iterator:
                yield event
        finally:
            if self._auto_cleanup_mcp:
                self._cleanup_managed_mcp_toolkits_sync()

    # ------------------------------------------------------------------
    # Cleanup helpers
    # ------------------------------------------------------------------
    async def aclose(self) -> None:
        await self._cleanup_managed_mcp_toolkits()

    def _cleanup_managed_mcp_toolkits_sync(self) -> None:
        if not self._managed_mcp_toolkits:
            return

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self._cleanup_managed_mcp_toolkits())
        finally:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

    async def _cleanup_managed_mcp_toolkits(self) -> None:
        if not self._managed_mcp_toolkits:
            return

        for toolkit in list(self._managed_mcp_toolkits):
            try:
                await toolkit.close()  # type: ignore[attr-defined]
            except KeyboardInterrupt:
                raise
            except BaseException as exc:  # pragma: no cover - defensive logging
                if not self._should_silence_mcp_error(exc):
                    toolkit_name = getattr(toolkit, "name", toolkit.__class__.__name__)
                    log_warning(f"Failed to close MCP toolkit {toolkit_name}: {exc}")

        self._managed_mcp_toolkits.clear()
        self._pending_mcp_toolkits.clear()
        self._failed_mcp_toolkits.clear()
        self._mcp_context = None
        self._rebuild_tools = True

    # ------------------------------------------------------------------
    # MCP utility helpers
    # ------------------------------------------------------------------
    def _ensure_sync_mcp_toolkits_ready(
        self,
        *,
        model: Optional[Model],
        run_response: RunOutput,
        session: AgentSession,
        session_state: Optional[Dict[str, Any]],
        user_id: Optional[str],
        knowledge_filters: Optional[Dict[str, Any]],
    ) -> None:
        if not self._pending_mcp_toolkits:
            return

        connected_any = False
        for toolkit in list(self._pending_mcp_toolkits):
            if getattr(toolkit, "_initialized", False):
                self._register_connected_toolkit(toolkit)
                continue

            self._connect_mcp_toolkit_blocking(toolkit)
            if getattr(toolkit, "_initialized", False):
                connected_any = True
                self._register_connected_toolkit(toolkit)

        self._pending_mcp_toolkits = [t for t in self._pending_mcp_toolkits if not getattr(t, "_initialized", False)]

        if connected_any:
            self._rebuild_tools_for_model(
                model=model,
                run_response=run_response,
                session=session,
                session_state=session_state,
                user_id=user_id,
                knowledge_filters=knowledge_filters,
                async_mode=True,
            )

    async def _ensure_async_mcp_toolkits_ready(
        self,
        *,
        model: Optional[Model],
        run_response: RunOutput,
        session: AgentSession,
        session_state: Optional[Dict[str, Any]],
        user_id: Optional[str],
        knowledge_filters: Optional[Dict[str, Any]],
    ) -> None:
        if not self._pending_mcp_toolkits:
            return

        pending = [t for t in self._pending_mcp_toolkits if id(t) not in self._failed_mcp_toolkits]
        if not pending:
            return

        connected_any = False
        for toolkit in pending:
            if getattr(toolkit, "_initialized", False):
                self._register_connected_toolkit(toolkit)
                continue

            try:
                await toolkit.connect()  # type: ignore[attr-defined]
                connected_any = True
                self._register_connected_toolkit(toolkit)
                log_debug(f"Automatically connected MCP toolkit {getattr(toolkit, 'name', toolkit)}")
            except KeyboardInterrupt:
                raise
            except BaseException as exc:  # pragma: no cover - defensive logging
                if not self._should_silence_mcp_error(exc):
                    toolkit_name = getattr(toolkit, "name", toolkit.__class__.__name__)
                    log_warning(f"Failed to auto-connect MCP toolkit {toolkit_name}: {exc}")
                self._failed_mcp_toolkits.add(id(toolkit))
                close_coro = getattr(toolkit, "close", None)
                if close_coro is not None:
                    with suppress(Exception, BaseException, asyncio.CancelledError):
                        await close_coro()  # type: ignore[misc]

        self._pending_mcp_toolkits = [t for t in self._pending_mcp_toolkits if not getattr(t, "_initialized", False)]

        if connected_any:
            self._rebuild_tools_for_model(
                model=model,
                run_response=run_response,
                session=session,
                session_state=session_state,
                user_id=user_id,
                knowledge_filters=knowledge_filters,
                async_mode=True,
            )

    def _rebuild_tools_for_model(
        self,
        *,
        model: Optional[Model],
        run_response: RunOutput,
        session: AgentSession,
        session_state: Optional[Dict[str, Any]],
        user_id: Optional[str],
        knowledge_filters: Optional[Dict[str, Any]],
        async_mode: bool,
    ) -> None:
        if model is None:
            return

        self._rebuild_tools = True
        self._mcp_rebuilding = True
        try:
            super()._determine_tools_for_model(
                model=model,
                run_response=run_response,
                session=session,
                session_state=session_state,
                user_id=user_id,
                async_mode=async_mode,
                knowledge_filters=knowledge_filters,
            )
        finally:
            self._mcp_rebuilding = False

    def _register_connected_toolkit(self, toolkit: Toolkit) -> None:
        if toolkit not in self._managed_mcp_toolkits:
            self._managed_mcp_toolkits.append(toolkit)
        self._failed_mcp_toolkits.discard(id(toolkit))

    def _connect_mcp_toolkit_blocking(self, toolkit: Toolkit) -> None:
        if getattr(toolkit, "_initialized", False):
            return

        log_debug(f"Connecting MCP toolkit {getattr(toolkit, 'name', toolkit)} (sync)")

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(toolkit.connect())  # type: ignore[attr-defined]
        except KeyboardInterrupt:
            raise
        except BaseException as exc:  # pragma: no cover - defensive logging
            if not self._should_silence_mcp_error(exc):
                toolkit_name = getattr(toolkit, "name", toolkit.__class__.__name__)
                log_warning(f"Failed to connect MCP toolkit {toolkit_name}: {exc}")
            self._failed_mcp_toolkits.add(id(toolkit))
        finally:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

    def _is_mcp_toolkit(self, tool: Any) -> bool:
        return isinstance(tool, MCPToolkitTypes)

    def _add_pending_mcp_toolkit(self, tool: Toolkit) -> None:
        if tool not in self._pending_mcp_toolkits:
            self._pending_mcp_toolkits.append(tool)
        self._failed_mcp_toolkits.discard(id(tool))

    def _should_silence_mcp_error(self, exc: BaseException) -> bool:
        target_msg = "Attempted to exit cancel scope in a different task than it was entered in"

        if isinstance(exc, BaseExceptionGroupType):  # type: ignore[arg-type]
            return all(self._should_silence_mcp_error(err) for err in exc.exceptions)  # type: ignore[attr-defined]

        return target_msg in str(exc)


__all__ = ["MCPAgent"]
