# Copyright 2025-2026 Horizon RL Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""AgentWorldModel server lifecycle using multiprocessing.

The script defines the app; this module controls how it runs.
Pipe-based readiness, graceful shutdown via SIGTERM.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import multiprocessing as mp
import signal
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from pathlib import Path

from awm.tools import normalize_scenario_name

logger = logging.getLogger(__name__)

SERVER_STARTUP_TIMEOUT = 30

_mp_context = mp.get_context("spawn")


def write_server_script(script_path: Path, scenario: str, envs_path: Path, work_db_path: Path) -> None:
    """Generate an app-only script (no uvicorn, no ``__main__``)."""
    normalized = normalize_scenario_name(scenario)
    with open(envs_path) as f:
        for line in f:
            entry = json.loads(line)
            if normalize_scenario_name(entry["scenario"]) == normalized:
                break
        else:
            raise ValueError(f"Scenario {normalized} not found in {envs_path}")

    lines: list[str] = [
        "import warnings",
        'warnings.filterwarnings("ignore", category=DeprecationWarning)',
    ]
    for src in entry["full_code"].split("\n"):
        if src.strip().startswith("if __name__"):
            break
        if "create_engine(" in src:
            left = src.split("create_engine(")[0]
            src = f"{left}create_engine('sqlite:///{work_db_path}', connect_args={{'check_same_thread': False}})"
        lines.append(src)

    lines.append("")
    lines.append("from fastapi_mcp import FastApiMCP")
    lines.append("FastApiMCP(app).mount_http()")

    script_path.write_text("\n".join(lines))


def _run_server(script: str, port: int, pipe: Connection) -> None:
    """Child process: exec script to get ``app``, run uvicorn, signal readiness."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # KeyboardInterrupt handled by stop_server(), not here
    try:
        import sys

        import uvicorn

        # exec into __main__ so Pydantic can resolve forward references in FastAPI models
        exec(compile(Path(script).read_text(), script, "exec"), vars(sys.modules["__main__"]))  # noqa: S102
        app = sys.modules["__main__"].app  # type: ignore[attr-defined]

        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        config.load()
        server = uvicorn.Server(config)
        server.lifespan = config.lifespan_class(config)

        async def _serve() -> None:
            await server.startup()  # bind socket, start accepting connections
            pipe.send(True)  # signal parent: server is ready
            pipe.close()  # done with pipe, release fd
            await server.main_loop()  # serve requests until SIGTERM
            await server.shutdown()  # close connections, clean up

        asyncio.run(_serve())
    except Exception as exc:
        with contextlib.suppress(Exception):
            # Send as RuntimeError — original exception may not be picklable (e.g. Pydantic errors)
            pipe.send(RuntimeError(f"{type(exc).__name__}: {exc}"))
            pipe.close()


def start_server(script: Path, port: int, timeout: float = SERVER_STARTUP_TIMEOUT) -> BaseProcess:
    """Start server in a child process. Blocks until uvicorn is accepting connections."""
    parent, child = _mp_context.Pipe(duplex=False)
    proc = _mp_context.Process(target=_run_server, args=(str(script), port, child), daemon=True)
    proc.start()
    child.close()
    try:
        if not parent.poll(timeout):
            raise TimeoutError(f"Server did not start within {timeout}s")
        result = parent.recv()
        if isinstance(result, Exception):
            raise result
    except BaseException:
        stop_server(proc)
        raise
    finally:
        parent.close()
    return proc


def stop_server(proc: BaseProcess | None, timeout: float = 5) -> None:
    """SIGTERM → join → SIGKILL → close."""
    if proc is None:
        return
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=timeout)
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=2)
    proc.close()
