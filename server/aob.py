"""Per-environment CouchDB databases and the AssetOpsBench MCP servers that read them.

Each environment instance owns four databases (`<prefix>_iot`, `_asset`, `_wo`,
`_fc`) and one IoT and one WO server process pointed at them, so concurrent
episodes never share state. Resetting recreates the databases; the servers keep
running and pick up the new contents on their next query.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import shutil
import threading
from contextlib import AsyncExitStack
from pathlib import Path
from urllib.parse import quote

import requests
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import get_default_environment, stdio_client

AOB_DIR = Path(os.environ.get("AOB_DIR", Path(__file__).resolve().parents[2] / "AssetOpsBench"))


class Couch:
    def __init__(self, prefix: str):
        self.url = os.environ.get("COUCHDB_URL", "http://localhost:5984").rstrip("/")
        self.user = os.environ.get("COUCHDB_USERNAME", "admin")
        self.password = os.environ.get("COUCHDB_PASSWORD", "password")
        self.db = {k: f"{prefix}_{k}" for k in ("iot", "asset", "wo", "fc")}
        self.http = requests.Session()
        self.http.auth = (self.user, self.password)

    def _req(self, method: str, path: str, **kw) -> requests.Response:
        return self.http.request(method, f"{self.url}/{path}", timeout=60, **kw)

    def server_env(self) -> dict:
        return {"COUCHDB_URL": self.url, "COUCHDB_USERNAME": self.user, "COUCHDB_PASSWORD": self.password,
                "IOT_DBNAME": self.db["iot"], "ASSET_DBNAME": self.db["asset"],
                "WO_DBNAME": self.db["wo"], "FAILURE_CODE_DBNAME": self.db["fc"]}

    def recreate(self) -> None:
        for name in self.db.values():
            self._req("DELETE", name)
            self._req("PUT", name).raise_for_status()
        # The IoT server sorts every query by (asset_id, timestamp); Mango needs the index.
        self._req("POST", f"{self.db['iot']}/_index",
                  json={"index": {"fields": ["asset_id", "timestamp"]}}).raise_for_status()

    def drop(self) -> None:
        for name in self.db.values():
            self._req("DELETE", name)

    def bulk(self, key: str, docs: list[dict]) -> None:
        for i in range(0, len(docs), 2000):
            r = self._req("POST", f"{self.db[key]}/_bulk_docs", json={"docs": docs[i:i + 2000]})
            r.raise_for_status()
            bad = [d for d in r.json() if "error" in d]
            if bad:
                raise RuntimeError(f"CouchDB bulk write to {key} failed: {bad[:3]}")

    def find(self, key: str, selector: dict) -> list[dict]:
        r = self._req("POST", f"{self.db[key]}/_find", json={"selector": selector, "limit": 100000})
        r.raise_for_status()
        return r.json()["docs"]

    def get(self, key: str, doc_id: str) -> dict | None:
        r = self._req("GET", f"{self.db[key]}/{quote(doc_id, safe='')}")
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()

    def put(self, key: str, doc: dict) -> None:
        self._req("PUT", f"{self.db[key]}/{quote(doc['_id'], safe='')}", json=doc).raise_for_status()


def _server_command(name: str) -> tuple[str, list[str]]:
    exe = AOB_DIR / ".venv" / "bin" / f"{name}-mcp-server"
    if exe.exists():
        return str(exe), []
    return shutil.which("uv") or "uv", ["run", "--project", str(AOB_DIR), f"{name}-mcp-server"]


class AOBBridge:
    """Synchronous facade over MCP stdio sessions to AssetOpsBench servers.

    The sessions live on a private event loop in a daemon thread, so callers on any
    thread or loop can use them. One owner task enters and exits the stdio contexts,
    which anyio requires.
    """

    def __init__(self, env: dict, servers: tuple[str, ...] = ("iot", "wo"), timeout: float = 60.0):
        self.timeout = timeout
        self.sessions: dict[str, ClientSession] = {}
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._stop: asyncio.Event | None = None
        ready: concurrent.futures.Future = concurrent.futures.Future()
        self._owner = asyncio.run_coroutine_threadsafe(self._own(env, servers, ready), self._loop)
        ready.result(timeout)

    async def _own(self, env: dict, servers, ready) -> None:
        try:
            async with AsyncExitStack() as stack:
                for name in servers:
                    cmd, args = _server_command(name)
                    params = StdioServerParameters(
                        command=cmd, args=args,
                        env={**get_default_environment(), "LOG_LEVEL": "WARNING", **env})
                    read, write = await stack.enter_async_context(stdio_client(params))
                    session = await stack.enter_async_context(ClientSession(read, write))
                    await session.initialize()
                    self.sessions[name] = session
                self._stop = asyncio.Event()
                ready.set_result(True)
                await self._stop.wait()
        except BaseException as e:
            if not ready.done():
                ready.set_exception(e)
            raise

    def _run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(self.timeout)

    def list_tools(self, server: str):
        return self._run(self.sessions[server].list_tools()).tools

    def call(self, server: str, tool: str, arguments: dict):
        res = self._run(self.sessions[server].call_tool(tool, arguments or {}))
        data = res.structuredContent
        if isinstance(data, dict) and set(data) == {"result"}:
            data = data["result"]
        if data is None:
            text = "".join(getattr(c, "text", "") for c in res.content)
            try:
                data = json.loads(text)
            except ValueError:
                data = text
        if res.isError:
            raise RuntimeError(data if isinstance(data, str) else json.dumps(data))
        return data

    def close(self) -> None:
        if self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
            try:
                self._owner.result(10)
            except Exception:
                pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(5)
