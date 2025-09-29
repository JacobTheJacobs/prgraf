from __future__ import annotations

import asyncio
from pathlib import Path
import json
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from .config import detect_provider_from_model, settings
from .graph_updater import GraphUpdater
from .main import (
    _create_configuration_table,
    _initialize_services_and_agent,
    _setup_common_initialization,
)
from .parser_loader import load_parsers
from .services.graph_service import MemgraphIngestor


ModelProvider = Literal["gemini", "openai", "local"]


MODEL_MAP: dict[str, dict[str, str]] = {
    "gemini": {
        "orchestrator": "gemini-2.5-pro",
        "cypher": "gemini-2.5-flash-lite-preview-06-17",
    },
    "local": {
        "orchestrator": "gpt-oss:latest",
        "cypher": "gpt-oss:latest",
    },
}


_FS_KEYS = {"directory_path", "current_path", "entries", "roots", "parent", "name", "path", "type"}


def _looks_like_fs_payload(obj: Any) -> bool:
    try:
        if isinstance(obj, str):
            data = json.loads(obj)
        else:
            data = obj
        if isinstance(data, dict):
            keys = set(data.keys())
            if not keys:
                return False
            # If all top-level keys are filesystem-ish, treat as tool output
            return keys.issubset(_FS_KEYS)
        return False
    except Exception:
        return False


def _extract_text_answer(raw: Any) -> str | None:
    """Best-effort normalization of agent output into plain text.

    - If output is a JSON object with 'answer', return that.
    - If output is a JSON string with the same, return it.
    - Otherwise return stringified text.
    """
    if raw is None:
        return None
    # Pydantic-AI Agent often returns objects with .output already str
    if isinstance(raw, dict):
        # Direct JSON with 'answer'
        ans = raw.get("answer")
        if isinstance(ans, str) and ans.strip():
            return ans.strip()
        # Fallback: stringify
        try:
            return json.dumps(raw, ensure_ascii=False)
        except Exception:
            return str(raw)
    text = str(raw)
    # Try parse JSON string
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            ans = data.get("answer")
            if isinstance(ans, str) and ans.strip():
                return ans.strip()
            # fallback to stringified dict
            return json.dumps(data, ensure_ascii=False)
    except Exception:
        pass
    return text


def _list_local_models() -> set[str]:
    """Return model ids exposed by the local OpenAI-compatible endpoint, if any."""
    try:
        import json
        from urllib.request import Request, urlopen

        base = str(settings.LOCAL_MODEL_ENDPOINT).rstrip("/")
        req = Request(
            base + "/models",
            headers={"Authorization": f"Bearer {settings.LOCAL_MODEL_API_KEY}"},
        )
        with urlopen(req, timeout=5) as resp:  # nosec - local endpoint
            payload = json.loads(resp.read().decode("utf-8"))
        ids: set[str] = set()
        for item in (payload.get("data") or []):
            mid = item.get("id")
            if isinstance(mid, str) and mid:
                ids.add(mid)
        return ids
    except Exception:
        return set()


def _apply_model_selection(provider: ModelProvider, orchestrator: str | None, cypher: str | None) -> None:
    # Fill defaults
    if orchestrator is None or cypher is None:
        defaults = MODEL_MAP.get(provider)
        if defaults:
            orchestrator = orchestrator or defaults["orchestrator"]
            cypher = cypher or defaults["cypher"]

    if provider == "local":
        # Try to ensure models actually exist on local endpoint; fallback if missing
        available = _list_local_models()
        # Prefer requested -> defaults -> common names
        candidate_order = [
            orchestrator or "",
            cypher or "",
            "gpt-oss:latest",
            "llama3:latest",
            "llama3",
        ]
        pick_orch = next((m for m in candidate_order if m and (not available or m in available)), orchestrator)
        pick_cyph = next((m for m in candidate_order if m and (not available or m in available)), cypher)
        if not pick_orch or not pick_cyph:
            logger.warning("No suitable local model found; using configured LOCAL_* defaults")
            pick_orch = pick_orch or settings.LOCAL_ORCHESTRATOR_MODEL_ID
            pick_cyph = pick_cyph or settings.LOCAL_CYPHER_MODEL_ID
        settings.set_orchestrator_model(pick_orch)
        settings.set_cypher_model(pick_cyph)
        return

    # cloud/openai paths
    if orchestrator:
        settings.set_orchestrator_model(orchestrator)
    if cypher:
        settings.set_cypher_model(cypher)


app = FastAPI(title="Graph-Code API", version="0.1")


static_dir = Path(__file__).resolve().parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Remember last successful ingest target for convenience
LAST_REPO_PATH: str | None = None


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html_path = static_dir / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h1>Graph-Code API</h1>")
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.post("/ingest")
def ingest(
    repo_path: str,
    clean: bool = False,
    export_json: str | None = None,
    provider: ModelProvider = Query("local", enum=["gemini", "local"]),
    orchestrator_model: str | None = None,
    cypher_model: str | None = None,
) -> JSONResponse:
    try:
        project_root = _setup_common_initialization(repo_path)
        _apply_model_selection(provider, orchestrator_model, cypher_model)

        with MemgraphIngestor(
            host=settings.MEMGRAPH_HOST, port=settings.MEMGRAPH_PORT
        ) as ingestor:
            if clean:
                ingestor.clean_database()
            ingestor.ensure_constraints()
            parsers, queries = load_parsers()
            updater = GraphUpdater(ingestor, project_root, parsers, queries)
            updater.run()

            if export_json:
                data = ingestor.export_graph_to_dict()
                Path(export_json).write_text(
                    __import__("json").dumps(data, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
        # Remember last repo on successful ingest
        global LAST_REPO_PATH
        LAST_REPO_PATH = str(project_root)
        return JSONResponse({"status": "ok", "repo": LAST_REPO_PATH})
    except Exception as e:
        logger.exception("Ingest failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
def models() -> JSONResponse:
    data = {
        "providers": {
            "local": {
                "orchestrator": ["gpt-oss:latest"],
                "cypher": ["gpt-oss:latest"],
            },
            "gemini": {
                "orchestrator": [
                    "gemini-2.5-pro",
                    "gemini-2.0-flash-thinking-exp-01-21",
                ],
                "cypher": [
                    "gemini-2.5-flash-lite-preview-06-17",
                    "gemini-2.0-flash-lite",
                ],
            },
        }
    }
    return JSONResponse(data)


@app.post("/ask")
async def ask(
    repo_path: str,
    question: str,
    provider: ModelProvider = Query("local", enum=["gemini", "local"]),
    orchestrator_model: str | None = None,
    cypher_model: str | None = None,
) -> JSONResponse:
    try:
        # Use remembered repo if none provided
        effective_repo = (repo_path or "").strip()
        if not effective_repo:
            if LAST_REPO_PATH:
                effective_repo = LAST_REPO_PATH
            else:
                raise HTTPException(status_code=400, detail="repo_path is required (no previous ingest found)")

        project_root = _setup_common_initialization(effective_repo)
        _apply_model_selection(provider, orchestrator_model, cypher_model)

        with MemgraphIngestor(
            host=settings.MEMGRAPH_HOST, port=settings.MEMGRAPH_PORT
        ) as ingestor:
            rag_agent = _initialize_services_and_agent(str(project_root), ingestor)
            response = await rag_agent.run(question, message_history=[])
            output = getattr(response, "output", None)
            # Normalize
            text = _extract_text_answer(output)
            if not text or _looks_like_fs_payload(text):
                # Defensive fallback: try once more with a simpler prompt
                logger.warning("Empty model response; retrying once with simplified prompt")
                response = await rag_agent.run(
                    (question.strip() or "Describe the project structure.") +
                    "\nPlease answer in plain text with a concise explanation.",
                    message_history=[]
                )
                output = getattr(response, "output", None) or ""
                text = _extract_text_answer(output)
            if not text:
                raise RuntimeError("Model returned empty response")
        return JSONResponse({"status": "ok", "answer": text, "repo": str(project_root)})
    except Exception as e:
        logger.exception("Ask failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/optimize")
async def optimize(
    repo_path: str,
    language: str,
    provider: ModelProvider = Query("local", enum=["gemini", "local"]),
    orchestrator_model: str | None = None,
    cypher_model: str | None = None,
    reference_document: str | None = None,
) -> JSONResponse:
    try:
        # Use remembered repo if none provided
        effective_repo = (repo_path or "").strip()
        if not effective_repo:
            if LAST_REPO_PATH:
                effective_repo = LAST_REPO_PATH
            else:
                raise HTTPException(status_code=400, detail="repo_path is required (no previous ingest found)")

        project_root = _setup_common_initialization(effective_repo)
        _apply_model_selection(provider, orchestrator_model, cypher_model)

        with MemgraphIngestor(
            host=settings.MEMGRAPH_HOST, port=settings.MEMGRAPH_PORT
        ) as ingestor:
            rag_agent = _initialize_services_and_agent(str(project_root), ingestor)
            instructions = (
                f"Analyze my {language} codebase and propose optimizations."
            )
            if reference_document:
                instructions += (
                    f" Use guidance from {reference_document} when proposing changes."
                )
            response = await rag_agent.run(instructions, message_history=[])
            output = getattr(response, "output", None) or ""
            text = _extract_text_answer(output) or ""
            if not text or _looks_like_fs_payload(text):
                logger.warning("Empty model response during optimize; retrying once")
                response = await rag_agent.run((instructions + "\nReturn a short bullet list in plain text.").strip(), message_history=[])
                output = getattr(response, "output", None) or ""
                text = _extract_text_answer(output) or ""
            if not text:
                raise RuntimeError("Model returned empty response")
        return JSONResponse({"status": "ok", "result": text, "repo": str(project_root)})
    except Exception as e:
        logger.exception("Optimize failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/fs/list")
def list_fs(path: str | None = None) -> JSONResponse:
    try:
        import os
        from pathlib import Path

        def list_drives() -> list[str]:
            drives: list[str] = []
            if os.name == "nt":
                for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                    drive = f"{letter}:\\"
                    if Path(drive).exists():
                        drives.append(drive)
            else:
                drives.append("/")
            return drives

        if not path:
            roots = list_drives()
            return JSONResponse({"current_path": None, "roots": roots, "entries": []})

        p = Path(path)
        if not p.exists() or not p.is_dir():
            raise ValueError("Path does not exist or is not a directory")

        entries: list[dict[str, str]] = []
        for child in sorted(p.iterdir()):
            if child.is_dir():
                entries.append({"name": child.name, "path": str(child), "type": "dir"})

        parent = str(p.parent) if p.parent != p else None
        return JSONResponse({
            "current_path": str(p),
            "parent": parent,
            "entries": entries,
        })
    except Exception as e:
        logger.exception("List FS failed")
        raise HTTPException(status_code=500, detail=str(e))


