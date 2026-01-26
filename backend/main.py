from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import re
import shutil
import subprocess
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select as sa_select

from . import storage
from . import config as cfg
from .config import (
    CLIPROXY_API_KEY,
    CLIPROXY_BASE_URL,
    COUNCIL_MODELS,
    DEFAULT_CONSOLIDATOR,
    DISCOVERED_MODEL_IDS,
    EXPORTS_DIR,
    ensure_data_dirs,
    set_discovered_model_ids,
)
from .council import QEEGCouncilWorkflow
from .cliproxy_status import status_payload
from .exports import render_markdown_to_pdf
from .llm_client import AsyncOpenAICompatClient, UpstreamError
from .patient_files import save_patient_file_upload
from .reports import extract_pdf_full, extract_text_from_pdf, report_dir as report_storage_dir, save_report_upload


app = FastAPI(title="qEEG Council API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1):517\d$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PatientCreate(BaseModel):
    label: str = Field(min_length=1)
    notes: str = ""


class PatientUpdate(BaseModel):
    label: str = Field(min_length=1)
    notes: str = ""


class RunCreate(BaseModel):
    patient_id: str
    report_id: str
    council_model_ids: list[str]
    consolidator_model_id: str


class SelectRequest(BaseModel):
    artifact_id: str | None = None
    stage_num: int | None = None
    model_id: str | None = None
    kind: str | None = None


class _EventBroker:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._subs: dict[str, set[asyncio.Queue[dict[str, Any]]]] = {}

    async def publish(self, run_id: str, payload: dict[str, Any]) -> None:
        async with self._lock:
            queues = list(self._subs.get(run_id, set()))
        for q in queues:
            try:
                q.put_nowait(payload)
            except Exception:
                pass

    async def subscribe(self, run_id: str) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        async with self._lock:
            self._subs.setdefault(run_id, set()).add(q)
        return q

    async def unsubscribe(self, run_id: str, q: asyncio.Queue[dict[str, Any]]) -> None:
        async with self._lock:
            subs = self._subs.get(run_id)
            if not subs:
                return
            subs.discard(q)
            if not subs:
                self._subs.pop(run_id, None)


class CliproxyStartRequest(BaseModel):
    config_path: str | None = None
    force_restart: bool = False


class CliproxyLoginRequest(BaseModel):
    mode: Literal["login", "claude", "codex", "gemini"] = "login"
    project_id: str | None = None
    config_path: str | None = None


class CliproxyInstallRequest(BaseModel):
    # Set true if Homebrew can't find the formula without tapping.
    tap_router_for_me: bool = False


def _repo_root() -> Path:
    # backend/main.py -> backend/ -> repo root
    return Path(__file__).resolve().parents[1]


_PORTAL_PATIENT_ID_RE = re.compile(r"^(?P<mm>\d{2})-(?P<dd>\d{2})-(?P<yyyy>\d{4})-(?P<n>\d+)$")


def _normalize_portal_patient_id(value: str) -> str | None:
    raw = (value or "").strip()
    m = _PORTAL_PATIENT_ID_RE.match(raw)
    if not m:
        return None
    mm = int(m.group("mm"))
    dd = int(m.group("dd"))
    yyyy = int(m.group("yyyy"))
    n = int(m.group("n"))
    if mm < 1 or mm > 12:
        return None
    if dd < 1 or dd > 31:
        return None
    if yyyy < 1900 or yyyy > 2100:
        return None
    if n < 0 or n > 999:
        return None
    return f"{mm:02d}-{dd:02d}-{yyyy:04d}-{n}"


def _portal_patients_dir() -> Path:
    configured = os.getenv("QEEG_PORTAL_PATIENTS_DIR")
    if configured:
        return Path(configured).expanduser()
    return cfg.DATA_DIR / "portal_patients"


def _ensure_portal_patient_folder(label: str) -> None:
    patient_id = _normalize_portal_patient_id(label)
    if patient_id is None:
        return
    try:
        base = _portal_patients_dir()
        base.mkdir(parents=True, exist_ok=True)
        (base / patient_id).mkdir(parents=True, exist_ok=True)
    except Exception:
        # Best-effort only; don't fail core API operations on local folder creation.
        return


def _safe_portal_filename(value: str, *, fallback: str = "upload.bin") -> str:
    raw = os.path.basename(str(value or "").strip())
    cleaned = re.sub(r"[/\\\u0000-\u001F\u007F]+", "_", raw).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned[:200]
    return cleaned or fallback


def _publish_file_to_portal_folder(*, patient_label: str, src_path: Path, filename: str) -> Path | None:
    patient_id = _normalize_portal_patient_id(patient_label)
    if patient_id is None:
        return None

    try:
        out_dir = _portal_patients_dir() / patient_id
        out_dir.mkdir(parents=True, exist_ok=True)
        dest_path = out_dir / _safe_portal_filename(filename)

        tmp_path = dest_path.with_name(f".{dest_path.name}.partial")
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

        ext = dest_path.suffix.lower()
        use_hardlink = ext in {".mp4", ".pdf", ".zip", ".docx", ".rtf"}

        if dest_path.exists() and dest_path.is_file():
            try:
                dest_path.unlink()
            except Exception:
                pass

        if use_hardlink:
            try:
                os.link(src_path, dest_path)
                return dest_path
            except Exception:
                # Fall back to copy.
                pass

        shutil.copy2(src_path, tmp_path)
        tmp_path.replace(dest_path)
        return dest_path
    except Exception:
        # Best-effort only; don't fail core API operations on portal publishing.
        return None


def _default_clipr_config_path() -> str:
    if os.getenv("CLIPROXY_CONFIG"):
        return os.path.expanduser(os.getenv("CLIPROXY_CONFIG", ""))
    local_cfg = _repo_root() / ".cli-proxy-api" / "cliproxyapi.conf"
    if local_cfg.exists():
        return str(local_cfg)
    brew_cfg = Path("/opt/homebrew/etc/cliproxyapi.conf")
    if brew_cfg.exists():
        return str(brew_cfg)
    return os.path.expanduser("~/.cli-proxy-api/config.yaml")


def _start_detached(args: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as f:
        proc = subprocess.Popen(
            args,
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return int(proc.pid)


def _cliproxy_bin() -> str:
    return shutil.which("cliproxyapi") or "/opt/homebrew/bin/cliproxyapi"


def _ensure_project_clipr_config() -> Path:
    """
    Creates a project-local CLIProxyAPI config that disables client API-key auth (single-user localhost)
    and stores OAuth tokens under the project folder to avoid cross-project confusion.
    """
    cfg = _repo_root() / ".cli-proxy-api" / "cliproxyapi.conf"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    if not cfg.exists():
        content = """host: \"127.0.0.1\"
port: 8317
auth-dir: \".cli-proxy-api/auth\"
api-keys: []
"""
        cfg.write_text(content, encoding="utf-8")
    return cfg


def _sync_home_auth_to_project() -> int:
    """
    Copies OAuth token JSON files from ~/.cli-proxy-api into ./.cli-proxy-api/auth
    so the project-local proxy sees all logged-in providers.
    Returns the number of newly-copied files.
    """
    copied = 0
    try:
        src = Path.home() / ".cli-proxy-api"
        dst = _repo_root() / ".cli-proxy-api" / "auth"
        if not src.exists():
            return 0
        dst.mkdir(parents=True, exist_ok=True)
        for p in src.glob("*.json"):
            target = dst / p.name
            if target.exists():
                continue
            target.write_bytes(p.read_bytes())
            copied += 1
    except Exception:
        return copied
    return copied


def _port_from_base_url(base_url: str) -> int | None:
    try:
        # http://127.0.0.1:8317
        host_port = base_url.split("://", 1)[1]
        if "/" in host_port:
            host_port = host_port.split("/", 1)[0]
        if ":" not in host_port:
            return 80
        return int(host_port.rsplit(":", 1)[1])
    except Exception:
        return None


def _kill_cliproxy_listener(port: int) -> list[int]:
    """
    Best-effort: kill cliproxyapi processes listening on the given TCP port.
    Returns list of killed PIDs.
    """
    killed: list[int] = []
    try:
        out = subprocess.check_output(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
            text=True,
        ).strip()
    except Exception:
        return killed
    pids = []
    for line in out.splitlines():
        try:
            pids.append(int(line.strip()))
        except Exception:
            continue
    for pid in pids:
        try:
            cmd = subprocess.check_output(["ps", "-p", str(pid), "-o", "comm="], text=True).strip()
            if "cliproxyapi" not in cmd:
                continue
            os.kill(pid, 15)
            killed.append(pid)
        except Exception:
            continue
    return killed


def _run_detached_brew_install(log_path: Path, *, tap_router_for_me: bool) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    script_lines = []
    if tap_router_for_me:
        script_lines.append("brew tap router-for-me/tap || true")
    script_lines.append("brew install cliproxyapi")
    script = "\n".join(script_lines) + "\n"
    with log_path.open("ab") as f:
        proc = subprocess.Popen(
            ["bash", "-lc", script],
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return int(proc.pid)


def _get_mock_llm_client() -> AsyncOpenAICompatClient | None:
    """
    Create a mock LLM client for testing if QEEG_MOCK_LLM env var is set.
    Returns None if not in mock mode.
    """
    if not os.getenv("QEEG_MOCK_LLM"):
        return None

    try:
        from backend.tests.fixtures.mock_llm import create_mock_transport, MOCK_MODEL_IDS
    except ImportError:
        # Tests not available (e.g., production deployment)
        return None

    transport = create_mock_transport()
    try:
        timeout_s = float(os.getenv("QEEG_LLM_TIMEOUT_S", "600") or "600")
    except Exception:
        timeout_s = 600.0
    client = AsyncOpenAICompatClient(
        base_url="http://mock-cliproxy",
        api_key="",
        timeout_s=timeout_s,
        transport=transport,
    )
    # Pre-set discovered models for mock mode
    set_discovered_model_ids(MOCK_MODEL_IDS)
    return client


async def _refresh_discovered_models(*, llm: AsyncOpenAICompatClient) -> None:
    try:
        discovered = await llm.list_models()
    except Exception:
        return
    set_discovered_model_ids(discovered)


async def _model_refresh_loop(*, llm: AsyncOpenAICompatClient, interval_s: float) -> None:
    # Default: weekly refresh to pick up CLIProxyAPI model catalog updates.
    if interval_s <= 0:
        return
    while True:
        await asyncio.sleep(interval_s)
        await _refresh_discovered_models(llm=llm)


def _model_visible_in_ui(model_id: str) -> bool:
    """
    Reduce clutter in the frontend model picker by hiding older provider versions.

    Policy (as requested):
    - OpenAI: hide gpt-* models older than 5.1
    - Anthropic: hide claude-* models older than 4.5
    - Gemini: hide gemini-* models older than 3.x

    Unknown/unparsed model ids are left visible.
    """

    mid = (model_id or "").strip()
    if not mid:
        return False
    lower = mid.lower()

    # OpenAI (gpt-*)
    openai_id = lower.removeprefix("openai/")
    m = re.match(r"^gpt-(?P<major>\d+)(?:\.(?P<minor>\d+))?", openai_id)
    if m:
        major = int(m.group("major"))
        minor = int(m.group("minor") or "0")
        return major > 5 or (major == 5 and minor >= 1)

    # Anthropic (claude-*)
    anthropic_id = lower.removeprefix("anthropic/")
    if anthropic_id.startswith("claude-"):
        parts = anthropic_id.split("-")

        # New format:
        # - claude-<family>-4-5-YYYYMMDD
        # - claude-<family>-4-YYYYMMDD
        if (
            len(parts) >= 4
            and parts[0] == "claude"
            and parts[1] in {"opus", "sonnet", "haiku"}
            and parts[2].isdigit()
        ):
            major = int(parts[2])
            minor = 0
            # Minor is present only when the 4th token is a small digit and the 5th token looks like a date.
            if len(parts) >= 5 and parts[3].isdigit() and len(parts[3]) <= 2 and parts[4].isdigit() and len(parts[4]) >= 6:
                minor = int(parts[3])
            return major > 4 or (major == 4 and minor >= 5)

        # Older format: claude-3-7-sonnet-YYYYMMDD
        if len(parts) >= 5 and parts[0] == "claude" and parts[1].isdigit() and parts[2].isdigit():
            major = int(parts[1])
            minor = int(parts[2])
            return major > 4 or (major == 4 and minor >= 5)

        # If we can't parse, keep it visible.
        return True

    # Gemini (gemini-*)
    gemini_id = lower.removeprefix("google/")
    m = re.match(r"^gemini-(?P<major>\d+)(?:\.(?P<minor>\d+))?", gemini_id)
    if m:
        major = int(m.group("major"))
        # minor unused for threshold (3.x+ only)
        return major >= 3

    return True


@app.on_event("startup")
async def _startup() -> None:
    ensure_data_dirs()
    storage.init_db()
    _ensure_project_clipr_config()
    _sync_home_auth_to_project()

    # Check for mock mode (for testing)
    mock_client = _get_mock_llm_client()
    if mock_client is not None:
        app.state.llm = mock_client
        app.state.mock_mode = True
    else:
        try:
            timeout_s = float(os.getenv("QEEG_LLM_TIMEOUT_S", "600") or "600")
        except Exception:
            timeout_s = 600.0
        if timeout_s <= 0:
            timeout_s = 600.0
        app.state.llm = AsyncOpenAICompatClient(
            base_url=CLIPROXY_BASE_URL, api_key=CLIPROXY_API_KEY, timeout_s=timeout_s
        )
        app.state.mock_mode = False

    app.state.workflow = QEEGCouncilWorkflow(llm=app.state.llm)
    app.state.broker = _EventBroker()
    app.state.cliproxy_pid = None
    app.state.model_refresh_task = None

    if not app.state.mock_mode:
        await _refresh_discovered_models(llm=app.state.llm)
        try:
            interval_s = float(os.getenv("QEEG_MODEL_REFRESH_INTERVAL_S", str(7 * 24 * 60 * 60)) or "0")
        except Exception:
            interval_s = 7 * 24 * 60 * 60
        if interval_s < 0:
            interval_s = 0
        app.state.model_refresh_task = asyncio.create_task(_model_refresh_loop(llm=app.state.llm, interval_s=interval_s))


@app.on_event("shutdown")
async def _shutdown() -> None:
    task = getattr(app.state, "model_refresh_task", None)
    if task is not None:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    llm: AsyncOpenAICompatClient | None = getattr(app.state, "llm", None)
    if llm is not None:
        await llm.aclose()


@app.get("/api/health")
async def health() -> dict[str, Any]:
    llm: AsyncOpenAICompatClient = app.state.llm
    mock_mode: bool = getattr(app.state, "mock_mode", False)

    # In mock mode, always return healthy
    if mock_mode:
        from backend.tests.fixtures.mock_llm import MOCK_MODEL_IDS
        return {
            **status_payload(
                base_url="http://mock-cliproxy",
                reachable=True,
                discovered_model_count=len(MOCK_MODEL_IDS),
                error=None,
            ),
            "mock_mode": True,
        }

    try:
        discovered = await llm.list_models()
        set_discovered_model_ids(discovered)
        return status_payload(
            base_url=CLIPROXY_BASE_URL,
            reachable=True,
            discovered_model_count=len(discovered),
            error=None,
        )
    except Exception as e:
        return status_payload(
            base_url=CLIPROXY_BASE_URL,
            reachable=False,
            discovered_model_count=0,
            error=e,
        )


@app.post("/api/cliproxy/start")
async def cliproxy_start(req: CliproxyStartRequest) -> dict[str, Any]:
    """
    Best-effort helper for local single-user setups:
    starts CLIProxyAPI as a detached process and writes logs under data/.
    """
    llm: AsyncOpenAICompatClient = app.state.llm
    try:
        discovered = await llm.list_models()
        set_discovered_model_ids(discovered)
        return {"ok": True, "already_running": True}
    except Exception:
        pass

    _sync_home_auth_to_project()

    if req.force_restart:
        port = _port_from_base_url(CLIPROXY_BASE_URL) or 8317
        killed = _kill_cliproxy_listener(port)
    else:
        killed = []

    config_path = (req.config_path or _default_clipr_config_path()).strip()
    expanded = Path(os.path.expanduser(config_path))
    if not expanded.exists():
        # Prefer generating a project-local config that disables api-key auth.
        expanded = _ensure_project_clipr_config()

    log_path = Path("data") / "cliproxyapi.log"
    args = [_cliproxy_bin(), "-config", str(expanded)]

    try:
        pid = _start_detached(args, log_path)
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="'cliproxyapi' not found (try Homebrew install)")
    app.state.cliproxy_pid = pid
    return {
        "ok": True,
        "pid": pid,
        "log_path": str(log_path),
        "command": args,
        "used_config": str(expanded) if expanded.exists() else None,
        "killed_pids": killed,
    }


@app.post("/api/cliproxy/login")
async def cliproxy_login(req: CliproxyLoginRequest) -> dict[str, Any]:
    """
    Best-effort helper for local single-user setups:
    launches CLIProxyAPI login flows (which may open a browser / prompt in terminal depending on provider).
    """
    _sync_home_auth_to_project()

    config_path = (req.config_path or _default_clipr_config_path()).strip()
    expanded = Path(os.path.expanduser(config_path))
    if not expanded.exists():
        expanded = _ensure_project_clipr_config()

    args = [_cliproxy_bin(), "-config", str(expanded)]
    if req.mode == "login":
        args.append("-login")
    elif req.mode == "claude":
        args.append("-claude-login")
    elif req.mode == "codex":
        args.append("-codex-login")
    elif req.mode == "gemini":
        args.append("-login")
        if req.project_id:
            args.extend(["-project_id", req.project_id])

    log_path = Path("data") / "cliproxy_login.log"
    try:
        pid = _start_detached(args, log_path)
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="'cliproxyapi' not found (try Homebrew install)")
    return {"ok": True, "pid": pid, "log_path": str(log_path), "command": args}


@app.post("/api/cliproxy/install")
async def cliproxy_install(req: CliproxyInstallRequest) -> dict[str, Any]:
    """
    Best-effort helper for local single-user macOS setups:
    installs CLIProxyAPI via Homebrew in the background and logs to data/.
    """
    try:
        # cheap check
        subprocess.run(["brew", "--version"], check=True, capture_output=True, text=True)
    except Exception:
        raise HTTPException(status_code=400, detail="'brew' not found or not usable")

    log_path = Path("data") / "cliproxy_install.log"
    pid = _run_detached_brew_install(log_path, tap_router_for_me=req.tap_router_for_me)
    return {"ok": True, "pid": pid, "log_path": str(log_path)}


@app.get("/api/models")
async def models() -> dict[str, Any]:
    discovered = sorted(DISCOVERED_MODEL_IDS)
    ui_models = [mid for mid in discovered if _model_visible_in_ui(mid)]
    configured = []
    for m in COUNCIL_MODELS:
        configured.append(
            {
                "id": m.id,
                "name": m.name,
                "source": m.source,
                "endpoint_preference": m.endpoint_preference,
                "available": m.id in DISCOVERED_MODEL_IDS if DISCOVERED_MODEL_IDS else False,
            }
        )
    return {
        "discovered_models": discovered,
        "ui_models": ui_models,
        "configured_models": configured,
    }


@app.get("/api/patients")
async def list_patients() -> list[dict[str, Any]]:
    with storage.session_scope() as session:
        pts = storage.list_patients(session)
        patient_ids = [p.id for p in pts]
        patient_ids_with_video: set[str] = set()
        if patient_ids:
            q = (
                sa_select(storage.PatientFile.patient_id)
                .where(
                    storage.PatientFile.patient_id.in_(patient_ids),
                    storage.PatientFile.mime_type == "video/mp4",
                )
                .distinct()
            )
            patient_ids_with_video = set(session.scalars(q).all())

        return [_patient_out(p, has_explainer_video=(p.id in patient_ids_with_video)) for p in pts]


@app.post("/api/patients")
async def create_patient(req: PatientCreate) -> dict[str, Any]:
    with storage.session_scope() as session:
        p = storage.create_patient(session, label=req.label, notes=req.notes)
        _ensure_portal_patient_folder(p.label)
        return _patient_out(p)


@app.post("/api/patients/bulk_upload")
async def bulk_upload_patients(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    """
    Bulk upload qEEG report files. Each file creates a new patient whose label is the filename stem.

    If a patient with the same label already exists (case-insensitive), the file is skipped and reported.
    """
    created: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    seen_labels: set[str] = set()
    for file in files:
        filename = (file.filename or "upload").strip() or "upload"
        patient_label = Path(filename).stem.strip()
        if not patient_label:
            errors.append({"filename": filename, "error": "Empty filename stem (cannot derive patient label)"})
            continue

        label_key = patient_label.lower()
        if label_key in seen_labels:
            skipped.append(
                {
                    "filename": filename,
                    "patient_label": patient_label,
                    "reason": "duplicate_label_in_batch",
                }
            )
            continue
        seen_labels.add(label_key)

        with storage.session_scope() as session:
            existing = storage.find_patients_by_label(session, patient_label)
            if existing:
                skipped.append(
                    {
                        "filename": filename,
                        "patient_label": patient_label,
                        "reason": "patient_label_exists",
                        "existing_patient_ids": [p.id for p in existing],
                    }
                )
                continue

        report_id = str(uuid.uuid4())
        report_folder: Path | None = None
        try:
            file_bytes = await file.read()
            preview = ""

            with storage.session_scope() as session:
                patient = storage.Patient(label=patient_label, notes="")
                session.add(patient)
                session.flush()
                patient_id = patient.id

                _ensure_portal_patient_folder(patient_label)

                report_folder = report_storage_dir(patient_id, report_id)
                original_path, extracted_path, mime_type, preview = save_report_upload(
                    patient_id=patient_id,
                    report_id=report_id,
                    filename=filename,
                    provided_mime_type=file.content_type,
                    file_bytes=file_bytes,
                )

                report = storage.Report(
                    id=report_id,
                    patient_id=patient_id,
                    filename=filename,
                    mime_type=mime_type,
                    stored_path=str(original_path),
                    extracted_text_path=str(extracted_path),
                )
                session.add(report)
                session.commit()
                session.refresh(patient)
                session.refresh(report)

            created.append(
                {
                    "filename": filename,
                    "patient": _patient_out(patient),
                    "report": _report_out(report),
                    "preview": preview,
                }
            )
        except Exception as e:
            if report_folder is not None:
                try:
                    shutil.rmtree(report_folder, ignore_errors=True)
                except Exception:
                    pass
            errors.append({"filename": filename, "patient_label": patient_label, "error": str(e)})

    return {
        "created": created,
        "skipped": skipped,
        "errors": errors,
        "counts": {"created": len(created), "skipped": len(skipped), "errors": len(errors)},
    }


@app.get("/api/patients/{patient_id}")
async def get_patient(patient_id: str) -> dict[str, Any]:
    with storage.session_scope() as session:
        p = storage.get_patient(session, patient_id)
        if p is None:
            raise HTTPException(status_code=404, detail="Patient not found")
        q = (
            sa_select(storage.PatientFile.id)
            .where(storage.PatientFile.patient_id == patient_id, storage.PatientFile.mime_type == "video/mp4")
            .limit(1)
        )
        has_video = session.scalars(q).first() is not None
        return _patient_out(p, has_explainer_video=has_video)


@app.put("/api/patients/{patient_id}")
async def update_patient(patient_id: str, req: PatientUpdate) -> dict[str, Any]:
    with storage.session_scope() as session:
        p = storage.update_patient(session, patient_id, label=req.label, notes=req.notes)
        if p is None:
            raise HTTPException(status_code=404, detail="Patient not found")
        _ensure_portal_patient_folder(p.label)
        q = (
            sa_select(storage.PatientFile.id)
            .where(storage.PatientFile.patient_id == patient_id, storage.PatientFile.mime_type == "video/mp4")
            .limit(1)
        )
        has_video = session.scalars(q).first() is not None
        return _patient_out(p, has_explainer_video=has_video)


@app.post("/api/patients/{patient_id}/reports")
async def upload_report(patient_id: str, file: UploadFile = File(...)) -> dict[str, Any]:
    with storage.session_scope() as session:
        p = storage.get_patient(session, patient_id)
        if p is None:
            raise HTTPException(status_code=404, detail="Patient not found")

    report_id = str(uuid.uuid4())
    file_bytes = await file.read()
    original_path, extracted_path, mime_type, preview = save_report_upload(
        patient_id=patient_id,
        report_id=report_id,
        filename=file.filename or "upload",
        provided_mime_type=file.content_type,
        file_bytes=file_bytes,
    )

    with storage.session_scope() as session:
        report = storage.create_report(
            session,
            report_id=report_id,  # Use the same ID used for file storage
            patient_id=patient_id,
            filename=file.filename or "upload",
            mime_type=mime_type,
            stored_path=original_path,
            extracted_text_path=extracted_path,
        )
        return {"report_id": report.id, "preview": preview, "report": _report_out(report)}


@app.get("/api/patients/{patient_id}/reports")
async def list_reports(patient_id: str) -> list[dict[str, Any]]:
    with storage.session_scope() as session:
        p = storage.get_patient(session, patient_id)
        if p is None:
            raise HTTPException(status_code=404, detail="Patient not found")
        reps = storage.list_reports(session, patient_id)
        return [_report_out(r) for r in reps]


@app.get("/api/patients/{patient_id}/files")
async def list_patient_files(patient_id: str) -> list[dict[str, Any]]:
    with storage.session_scope() as session:
        p = storage.get_patient(session, patient_id)
        if p is None:
            raise HTTPException(status_code=404, detail="Patient not found")
        files = storage.list_patient_files(session, patient_id)
        return [_patient_file_out(f) for f in files]


@app.post("/api/patients/{patient_id}/files")
async def upload_patient_file(patient_id: str, file: UploadFile = File(...)) -> dict[str, Any]:
    with storage.session_scope() as session:
        p = storage.get_patient(session, patient_id)
        if p is None:
            raise HTTPException(status_code=404, detail="Patient not found")
        patient_label = p.label

    file_id = str(uuid.uuid4())
    filename = file.filename or "upload"
    original_path, mime_type, size_bytes = save_patient_file_upload(
        patient_id=patient_id,
        file_id=file_id,
        filename=filename,
        provided_mime_type=file.content_type,
        src=file.file,
    )

    with storage.session_scope() as session:
        pf = storage.create_patient_file(
            session,
            file_id=file_id,
            patient_id=patient_id,
            filename=filename,
            mime_type=mime_type,
            size_bytes=size_bytes,
            stored_path=original_path,
        )
        portal_path = _publish_file_to_portal_folder(
            patient_label=patient_label,
            src_path=Path(pf.stored_path),
            filename=pf.filename,
        )
        return {"file": _patient_file_out(pf), "portal_published_path": str(portal_path) if portal_path else None}


@app.get("/api/patient_files/{file_id}")
async def get_patient_file(file_id: str):
    with storage.session_scope() as session:
        pf = storage.get_patient_file(session, file_id)
        if pf is None:
            raise HTTPException(status_code=404, detail="File not found")
        path = Path(pf.stored_path)
        if not path.exists():
            raise HTTPException(status_code=404, detail="File not found on disk")
        return FileResponse(
            str(path),
            media_type=pf.mime_type,
            filename=pf.filename,
            content_disposition_type="inline",
        )


@app.delete("/api/patient_files/{file_id}")
async def delete_patient_file(file_id: str) -> dict[str, Any]:
    with storage.session_scope() as session:
        pf = storage.get_patient_file(session, file_id)
        if pf is None:
            raise HTTPException(status_code=404, detail="File not found")
        stored_path = Path(pf.stored_path)

    with storage.session_scope() as session:
        deleted = storage.delete_patient_file(session, file_id)
        if deleted is None:
            raise HTTPException(status_code=404, detail="File not found")

    try:
        # Delete the whole per-file folder (…/<patient_id>/<file_id>/…)
        shutil.rmtree(stored_path.parent, ignore_errors=True)
    except Exception:
        pass

    return {"ok": True}


@app.get("/api/reports/{report_id}/extracted")
async def get_report_extracted(report_id: str):
    with storage.session_scope() as session:
        report = storage.get_report(session, report_id)
        if report is None:
            raise HTTPException(status_code=404, detail="Report not found")
        extracted_path = Path(report.extracted_text_path)
        report_dir = extracted_path.parent

        # Prefer the enhanced multi-source extraction when present so the UI preview never shows only a
        # single (potentially noisy) OCR engine.
        enhanced_path = report_dir / "extracted_enhanced.txt"
        path = enhanced_path if enhanced_path.exists() and enhanced_path.stat().st_size > 0 else extracted_path
        if not path.exists():
            raise HTTPException(status_code=404, detail="Extracted text not found")
        if path == enhanced_path:
            # Older artifacts may label Apple Vision OCR as "VISION OCR" which is easy to confuse with a
            # multimodal "vision model". Normalize labels at read-time for clarity without requiring a re-extract.
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
                normalized = (
                    text.replace("--- VISION OCR ---", "--- APPLE VISION OCR ---")
                    .replace("--- OCR TEXT ---", "--- TESSERACT OCR ---")
                    .replace("--- OCR ---", "--- TESSERACT OCR ---")
                )
                if normalized != text:
                    return PlainTextResponse(normalized)
            except Exception:
                pass
        return FileResponse(str(path), media_type="text/plain", filename="extracted.txt")


@app.post("/api/reports/{report_id}/reextract")
async def reextract_report(report_id: str) -> dict[str, Any]:
    with storage.session_scope() as session:
        report = storage.get_report(session, report_id)
        if report is None:
            raise HTTPException(status_code=404, detail="Report not found")
        original_path = Path(report.stored_path)
        extracted_path = Path(report.extracted_text_path)

    if original_path.suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Re-extract only supported for PDFs")

    # Best-effort: also regenerate enhanced OCR text + page images for multimodal Stage 1.
    # IMPORTANT: write into the same folder as extracted_path (some older reports used a
    # different folder id than report_id).
    report_dir = extracted_path.parent
    enhanced_chars: int | None = None
    page_images_written = 0
    try:
        full = extract_pdf_full(original_path)
        enhanced_text = full.enhanced_text
        page_images = full.page_images
        # Keep extracted.txt aligned with enhanced extraction so UI/verification always sees the full union.
        extracted_path.write_text(enhanced_text, encoding="utf-8")
        enhanced_path = report_dir / "extracted_enhanced.txt"
        enhanced_path.write_text(enhanced_text, encoding="utf-8")
        enhanced_chars = len(enhanced_text)

        pages_dir = report_dir / "pages"
        pages_dir.mkdir(parents=True, exist_ok=True)
        for img in page_images:
            if not isinstance(img, dict):
                continue
            page_num = img.get("page")
            b64_png = img.get("base64_png")
            if not isinstance(page_num, int) or not isinstance(b64_png, str):
                continue
            try:
                img_bytes = base64.b64decode(b64_png)
                (pages_dir / f"page-{page_num}.png").write_bytes(img_bytes)
                page_images_written += 1
            except Exception:
                continue

        # Save per-page source text (audit/debug)
        sources_dir = report_dir / "sources"
        sources_dir.mkdir(parents=True, exist_ok=True)
        for p in full.per_page_sources:
            page_num = p.get("page")
            if not isinstance(page_num, int):
                continue
            try:
                (sources_dir / f"page-{page_num}.pypdf.txt").write_text(p.get("pypdf_text", ""), encoding="utf-8")
                (sources_dir / f"page-{page_num}.pymupdf.txt").write_text(p.get("pymupdf_text", ""), encoding="utf-8")
                (sources_dir / f"page-{page_num}.apple_vision.txt").write_text(p.get("vision_ocr_text", ""), encoding="utf-8")
                (sources_dir / f"page-{page_num}.tesseract.txt").write_text(p.get("tesseract_ocr_text", ""), encoding="utf-8")
            except Exception:
                continue

        metadata = dict(full.metadata)
        metadata.update(
            {
                "has_enhanced_ocr": True,
                "has_page_images": page_images_written > 0,
                "page_images_written": page_images_written,
                "sources_dir": "sources",
            }
        )
        (report_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    except Exception:
        # Fallback: regenerate extracted.txt with basic extraction (no multimodal assets).
        text = extract_text_from_pdf(original_path)
        extracted_path.write_text(text, encoding="utf-8")
        return {
            "ok": True,
            "chars": len(text),
            "enhanced_chars": None,
            "page_images_written": 0,
        }

    return {"ok": True, "chars": enhanced_chars or 0, "enhanced_chars": enhanced_chars, "page_images_written": page_images_written}


@app.get("/api/reports/{report_id}/original")
async def get_report_original(report_id: str):
    """Serve the original PDF file for a report."""
    with storage.session_scope() as session:
        report = storage.get_report(session, report_id)
        if report is None:
            raise HTTPException(status_code=404, detail="Report not found")
        original_path = Path(report.stored_path)

    if not original_path.exists():
        raise HTTPException(status_code=404, detail="Original file not found on disk")

    # Determine media type from extension
    suffix = original_path.suffix.lower()
    media_type = "application/pdf" if suffix == ".pdf" else "application/octet-stream"

    return FileResponse(
        str(original_path),
        media_type=media_type,
        filename=original_path.name,
    )


@app.get("/api/reports/{report_id}/pages")
async def list_report_pages(report_id: str) -> dict[str, Any]:
    """List available extracted page images for a report."""
    with storage.session_scope() as session:
        report = storage.get_report(session, report_id)
        if report is None:
            raise HTTPException(status_code=404, detail="Report not found")
        extracted_path = Path(report.extracted_text_path)

    # Pages are stored in the same directory as extracted text
    report_dir = extracted_path.parent
    pages_dir = report_dir / "pages"

    if not pages_dir.exists():
        return {"pages": [], "total": 0}

    # Find all page-N.png files
    pages: list[dict[str, Any]] = []
    for png_file in sorted(pages_dir.glob("page-*.png")):
        # Extract page number from filename like "page-0.png"
        try:
            page_num = int(png_file.stem.split("-")[1])
            pages.append({
                "page": page_num,
                "url": f"/api/reports/{report_id}/pages/{page_num}",
            })
        except (IndexError, ValueError):
            continue

    # Sort by page number
    pages.sort(key=lambda p: p["page"])

    return {"pages": pages, "total": len(pages)}


@app.get("/api/reports/{report_id}/pages/{page_num}")
async def get_report_page(report_id: str, page_num: int):
    """Serve a single extracted page image."""
    with storage.session_scope() as session:
        report = storage.get_report(session, report_id)
        if report is None:
            raise HTTPException(status_code=404, detail="Report not found")
        extracted_path = Path(report.extracted_text_path)

    report_dir = extracted_path.parent
    page_path = report_dir / "pages" / f"page-{page_num}.png"

    if not page_path.exists():
        raise HTTPException(status_code=404, detail=f"Page {page_num} not found")

    return FileResponse(
        str(page_path),
        media_type="image/png",
        filename=f"page-{page_num}.png",
    )


@app.get("/api/reports/{report_id}/metadata")
async def get_report_metadata(report_id: str) -> dict[str, Any]:
    """Get extraction metadata for a report."""
    with storage.session_scope() as session:
        report = storage.get_report(session, report_id)
        if report is None:
            raise HTTPException(status_code=404, detail="Report not found")
        extracted_path = Path(report.extracted_text_path)

    report_dir = extracted_path.parent
    metadata_path = report_dir / "metadata.json"

    if not metadata_path.exists():
        # Return basic info if no metadata file
        return {
            "report_id": report_id,
            "has_metadata_file": False,
        }

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["report_id"] = report_id
        metadata["has_metadata_file"] = True
        return metadata
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read metadata: {e}")


@app.get("/api/patients/{patient_id}/runs")
async def list_runs(patient_id: str) -> list[dict[str, Any]]:
    with storage.session_scope() as session:
        p = storage.get_patient(session, patient_id)
        if p is None:
            raise HTTPException(status_code=404, detail="Patient not found")
        runs = storage.list_runs(session, patient_id)
        return [_run_out(r) for r in runs]


@app.post("/api/runs")
async def create_run(req: RunCreate) -> dict[str, Any]:
    if not req.council_model_ids:
        raise HTTPException(status_code=400, detail="council_model_ids must be non-empty")
    if not req.consolidator_model_id:
        raise HTTPException(status_code=400, detail="consolidator_model_id is required")

    with storage.session_scope() as session:
        if storage.get_patient(session, req.patient_id) is None:
            raise HTTPException(status_code=404, detail="Patient not found")
        if storage.get_report(session, req.report_id) is None:
            raise HTTPException(status_code=404, detail="Report not found")
        run = storage.create_run(
            session,
            patient_id=req.patient_id,
            report_id=req.report_id,
            council_model_ids=req.council_model_ids,
            consolidator_model_id=req.consolidator_model_id,
        )
        return _run_out(run)


@app.post("/api/runs/{run_id}/start")
async def start_run(run_id: str) -> dict[str, Any]:
    with storage.session_scope() as session:
        run = storage.get_run(session, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        if run.status == "running":
            return _run_out(run)

    broker: _EventBroker = app.state.broker
    workflow: QEEGCouncilWorkflow = app.state.workflow

    async def _runner() -> None:
        async def on_event(payload: dict[str, Any]) -> None:
            await broker.publish(run_id, payload)

        await workflow.run_pipeline(run_id, on_event=on_event)

    asyncio.create_task(_runner())
    return {"ok": True}


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str) -> dict[str, Any]:
    with storage.session_scope() as session:
        run = storage.get_run(session, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return _run_out(run)


@app.get("/api/runs/{run_id}/artifacts")
async def get_artifacts(run_id: str) -> list[dict[str, Any]]:
    with storage.session_scope() as session:
        run = storage.get_run(session, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        arts = storage.list_artifacts(session, run_id)

    out = []
    for a in arts:
        content = Path(a.content_path).read_text(encoding="utf-8", errors="replace")
        out.append(
            {
                "id": a.id,
                "run_id": a.run_id,
                "stage_num": a.stage_num,
                "stage_name": a.stage_name,
                "model_id": a.model_id,
                "kind": a.kind,
                "content_type": a.content_type,
                "created_at": a.created_at.isoformat(),
                "content": content,
            }
        )
    return out


@app.get("/api/runs/{run_id}/stream")
async def stream(run_id: str):
    broker: _EventBroker = app.state.broker
    q = await broker.subscribe(run_id)

    async def gen():
        try:
            # initial snapshot
            with storage.session_scope() as session:
                run = storage.get_run(session, run_id)
                if run is not None:
                    yield _sse({"run_id": run_id, "type": "snapshot", "run": _run_out(run)})

            while True:
                try:
                    payload = await asyncio.wait_for(q.get(), timeout=15.0)
                    yield _sse(payload)
                except asyncio.TimeoutError:
                    yield ":\n\n"  # heartbeat
        finally:
            await broker.unsubscribe(run_id, q)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.post("/api/runs/{run_id}/select")
async def select(run_id: str, req: SelectRequest) -> dict[str, Any]:
    with storage.session_scope() as session:
        run = storage.get_run(session, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")

        artifact_id = req.artifact_id
        if artifact_id is None:
            if req.stage_num is None or req.model_id is None or req.kind is None:
                raise HTTPException(status_code=400, detail="Provide artifact_id or (stage_num, model_id, kind)")
            art = storage.find_artifact(
                session, run_id=run_id, stage_num=req.stage_num, model_id=req.model_id, kind=req.kind
            )
            if art is None:
                raise HTTPException(status_code=404, detail="Artifact not found")
            artifact_id = art.id

        run2 = storage.select_artifact(session, run_id, artifact_id)
        if run2 is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return _run_out(run2)


@app.post("/api/runs/{run_id}/export")
async def export(run_id: str) -> dict[str, Any]:
    with storage.session_scope() as session:
        run = storage.get_run(session, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        patient = storage.get_patient(session, run.patient_id)
        patient_label = patient.label if patient is not None else ""
        if not run.selected_artifact_id:
            raise HTTPException(status_code=400, detail="No selected artifact for run")
        art = session.get(storage.Artifact, run.selected_artifact_id)
        if art is None:
            raise HTTPException(status_code=400, detail="Selected artifact not found")
        md = Path(art.content_path).read_text(encoding="utf-8", errors="replace")

    export_dir = EXPORTS_DIR / run_id
    export_dir.mkdir(parents=True, exist_ok=True)
    md_path = export_dir / "final.md"
    pdf_path = export_dir / "final.pdf"
    md_path.write_text(md, encoding="utf-8")
    render_markdown_to_pdf(md, pdf_path)

    portal_md = _publish_file_to_portal_folder(
        patient_label=patient_label,
        src_path=md_path,
        filename=f"{patient_label}.md" if patient_label else "final.md",
    )
    portal_pdf = _publish_file_to_portal_folder(
        patient_label=patient_label,
        src_path=pdf_path,
        filename=f"{patient_label}.pdf" if patient_label else "final.pdf",
    )

    return {
        "ok": True,
        "final_md": str(md_path),
        "final_pdf": str(pdf_path),
        "portal_final_md": str(portal_md) if portal_md else None,
        "portal_final_pdf": str(portal_pdf) if portal_pdf else None,
    }


@app.get("/api/runs/{run_id}/export/final.md")
async def get_final_md(run_id: str):
    path = EXPORTS_DIR / run_id / "final.md"
    if not path.exists():
        raise HTTPException(status_code=404, detail="final.md not found")
    return FileResponse(str(path), media_type="text/markdown", filename="final.md")


@app.get("/api/runs/{run_id}/export/final.pdf")
async def get_final_pdf(run_id: str):
    path = EXPORTS_DIR / run_id / "final.pdf"
    if not path.exists():
        raise HTTPException(status_code=404, detail="final.pdf not found")
    return FileResponse(str(path), media_type="application/pdf", filename="final.pdf")


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _patient_out(p: storage.Patient, *, has_explainer_video: bool = False) -> dict[str, Any]:
    return {
        "id": p.id,
        "label": p.label,
        "notes": p.notes,
        "has_explainer_video": bool(has_explainer_video),
        "created_at": p.created_at.isoformat(),
        "updated_at": p.updated_at.isoformat(),
    }


def _report_out(r: storage.Report) -> dict[str, Any]:
    return {
        "id": r.id,
        "patient_id": r.patient_id,
        "filename": r.filename,
        "mime_type": r.mime_type,
        "created_at": r.created_at.isoformat(),
    }


def _patient_file_out(f: storage.PatientFile) -> dict[str, Any]:
    return {
        "id": f.id,
        "patient_id": f.patient_id,
        "filename": f.filename,
        "mime_type": f.mime_type,
        "size_bytes": f.size_bytes,
        "created_at": f.created_at.isoformat(),
    }


def _run_out(r: storage.Run) -> dict[str, Any]:
    try:
        council_model_ids = json.loads(r.council_model_ids_json)
    except Exception:
        council_model_ids = []
    try:
        label_map = json.loads(r.label_map_json or "{}")
    except Exception:
        label_map = {}

    return {
        "id": r.id,
        "patient_id": r.patient_id,
        "report_id": r.report_id,
        "status": r.status,
        "error_message": r.error_message,
        "council_model_ids": council_model_ids,
        "consolidator_model_id": r.consolidator_model_id,
        "label_map": label_map,
        "started_at": r.started_at.isoformat() if r.started_at else None,
        "completed_at": r.completed_at.isoformat() if r.completed_at else None,
        "selected_artifact_id": r.selected_artifact_id,
        "created_at": r.created_at.isoformat(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
