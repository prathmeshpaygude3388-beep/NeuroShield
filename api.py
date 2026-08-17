import logging
import os
import time
import threading

from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import List

import torch

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel, Field, field_validator

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from database import (
    init_db,
    create_scan,
    get_scans,
    clear_scans,
)


# ============================================================
# CONFIG
# ============================================================

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


MODEL_DIR = os.environ.get(
    "NEUROSHIELD_MODEL_DIR",
    "saved_models/german_bert_prompt_injection"
)

FRONTEND_DIR = os.environ.get(
    "NEUROSHIELD_FRONTEND_DIR",
    "frontend"
)

MAX_LENGTH = _env_int("NEUROSHIELD_MAX_LENGTH", 256)
MAX_CHARS = _env_int("NEUROSHIELD_MAX_CHARS", 8000)
MAX_BATCH_SIZE = _env_int("NEUROSHIELD_MAX_BATCH_SIZE", 32)

THRESHOLD = _env_float("NEUROSHIELD_THRESHOLD", 0.50)

RATE_LIMIT_REQUESTS = _env_int("NEUROSHIELD_RATE_LIMIT", 60)
RATE_LIMIT_WINDOW_SECONDS = _env_int("NEUROSHIELD_RATE_WINDOW", 60)

CORS_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("NEUROSHIELD_CORS_ORIGINS", "*").split(",")
]

APP_VERSION = "1.1.0"

START_TIME = time.time()


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=os.environ.get("NEUROSHIELD_LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-8s | neuroshield | %(message)s",
)

logger = logging.getLogger("neuroshield")


# ============================================================
# MODEL STATE
# ============================================================

class ModelState:
    """Holds the loaded model/tokenizer and readiness flag.

    Kept as a small stateful object (rather than module globals set at
    import time) so the model loads inside the app's lifespan — this
    means importing this module for tests, tooling, or docs generation
    doesn't also load a multi-hundred-MB model onto a device.
    """

    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.device = "cpu"
        self.ready = False
        # Guards concurrent forward passes through the single shared
        # model instance. This trades a little throughput for
        # correctness under concurrent requests.
        self.lock = threading.Lock()


state = ModelState()


# ============================================================
# STATS
# ============================================================

class Stats:

    def __init__(self):
        self.lock = threading.Lock()
        self.total = 0
        self.injections = 0
        self.safe = 0
        self.total_latency_ms = 0.0

    def record(self, injection: bool, latency_ms: float) -> None:

        with self.lock:

            self.total += 1
            self.total_latency_ms += latency_ms

            if injection:
                self.injections += 1
            else:
                self.safe += 1

    def snapshot(self) -> dict:

        with self.lock:

            average_latency = (
                self.total_latency_ms / self.total
                if self.total
                else 0.0
            )

            return {
                "total_scans": self.total,
                "threats_blocked": self.injections,
                "safe_allowed": self.safe,
                "average_latency_ms": round(average_latency, 2),
            }


stats = Stats()


# ============================================================
# RATE LIMITING
# ============================================================

class RateLimiter:
    """Simple in-memory sliding-window limiter, keyed per client IP.

    Good enough for a single-process deployment. For multi-worker or
    multi-instance deployments, swap this for a shared store (Redis).
    """

    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.hits = defaultdict(deque)
        self.lock = threading.Lock()

    def allow(self, key: str) -> bool:

        now = time.time()

        with self.lock:

            bucket = self.hits[key]

            while bucket and now - bucket[0] > self.window_seconds:
                bucket.popleft()

            if len(bucket) >= self.max_requests:
                return False

            bucket.append(now)
            return True


rate_limiter = RateLimiter(
    RATE_LIMIT_REQUESTS,
    RATE_LIMIT_WINDOW_SECONDS,
)


# ============================================================
# LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("========================================")
    logger.info("NEUROSHIELD — AI PROMPT DEFENSE")
    logger.info("========================================")

    init_db()
    logger.info("Scan history database initialized.")

    state.device = "cuda" if torch.cuda.is_available() else "cpu"

    try:

        logger.info("Loading German BERT model from '%s'...", MODEL_DIR)

        state.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

        state.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_DIR
        )

        state.model.to(state.device)
        state.model.eval()

        state.ready = True

        logger.info("Model loaded successfully.")
        logger.info("Device: %s", state.device)
        logger.info("Threshold: %s", THRESHOLD)

    except Exception:

        # Don't crash the process — let it come up and report itself
        # as unhealthy so an orchestrator can see and restart it,
        # rather than dying silently before binding a port.
        logger.exception(
            "Failed to load model. The API will start but report "
            "unhealthy until this is resolved."
        )

        state.ready = False

    yield

    logger.info("Shutting down NeuroShield.")


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="NeuroShield",
    description="AI-powered German prompt injection detection.",
    version=APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=512)


@app.middleware("http")
async def rate_limit_and_timing(request: Request, call_next):

    if request.url.path.startswith("/predict"):

        client_key = (
            request.client.host
            if request.client
            else "unknown"
        )

        if not rate_limiter.allow(client_key):

            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": "Rate limit exceeded. Please slow down."
                },
            )

    start = time.perf_counter()

    response = await call_next(request)

    elapsed_ms = (time.perf_counter() - start) * 1000

    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"

    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Invalid request.",
            "errors": exc.errors(),
        },
    )


if os.path.isdir(f"{FRONTEND_DIR}/assets"):

    app.mount(
        "/assets",
        StaticFiles(directory=f"{FRONTEND_DIR}/assets"),
        name="assets",
    )


# ============================================================
# REQUEST MODELS
# ============================================================

class PromptRequest(BaseModel):

    prompt: str = Field(..., min_length=1, max_length=MAX_CHARS)

    @field_validator("prompt")
    @classmethod
    def not_blank(cls, value: str) -> str:

        if not value.strip():
            raise ValueError("Prompt cannot be empty.")

        return value


class BatchPromptRequest(BaseModel):

    prompts: List[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_BATCH_SIZE,
    )

    @field_validator("prompts")
    @classmethod
    def no_blank_entries(cls, value: List[str]) -> List[str]:

        if any(not item.strip() for item in value):
            raise ValueError("Prompts cannot be empty.")

        return value


# ============================================================
# RESPONSE MODELS
# ============================================================

class PredictionResponse(BaseModel):

    prediction: str
    safe_probability: float
    injection_probability: float
    threshold: float
    action: str
    truncated: bool
    latency_ms: float


class BatchPredictionResponse(BaseModel):

    results: List[PredictionResponse]
    count: int
    total_latency_ms: float


class HealthResponse(BaseModel):

    status: str
    model_loaded: bool
    model: str
    threshold: float
    device: str
    uptime_seconds: float
    version: str


class StatsResponse(BaseModel):

    total_scans: int
    threats_blocked: int
    safe_allowed: int
    average_latency_ms: float


class HistoryRequest(BaseModel):

    prompt: str = Field(..., min_length=1, max_length=MAX_CHARS)
    prediction: str
    safe_probability: float
    injection_probability: float
    threshold: float
    action: str
    latency_ms: float


class ModelInfoResponse(BaseModel):

    model_dir: str
    device: str
    max_length: int
    threshold: float
    labels: dict


# ============================================================
# INFERENCE
# ============================================================

def _ensure_ready() -> None:

    if not state.ready:

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready yet. Check /health.",
        )


def _predict_single(prompt: str) -> dict:

    start = time.perf_counter()

    # --------------------------------------------------------
    # TRUNCATION CHECK
    # --------------------------------------------------------

    full_token_count = len(state.tokenizer.encode(prompt))
    truncated = full_token_count > MAX_LENGTH

    # --------------------------------------------------------
    # TOKENIZATION
    # --------------------------------------------------------

    encoded = state.tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )

    encoded = {
        key: value.to(state.device)
        for key, value in encoded.items()
    }

    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    with state.lock:

        with torch.no_grad():

            outputs = state.model(**encoded)

            probabilities = torch.softmax(
                outputs.logits,
                dim=1,
            )

    safe_probability = float(probabilities[0][0].item())
    injection_probability = float(probabilities[0][1].item())

    # --------------------------------------------------------
    # DECISION
    # --------------------------------------------------------

    if injection_probability >= THRESHOLD:
        prediction = "INJECTION"
        action = "BLOCK"
    else:
        prediction = "SAFE"
        action = "ALLOW"

    latency_ms = (time.perf_counter() - start) * 1000

    stats.record(prediction == "INJECTION", latency_ms)

    return {
        "prediction": prediction,
        "safe_probability": round(safe_probability, 6),
        "injection_probability": round(injection_probability, 6),
        "threshold": THRESHOLD,
        "action": action,
        "truncated": truncated,
        "latency_ms": round(latency_ms, 2),
    }


# ============================================================
# FRONTEND
# ============================================================

@app.get("/", include_in_schema=False)
def frontend():

    return FileResponse(f"{FRONTEND_DIR}/index.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():

    path = f"{FRONTEND_DIR}/favicon.ico"

    if os.path.isfile(path):
        return FileResponse(path)

    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ============================================================
# HEALTH
# ============================================================

@app.get("/health", response_model=HealthResponse)
def health():

    return {
        "status": "healthy" if state.ready else "degraded",
        "model_loaded": state.ready,
        "model": "German BERT",
        "threshold": THRESHOLD,
        "device": state.device,
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "version": APP_VERSION,
    }


# ============================================================
# MODEL INFO
# ============================================================

@app.get("/model/info", response_model=ModelInfoResponse)
def model_info():

    _ensure_ready()

    return {
        "model_dir": MODEL_DIR,
        "device": state.device,
        "max_length": MAX_LENGTH,
        "threshold": THRESHOLD,
        "labels": {"0": "SAFE", "1": "INJECTION"},
    }


# ============================================================
# STATS
# ============================================================

@app.get("/stats", response_model=StatsResponse)
def get_stats():

    return stats.snapshot()


# ============================================================
# SCAN HISTORY
# ============================================================

@app.get("/history")
def history(limit: int = 50):

    limit = max(1, min(limit, 200))

    return {
        "scans": get_scans(limit)
    }


@app.post("/history")
def save_history(request: HistoryRequest):

    return create_scan(
        prompt=request.prompt.strip(),
        prediction=request.prediction,
        safe_probability=request.safe_probability,
        injection_probability=request.injection_probability,
        threshold=request.threshold,
        action=request.action,
        latency_ms=request.latency_ms,
    )


@app.delete("/history")
def delete_history():

    clear_scans()

    return {
        "status": "cleared"
    }


# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PromptRequest):

    _ensure_ready()

    prompt = request.prompt.strip()

    if not prompt:

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Prompt cannot be empty.",
        )

    return _predict_single(prompt)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPromptRequest):

    _ensure_ready()

    cleaned = [item.strip() for item in request.prompts]

    results = [_predict_single(item) for item in cleaned]

    return {
        "results": results,
        "count": len(results),
        "total_latency_ms": round(
            sum(item["latency_ms"] for item in results),
            2,
        ),
    }