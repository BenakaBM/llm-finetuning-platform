# LLM Fine-Tuning & Evaluation Platform

A local-first platform for uploading datasets, fine-tuning open-source
LLMs with LoRA/QLoRA, evaluating base vs fine-tuned models, tracking
experiments, and testing models in an interactive playground — built with
FastAPI, React, PyTorch, Hugging Face Transformers/PEFT, and Celery.

No fabricated metrics, no mocked training, no simulated evaluation
results: every number in the UI is computed by actually running the
model.

---

## Features

- **Datasets** — upload JSON/JSONL/CSV/Parquet, automatic format
  detection (instruction or conversational), real duplicate/missing-value/
  token-length analysis, a transparent quality score, and train/val/test
  splitting with versioning.
- **Model registry** — register any Hugging Face causal LM; weights are
  only downloaded when actually needed for training or inference.
- **Fine-tuning** — real LoRA and QLoRA fine-tuning via PEFT + Transformers,
  configurable hyperparameters, background execution via Celery so the API
  never blocks.
- **Live training monitoring** — Server-Sent Events stream of real loss/
  step/epoch as training runs, backed by a `training_metrics` table.
- **Evaluation** — ROUGE, BLEU, exact match, BERTScore, classification
  metrics, and an LLM-as-judge mode, all pluggable via a metric registry.
- **Base vs fine-tuned comparison** — every training run automatically
  evaluates both models on the same held-out test split.
- **Experiment tracking** — every run captures full config, metrics,
  hardware, duration, and artifacts.
- **Model comparison** — compare 2-4 experiments side by side.
- **Playground** — chat with any registered model, with side-by-side
  comparison.
- **HTML reports** — a self-contained report generated per experiment.
- **Resource validation** — estimates GPU memory requirements before
  training and warns (with concrete suggestions) instead of crashing.

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React, TypeScript, Vite, Tailwind CSS, React Query, React Router, Recharts |
| Backend | Python 3.11, FastAPI, Pydantic, SQLAlchemy, Alembic |
| Database | PostgreSQL (SQLite fallback works for the test suite) |
| Queue | Redis + Celery |
| ML | PyTorch, Transformers, PEFT, Accelerate, Datasets, Evaluate, rouge-score, sacrebleu, scikit-learn |
| Storage | Local filesystem (`storage/`) |

## Architecture

```
React Frontend → FastAPI → PostgreSQL / Redis → Celery Workers → Local Storage
```

See [docs/architecture.md](docs/architecture.md) for diagrams and the full
data flow from upload through report generation.

---

## Installation

### Prerequisites

- Python 3.11+
- Node.js 20+
- PostgreSQL 14+ (or use Docker Compose)
- Redis 7+ (or use Docker Compose)

### 1. Clone and configure

```bash
cp .env.example .env
```

### 2. Database setup

```bash
createdb llmft   # or use any Postgres instance; update DATABASE_URL in .env
```

Tables are auto-created on backend startup for local dev. To use real
migrations instead:

```bash
cd backend
alembic upgrade head
```

### 3. Backend (CPU setup)

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### GPU setup

```bash
pip install -r requirements.txt
pip install -r requirements-gpu.txt   # adds bitsandbytes for QLoRA
# Install a CUDA-enabled torch build matching your CUDA version, e.g.:
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

`GET /api/system/info` reports whether a GPU was detected.

### 4. Redis + Celery worker

```bash
redis-server                                     # in one terminal
cd backend
celery -A app.workers.celery_app worker --loglevel=info --concurrency=1
```

### 5. Frontend

```bash
cd frontend
npm install
npm run dev
```

Visit `http://localhost:5173`. The Vite dev server proxies `/api` to
`http://localhost:8000`.

### Docker (all-in-one)

```bash
cp .env.example .env
docker compose up --build
```

This starts Postgres, Redis, the backend, a worker, and the frontend.

---

## Demo Mode

A complete, self-contained pipeline run that works on CPU with no GPU:

```bash
python scripts/demo.py
```

This does **not** require Postgres, Redis, or FastAPI to be running — it
calls the same ML modules the app uses directly. It will:

1. Load and validate `examples/demo_instruction_dataset.jsonl`
2. Split it into train/validation/test
3. Load `sshleifer/tiny-gpt2` (a ~2MB model, chosen so training finishes
   in roughly a minute on CPU)
4. Run a real LoRA fine-tuning job (real gradient updates, real loss)
5. Evaluate the base model on the test split
6. Evaluate the fine-tuned model on the same test split
7. Compute and print the comparison
8. Save experiment artifacts to `storage/demo_run/`
9. Generate `storage/demo_run/report.html`

> **Note:** this requires internet access to download the tiny model
> from the Hugging Face Hub the first time it runs (subsequent runs use
> the local cache in `data/hf_cache/`). If you're running this inside a
> network-restricted sandbox, make sure `huggingface.co` is reachable.

---

## Dataset Format

**Instruction format:**
```json
{"instruction": "Explain machine learning", "input": "", "output": "Machine learning is..."}
```

**Conversational format:**
```json
{"messages": [{"role": "user", "content": "What is Python?"}, {"role": "assistant", "content": "Python is..."}]}
```

Both are normalized internally to `{instruction, input, output}` so the
rest of the pipeline only deals with one shape. See
[app/ml/datasets/loader.py](backend/app/ml/datasets/loader.py).

## Fine-Tuning

See [docs/training.md](docs/training.md) for the full LoRA/QLoRA
explanation, hyperparameters, and CPU-demo notes.

## Evaluation

See [docs/evaluation.md](docs/evaluation.md) for the metric registry,
LLM-as-judge details, and how base-vs-fine-tuned comparison works.

## Experiment Tracking & Model Comparison

Every training job automatically becomes an `Experiment` with full
config, hardware, training curves, and evaluation scores. Compare 2-4
experiments on the Experiments page, or via `POST /api/experiments/compare`.

## Playground

`/playground` lets you chat with any registered base or fine-tuned model,
with temperature/top-p/max-tokens controls and side-by-side comparison
between two models.

---

## Testing

```bash
cd backend
pip install -r requirements-dev.txt
pytest -q
```

The test suite (30 tests) covers auth, dataset validation/normalization,
evaluation metrics, hardware/resource estimation, and dataset API
integration. It runs against a SQLite fallback DB (no Postgres required)
and does not require downloading any large models — dataset/metric tests
use plain strings, and API tests use FastAPI's `TestClient`.

Frontend:
```bash
cd frontend
npm run test
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: psycopg2` | `pip install psycopg2-binary` (in requirements.txt) |
| `password cannot be longer than 72 bytes` from passlib | Ensure `bcrypt==4.0.1` is installed — newer bcrypt releases break passlib's backend detection |
| Training job stuck in `PENDING` | Make sure a Celery worker is running (`make worker`) and Redis is reachable |
| `OSError: couldn't connect to huggingface.co` | The base model needs to be downloaded on first use — check internet access, or use a model already in `data/hf_cache/` |
| Out of GPU memory | Enable QLoRA, reduce batch size, reduce max sequence length, or increase gradient accumulation steps — the platform will also proactively warn about this before starting the job |
| Frontend shows CORS errors | Check `CORS_ORIGINS` in `.env` includes your frontend's origin |

---

## Project Structure

```
llm-finetuning-platform/
├── frontend/           React + TS + Vite + Tailwind dashboard
├── backend/
│   ├── app/
│   │   ├── api/         FastAPI routers
│   │   ├── core/        config, security, logging, error handling
│   │   ├── db/          SQLAlchemy session + cross-dialect UUID type
│   │   ├── models/      SQLAlchemy ORM models
│   │   ├── schemas/     Pydantic request/response schemas
│   │   ├── services/    business logic / orchestration
│   │   ├── workers/     Celery app + tasks (training, evaluation)
│   │   ├── ml/
│   │   │   ├── datasets/    loading, normalization, validation, quality
│   │   │   ├── training/    LoRA/QLoRA trainer, hardware detection
│   │   │   ├── evaluation/  metrics registry, LLM-as-judge
│   │   │   └── inference/   text generation
│   │   └── main.py
│   ├── alembic/         real DB migrations
│   └── tests/           pytest suite (30 tests)
├── examples/             demo instruction dataset
├── scripts/demo.py       standalone end-to-end demo
├── docs/                 architecture, training, evaluation, API, deployment docs
├── docker-compose.yml
└── .env.example
```

## Future Improvements

- WebSocket-based live log tail (currently metrics stream via SSE; full
  stdout/stderr log tailing would need a log-shipping mechanism from the
  Celery worker process)
- Multi-GPU / distributed training via `accelerate launch`
- Full-parameter fine-tuning option alongside LoRA/QLoRA
- Dataset augmentation and synthetic data generation tools
- Role-based access control beyond the current USER/ADMIN split
- A proper frontend test suite beyond the current scaffold

## Resume Value

Developed a local-first LLM fine-tuning and evaluation platform using
FastAPI, React, PyTorch, Hugging Face Transformers, PEFT, and Celery,
enabling dataset management, LoRA/QLoRA fine-tuning, asynchronous
training, automated model evaluation, experiment tracking, model
comparison, and interactive inference.
