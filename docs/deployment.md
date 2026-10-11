# Deployment (Local)

This project is local-first. There is no cloud deployment path baked in,
but the same Docker images could be pushed to any container registry and
run on a single VM with a GPU if desired.

## Docker Compose (recommended)

```bash
cp .env.example .env
docker compose up --build
```

This starts: `postgres`, `redis`, `backend` (FastAPI on :8000), `worker`
(Celery), `frontend` (served on :5173).

## Non-Docker local development

See the main [README](../README.md#installation) for the full manual
setup (Python venv, Postgres, Redis, Alembic, npm).

## GPU support

If a CUDA-capable GPU and drivers are available on the host:

1. Install GPU-enabled PyTorch (`pip install torch --index-url
   https://download.pytorch.org/whl/cu121`, adjust for your CUDA version)
   instead of the CPU wheel.
2. `pip install -r requirements-gpu.txt` for `bitsandbytes` (required for
   QLoRA's 4-bit quantization).
3. Torch will automatically detect and use the GPU — no config changes
   needed; `app/ml/training/hardware.py` reports the detected GPU in the
   UI and via `GET /api/system/info`.

For Docker, use an NVIDIA-CUDA base image for the `backend`/`worker`
services and add `deploy.resources.reservations.devices` (or
`--gpus all` with plain `docker run`) to pass the GPU through. This is
left out of the default `docker-compose.yml` so the project runs
out-of-the-box on machines without a GPU or without the NVIDIA Container
Toolkit installed.

## Environment variables

See `.env.example` for the full list. Nothing is hard-coded — database
URL, Redis URL, secret key, storage paths, and default models are all
configurable.
