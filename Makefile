.PHONY: setup backend frontend worker demo docker-up docker-down test lint migrate

setup:
	cd backend && python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
	cd frontend && npm install

backend:
	cd backend && uvicorn app.main:app --reload --port 8000

frontend:
	cd frontend && npm run dev

worker:
	cd backend && celery -A app.workers.celery_app worker --loglevel=info --concurrency=1

demo:
	python3 scripts/demo.py

docker-up:
	docker compose up --build

docker-down:
	docker compose down

test:
	cd backend && pytest -q

migrate:
	cd backend && alembic upgrade head

migration:
	cd backend && alembic revision --autogenerate -m "$(m)"
