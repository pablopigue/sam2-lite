.PHONY: setup checkpoints lint format test eval bench train profile train-memory export bundle app api docker-build docker-run gate

MODEL ?= teacher
CONFIG ?= configs/train/smoke.yaml
SKIP_INFERENCE ?= false
CKPT ?=

setup:  ## Install the locked environment and the git hooks
	uv sync
	uv run pre-commit install

checkpoints:  ## Download and verify the SAM 2.1 teacher weights
	./scripts/download_checkpoints.sh

lint:  ## Check style without modifying files
	uv run ruff check .
	uv run ruff format --check .

format:  ## Auto-fix lint issues and format the code
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest

eval:  ## DAVIS 2017 val: inference + J&F, logged to MLflow (SKIP_INFERENCE=true reuses predictions)
	uv run python -m sam2lite.eval.run_vos --config configs/eval/davis_val.yaml \
		model.name=$(MODEL) skip_inference=$(SKIP_INFERENCE) $(if $(CKPT),model.ckpt=$(CKPT))

bench:  ## Per-frame latency (encoder + full pipeline, GPU and CPU), logged to MLflow
	uv run python -m sam2lite.bench.latency --config configs/bench/latency.yaml \
		model.name=$(MODEL) $(if $(CKPT),model.ckpt=$(CKPT))

train:  ## Distillation
	uv run python -m sam2lite.train.distill --config $(CONFIG)

profile:  ## Per-component CPU time of a video frame logged to MLflow
	uv run python -m sam2lite.bench.profile_pipeline --config configs/bench/latency.yaml \
		model.name=$(MODEL) $(if $(CKPT),model.ckpt=$(CKPT))

train-memory:  ## Distil the memory attention to K memory frames
	uv run python -m sam2lite.train.distill_memory --config $(CONFIG)

export:  ## Student encoder -> ONNX (1024 and 576), checked against PyTorch with ONNX Runtime
	uv run python -m sam2lite.export.onnx --config configs/export/onnx.yaml

bundle:  ## Final tracker -> checkpoints/bundle/sam2-lite (model.safetensors + config.yaml)
	uv run python -m sam2lite.export.bundle --config configs/export/bundle.yaml

app:  ## Gradio click-to-track demo on http://127.0.0.1:7860
	uv run --group app python app/app.py

api:  ## FastAPI service on http://127.0.0.1:8000
	uv run --group api uvicorn api.main:app --port 8000

docker-build:  ## CPU image of the API
	docker build -f api/Dockerfile -t sam2-lite-api .

docker-run:  ## Serve the API from the image, with the local bundle and ONNX files mounted read-only
	docker run --rm -p 8000:8000 -v $(PWD)/checkpoints:/app/checkpoints:ro sam2-lite-api

gate:  ## CI eval gate: J&F on 3 DAVIS val videos must not regress
	uv run python scripts/eval_gate.py
