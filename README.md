# Reasoning Experiments

This repository contains scripts and notebooks for studying reasoning, refusal behavior, and response generation across open and API-backed language models. The code focuses on harmful/harmless prompt datasets, think versus no-think generation, residual collection, safety judging, and pass@k-style response analysis.

## Repository Layout

- `src/reasoning/`: importable project utilities and the main harmful response generator module.
- `scripts/inference/`: response generation and model-run orchestration scripts.
- `scripts/eval/`: judging, labeling, and pass@k evaluation scripts.
- `scripts/analysis/`: residual extraction and token/trajectory analysis scripts.
- `notebooks/`: exploratory notebooks, grouped by topic.
- `artifacts/`: ignored local folder for datasets, generated responses, residual tensors, plots, model caches, and other large outputs.
- `external/`: ignored local folder for copied or cloned third-party repositories, if needed.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env` for GPT-4o labeling or analysis. Set `HF_TOKEN` if you need gated Hugging Face models.

## Data And Artifacts

Datasets and generated outputs are intentionally not tracked. Place local datasets under:

```text
artifacts/data/
artifacts/data/splits/harmful_test.json
artifacts/data/splits/harmless_test.json
```

Generated responses, residual tensors, figures, and model caches should stay under `artifacts/`. The default scripts use repo-root paths such as `artifacts/responses/`, `artifacts/results/`, `artifacts/residuals/`, and `artifacts/hf_cache/`.

## Running Scripts

Run commands from the repository root.

```bash
python -m reasoning.harmful_response_generator
python scripts/inference/generate_responses.py
python scripts/inference/gaslight_generation.py
python scripts/eval/label_responses.py
python scripts/eval/safety_judge.py
python scripts/analysis/residual_generation.py
```

Most scripts have configuration constants near the bottom of the file. Check dataset paths, model names, sample counts, and whether GPT-4o analysis is enabled before running. Heavy model generation and evaluation can require substantial GPU memory and API usage.

## Ignored Content

The repo ignores local secrets, generated model outputs, datasets, checkpoints, residual tensors, plots, logs, model caches, virtual environments, notebook checkpoints, and copied external repositories. This keeps GitHub focused on runnable code and documentation while preserving local experiment artifacts in `artifacts/`.
