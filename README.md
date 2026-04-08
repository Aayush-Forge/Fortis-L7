# Fortis-L7 / DoWGuard-Env

OpenEnv hackathon benchmark for Layer-7 API traffic defense.

The core simulation is in:

- `env.py`
- `risk_engine.py`
- `data_generator.py`

OpenEnv wrapper/compliance layer is in:

- `openenv.yaml`
- `schemas.py`
- `tasks.py`
- `graders.py`
- `inference.py`
- `pyproject.toml`
- `server/app.py`

## Environment Model

Observation (vector length 5):

`[ip_reputation, velocity_score, entropy_level, navigation_path_index, jitter_value]`

Action space:

- `0` Allow
- `1` Throttle
- `2` Challenge
- `3` Block

Risk score:

`RS = 0.30*ip + 0.25*velocity + 0.20*entropy + 0.15*(1-navigation) + 0.10*jitter`

Classification:

- `RS >= 0.80` -> `hard_bot`
- `RS >= 0.60` -> `soft_bot`
- `RS >= 0.40` -> `ambiguous`
- `RS >= 0.20` -> `probably_human`
- else -> `verified_human`

## Local Setup (Windows / PowerShell)

```powershell
cd "C:\Users\User\Documents\RL env\Fortis-L7"
python -m pip install -r requirements.txt
openenv validate
```

Expected:

`[OK] Fortis-L7: Ready for multi-mode deployment`

## Run Inference

Set required environment variables:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

Then run:

```powershell
python inference.py
```

`inference.py` prints strict judge-friendly logs:

- `[START] ...`
- `[STEP] ...`
- `[END] ...`

## Docker

Build:

```powershell
cd "C:\Users\User\Documents\RL env\Fortis-L7"
docker build -t fortis-l7-env .
```

Run service:

```powershell
docker run --rm -p 7860:7860 fortis-l7-env
```

Health check:

- `http://localhost:7860/health`

## Hugging Face Spaces (Docker SDK)

Use this repository as Docker Space source.

Runtime expectations:

- container starts with `python server/app.py`
- listens on `PORT` (defaults to `7860`)
- health route: `/health`

Optional secret vars for inference in Space:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

## Notes

- Scores are normalized to `[0,1]` in graders.
- Core simulation logic is preserved; OpenEnv layers are additive.
- For deep implementation rationale, see `ai.md`.
