# IELTS-Band-Predictor 📊✍️

_End-to-end pipeline that scores **IELTS Writing Task 2** essays (band 4 – 9)._
Given a **PROMPT** and the candidate’s **ESSAY**, the model returns a floating-point band (e.g. `6.5`).

---

## 1 · Quick start

```bash
# clone and enter the repo
git clone https://github.com/you/ielts-band-predictor.git
cd ielts-band-predictor

# install runtime + dev dependencies into a Poetry venv
poetry install --with dev

# activate Git hooks (black, isort, flake8, prettier, etc.)
poetry run pre-commit install
```

> Run `poetry run pre-commit run -a` at any time – all hooks should be green ✅.

---

## 2 · Train the model

```bash
poetry run python -m ielts_band_predictor.scripts.train
```

- Hyper-parameters are managed by **Hydra** – override anything from the CLI:
  `model.lr=1e-5 datamodule.batch_size=32`.

- The best checkpoint is saved to `artifacts/checkpoints/best.ckpt` and symlinked
  as `best-epoch=…-val_mae=….ckpt`.

---

## 3 · Sanity-check on random essays

```bash
poetry run python -m ielts_band_predictor.scripts.eval_random \
  k=10                                           # sample 10 essays
```

Outputs band, prediction, absolute error and reports MAE / RMSE.

---

## 4 · Model export

| Step         | Command                                                                          | Result                 |
| ------------ | -------------------------------------------------------------------------------- | ---------------------- |
| **ONNX**     | `poetry run python -m ielts_band_predictor.scripts.export_onnx`                  | `artifacts/model.onnx` |
| **TensorRT** | _(inside NVIDIA container)_<br>`bash ielts_band_predictor/scripts/export_trt.sh` | `artifacts/model.plan` |

```bash
# fire up the TensorRT image
docker run --gpus all --rm -it -v $(pwd):/workspace \
  nvcr.io/nvidia/tensorrt:24.04-py3
# inside the shell:
bash ielts_band_predictor/scripts/export_trt.sh
```

---

## 5 · Serve with Triton Inference Server

### Docker image (all-in-one)

```bash
docker build -t ielts-triton:1.0 .
docker run --gpus '"device=0"' --rm \
  -p8010:8000  -p8011:8001  -p8012:8002 \
  ielts-triton:1.0
```

- **8000** – HTTP/REST  **8001** – gRPC  **8002** – Prometheus metrics
- Health-check: `curl localhost:8010/v2/health/ready → OK`

### docker-compose alternative

```bash
docker compose up --build -d
```

---

## 6 · Score an essay via REST

```bash
poetry run python -m ielts_band_predictor.scripts.client_test \
  server.url=http://localhost:8010 \
  k=3                      # evaluate 3 random essays
```

The script

1. samples essays from `data/raw/essays.jsonl`,
2. strips non-ASCII garbage,
3. sends them to Triton (`ielts_pipeline` ensemble),
4. prints predicted bands.

---

### Directory structure (abridged)

```
├─ ielts_band_predictor/
│  ├─ scripts/
│  │   ├─ train.py            # Hydra + Lightning
│  │   ├─ export_onnx.py
│  │   ├─ export_trt.sh
│  │   ├─ client_test.py      # REST → Triton
│  │   └─ eval_random.py
│  ├─ data.py                 # DataModule
│  └─ models.py               # Bert/Longformer regressors
├─ scripts/                   # helper notebooks, etc.
├─ configs/                   # Hydra YAMLs
├─ triton_models/             # tokenizer, engine, ensemble
└─ artifacts/                 # checkpoints, .onnx, .plan
```

---

## 7 · Useful one-liners

```bash
# check code style only
poetry run pre-commit run --all-files --show-diff-on-failure

# run unit tests
poetry run pytest -q

# reproduce latest DVC stage (dataset, onnx, trt…)
dvc repro

# open MLflow UI (if enabled in train.yaml)
mlflow ui --port 5000
```

---

Happy scoring ✨
