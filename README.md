# IELTS-Band-Predictor 📊✍️

_End-to-end pipeline that scores **IELTS Writing Task 2** essays (band 4 – 9)._
Given a **PROMPT** and the candidate’s **ESSAY**, the model returns a floating-point band (e.g. `6.5`).

---

## 0 · Description

The dataset, scraped from [writing9.com](https://writing9.com), contains 11 000 IELTS Task-2 essays—exactly 1 000 for every half-band from 4.0 to 9.0. Before training, each text is passed through a lightweight filter that strips non-ASCII characters; if an essay loses more than a small threshold it is discarded.
For scoring, the Hugging Face checkpoint [**bert-base-uncased**](https://huggingface.co/google-bert/bert-base-uncased) is fine-tuned with a single linear head. The prompt and essay are concatenated as one input string—`PROMPT: <prompt> ESSAY: <essay>`. Ten of the twelve Transformer layers remain frozen, and the data are split 90 %/10 % (stratified) for training and validation. After five epochs (≈ 3 min on a single GPU) the model attains a validation MAE of ≈ 0.8 band, approaching the \~0.5-band consistency typical of human graders.

---

## 1 · Quick start

_Requires python 3.10 or higher._

```bash
# clone and enter the repo
git clone https://github.com/jatana/ielts-band-predictor.git
cd ielts-band-predictor

# install runtime + dev dependencies into a Poetry venv
poetry install

# activate Git hooks (black, isort, flake8, prettier, etc.)
poetry run pre-commit install
```

> Run `poetry run pre-commit run -a` at any time – all hooks should be green.

---

## 2 · Train the model

```bash
poetry run python -m ielts_band_predictor.scripts.train
```

- Hyper-parameters are managed by **Hydra**. Training parameters are located in `configs/train.yaml`.

- Displays graphs at MLFlow. Expects that the MLFlow server is already up. The address can be configured in `configs/logger/mlflow.yaml` (by default 127.0.0.1:8080).

- The best checkpoint is saved to `artifacts/checkpoints/best.ckpt`. All checkpoints can be found in `artifacts/checkpoints/`.

---

MLFlow server can be started by

```bash
poetry run mlflow server --host 127.0.0.1 --port 8080 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

## 3 · Sanity-check on random essays

```bash
poetry run python -m ielts_band_predictor.scripts.eval_random
```

Check trained model on random essays. Hyperparameters can be configured in `configs/eval_random.yaml`. Outputs band, prediction, absolute error and reports MAE / RMSE.

---

## 4 · Model export

To export .ckpt model to ONNX format run

```bash
poetry run python -m ielts_band_predictor.scripts.export_onnx
```

By default it will convert `artifacts/checkpoints/best.ckpt` to `artifacts/model.onnx`. To specify custom files modify `configs/export_onnx.yaml`.

To convert to TensorRT format run:

```bash
# fire up the TensorRT image
docker run --gpus all --rm -v "$(pwd)":/workspace \
  nvcr.io/nvidia/tensorrt:24.04-py3 \
  bash -c "cd /workspace && bash ielts_band_predictor/scripts/export_trt.sh"
```

Hyperparameters of exporting to TensorRT can be configured in the bash file.

---

## 5 · Serve with Triton Inference Server

The following command copies TensorRT model to the inference server and prepares the server for running.

```bash
poetry run python -m ielts_band_predictor.scripts.prepare_triton +src_plan=artifacts/model.plan
```

After that command the server is ready to be started.

### Docker image (all-in-one)

```bash
cd triton_inference_server
docker build -t ielts-triton:1.0 .
docker run --gpus '"device=0"' --rm \
  -p8010:8000  -p8011:8001  -p8012:8002 \
  ielts-triton:1.0
```

- **8010** – HTTP/REST  **8011** – gRPC  **8012** – Prometheus metrics
- Health-check: `curl localhost:8010/v2/health/ready → OK`

### docker-compose alternative

```bash
cd triton_inference_server
docker compose up --build -d
```

---

## 6 · Score an essay via REST

Do not forget to get back to the root directory before running the next command!

```bash
poetry run python -m ielts_band_predictor.scripts.client_test
```

The script

1. samples essays from `data/raw/essays.jsonl`,
2. strips non-ASCII garbage,
3. sends them to Triton (`ielts_pipeline` ensemble),
4. prints predicted bands.

Hyperparameters can be configured in `configs/infer.yaml`.

---
