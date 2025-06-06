FROM nvcr.io/nvidia/tritonserver:24.04-py3

# ---- Python deps for tokenizer backend ----
COPY triton_models/tokenizer/requirements.txt /tmp/req.txt
RUN pip install --no-cache-dir -r /tmp/req.txt
RUN rm /tmp/req.txt

# ---- copy model repository ----
COPY triton_models /models

# ENV NVIDIA_TF32_OVERRIDE=0
EXPOSE 8000 8001 8002
CMD ["tritonserver", "--model-repository=/models", "--log-verbose=0"]
