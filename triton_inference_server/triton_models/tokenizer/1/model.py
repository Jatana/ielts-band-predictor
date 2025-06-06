import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer

TOKENIZER = AutoTokenizer.from_pretrained("bert-base-uncased")
MAX_LEN = 512


class TritonPythonModel:
    def initialize(self, args):
        pass

    def execute(self, requests):
        responses = []
        for req in requests:
            essay_bytes = pb_utils.get_input_tensor_by_name(req, "TEXT").as_numpy()

            if essay_bytes.ndim == 2 and essay_bytes.shape[1] == 1:
                essay_bytes = essay_bytes.squeeze(1)

            texts = [t.decode("utf-8") for t in essay_bytes]

            enc = TOKENIZER(
                texts,
                truncation=True,
                padding="max_length",
                max_length=MAX_LEN,
                return_tensors="np",
            )
            ids = enc["input_ids"].astype(np.int64)
            mask = enc["attention_mask"].astype(np.int64)

            tens_ids = pb_utils.Tensor("input_ids", ids)
            tens_mask = pb_utils.Tensor("attention_mask", mask)
            responses.append(pb_utils.InferenceResponse([tens_ids, tens_mask]))
        return responses
