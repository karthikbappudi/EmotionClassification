from onnxruntime import (
    GraphOptimizationLevel,
    InferenceSession,
    SessionOptions,
)

from scipy.special import softmax
from transformers import AutoConfig
import numpy as np

pipeline_path = "./trained_models/pipe"
model_config = AutoConfig.from_pretrained(pipeline_path)


def create_model_for_provider(model_path, provider="CPUExecutionProvider"):
    options = SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    session = InferenceSession(str(model_path), options, providers=[provider])
    session.disable_fallback()
    return session


class OnnxPipeline:

    """We are replicating the huggingface piplines since onnx models are not compatible with huggingface pipline class."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, query):
        model_inputs = self.tokenizer(query, return_tensors="pt")
        inputs_onnx = {
            k: v.cpu().detach().numpy() for k, v in model_inputs.items()
        }
        logits = self.model.run(None, inputs_onnx)[0][0, :]
        probs = softmax(logits)
        pred_idx = np.argmax(probs).item()
        return [
            {
                "predcition": model_config.id2label[pred_idx],
                "score": str(probs[pred_idx]),
            }
        ]
