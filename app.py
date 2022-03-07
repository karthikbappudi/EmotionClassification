import flask
import time
from flask import Flask
from flask import request
from onnxruntime import (
    GraphOptimizationLevel,
    InferenceSession,
    SessionOptions,
)

import onnx_model
from transformers import pipeline, AutoConfig, AutoTokenizer


app = Flask(__name__)


# loading the onnx model
onnx_model_path = "./trained_models/pipe/model.quant.onnx"
onnx_quantized_model = onnx_model.create_model_for_provider(onnx_model_path)

# loading the tokenizer
pipeline_path = "./trained_models/pipe"
tokenizer = AutoTokenizer.from_pretrained(pipeline_path)

# instantiating the onnx pipeline
onnx_quat_pipe = onnx_model.OnnxPipeline(onnx_quantized_model, tokenizer)


@app.route("/encode", methods=["POST"])
def predict():
    """
    This function takes text as input and returns predictions from the model and assosiated softmax score to it.
    """
    queries = request.json["text"]

    predictions = onnx_quat_pipe(queries)
    response = predictions
    return flask.jsonify(response)


if __name__ == "__main__":

    print("loaded the model")

    app.run()
