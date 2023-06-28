# Copyright 2023 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import torch
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from .utils import load, setup_model_parallel

APP = Flask(__name__)

CORS(APP)

local_rank, world_size = setup_model_parallel()


def construct_response(code, error, model_name, output, finish_reason="null"):
    response = {
        "code": code,
        "error": error,
        "model_name": model_name,
        "output": output,
        "finish_reason": finish_reason,
    }
    return response


@APP.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        data = json.loads(data)
        instruction = data['instruction']
        prompt = [[instruction]]
        model_name = data['model']
        generator = load(FLAGS.ckpt_dir, FLAGS.tokenizer_path, local_rank, world_size, 512, 1)
        with torch.no_grad():
            results = generator.generate(prompt[0], max_gen_len=50, temperature=0.8, top_p=0.95)
        response = make_response(jsonify(construct_response(1, "", model_name, results[0])))
        return response

    except KeyError as err:
        error = f"Param {str(err)} was not provided in the request"
        return jsonify(construct_response(0, error, "", [], ""))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--ckpt_dir', type=str, required=False)
    parser.add_argument('-t', '--tokenizer_path', type=str, required=False)
    FLAGS = parser.parse_args()
    APP.run(debug=True)
