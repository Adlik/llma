# Copyright 2023 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import requests


MODEL_NAME = 'LLaMA-7B'
PROMPT = 'I believe the meaning of life is'
info = {
    "instruction": PROMPT,
    "model": MODEL_NAME
}
headers = {"Content-Type": "application/json"}


def get_response():
    response = requests.post(FLAGS.url, json=json.dumps(info), headers=headers)
    if response.status_code == 200:
        for chunk in response.iter_content(chunk_size=1):
            if chunk:
                print(chunk.decode('utf-8'), end='')
    else:
        print('Error:', response.status_code, response.reason)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--url', type=str, required=False)
    FLAGS = parser.parse_args()
    get_response()
