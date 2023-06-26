# Large Language Model Accelerator

[![Build Status](https://dev.azure.com/Adlik/GitHub/_apis/build/status/Adlik.model_optimizer?branchName=main)](https://dev.azure.com/Adlik/GitHub/_build/results?buildId=3472&view=results)
[![Bors enabled](https://bors.tech/images/badge_small.svg)](https://app.bors.tech/repositories/65566)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

LLMA is an end-to-end inference service framework, which can help developers efficiently deploy large language models both on cloud and embedded environment and accelerate inference process. 

## Support features

### 1.1 Support large language model inference

For the large language model such as LLAMA-7B, LLMA can deploy it on a seperate hardware in the server and do inference with client requests. Specifically, the client sends an inference request and LLMA receives the request and returns the inference result to the client.

## TUTORIALS

This example demonstrates how to use LLAM to deploy LLAMA-7B on Cloudblazer Yunsui t20.

### Setup

In a conda env with pytorch available, run:

```sh
cd LLMA/examples/llama-7b
pip install -r requirements.txt
```
### Deploy LLAMA-7B 

In the deployment environment, run:

```sh
cd LLMA/examples/llama-7b
bash ./run.sh $CKPT_DIR $TOKENIZER_PATH
```
### Do inference

Examples of model inference are as follows:

Infer with the python script.

In the LLMA/examples/llama-7b directory, run the command:

```sh
python3 client.py  
```

The results will be as follows:

```sh
{
  "code": 1, 
  "error": "", 
  "finish_reason": "null", 
  "model_name": "llama-7b", 
  "output": "I believe the meaning of life is to live to the fullest extent to help others and to grow spiritually developed through relationships and the expression of gratitude. It's all about who you're surrounded by and the ones who make you smile. I'm a hopeless"
}
```

## License

Apache License 2.0
