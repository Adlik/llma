# Large Language Model Accelerator

[![Build Status](https://dev.azure.com/Adlik/GitHub/_apis/build/status/Adlik.model_optimizer?branchName=main)](https://dev.azure.com/Adlik/GitHub/_build/results?buildId=3472&view=results)
[![Bors enabled](https://bors.tech/images/badge_small.svg)](https://app.bors.tech/repositories/65566)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

LLMA is an inference service framework, which can help developers efficiently deploy large language models both on cloud and embedded environment and accelerate inference process.

## Support features

### 1.1 Support popular deep learning frameworks

LLMA supports almost training and inference frameworks, such as: TensorFLow, TensorRT, PyTorch, OpenVino, ONNX, etc. At the same time, LLMA provides an API that allows adding custom backends.

### 1.2 High performance model inferecne

LLMA supports model multiple instances, model concurrent execution, dynamic batch to maximize throughput and utilization. At the same time, LLMA supports model scheduling and can efficiently allocate hardware resources by grouping models from different frameworks.

### 1.3 Support large language model inference

For the large language modelLLMA such as GPT3-175B, LLMA can divide it into multiple smaller files and execute each file on a separate hardware in the server.




