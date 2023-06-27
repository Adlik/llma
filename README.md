# Large Language Model Accelerator

[![Bors enabled](https://bors.tech/images/badge_small.svg)](https://app.bors.tech/repositories/65566)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

LLMA is an end-to-end optimizing framework for large language models.

The goal of LLMA is to accelerate large language models inference process both on cloud and embedded environment.

With LLMA framework, different large language models can be deployed to different platforms with high performance in a flexible and easy way.

## Support features

### 1.1 Support large language model inference

For the large language model such as LLaMA-7B, LLMA can deploy it on different hardwares like NVIDIA GPU and Cloudblazer Yunsui t20.

LLMA supports doing inference with client requests. Specifically, the client sends an inference request and LLMA returns the inference result to the client.

### 1.2 Support large language model optimization

LLMA supports several optimizing technologies like model fine-tuning and model quantization.

## Getting Started

The following tutorial demonstrates how to use LLAM to deploy LLaMA-7B.

- [Tutorial](example/TUTORIAL.md)

## License

Apache License 2.0
