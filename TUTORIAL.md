## TUTORIAL

This example demonstrates how to use LLAM to deploy LLaMA-7B on Cloudblazer Yunsui t20.

### Setup

In a conda env with pytorch available, run:

```sh
cd LLMA
pip install -e .
```
### Deploy LLaMA-7B on Cloudblazer Yunsui t20 

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
python3 client.py -u $url 
```

The results will be as follows:

```sh
{
  "code": 1, 
  "error": "", 
  "finish_reason": "null", 
  "model_name": "LLaMA-7B", 
  "output": "I believe the meaning of life is to live to the fullest extent to help others and to grow spiritually developed through relationships and the expression of gratitude. It's all about who you're surrounded by and the ones who make you smile. I'm a hopeless"
}
```