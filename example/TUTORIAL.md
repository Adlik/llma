# TUTORIAL

## Preparation

### Pull the docker image

In the server with Yunsui t20, run:

```sh
docker pull artifact.enflame.cn/enflame_docker_images/ubuntu/qic_ubuntu_1804_gcc7:latest
```

### Load LLaMA-7B

```sh
cd LLMA && \
mkdir tmp/llama-7b/7B && \
cd tmp/llama-7b/7B && \
wget https://llama-7b.oss-cn-beijing.aliyuncs.com/7B/ && \
cd .. && \
wget https://llama-7b.oss-cn-beijing.aliyuncs.com/tokenizer.model
```

### Run the image

Run the following command:

```sh
cd LLMA
docker run -it -v $PWD:/home/join/model --privileged -p 7999:8080 artifact.enflame.cn/enflame_docker_images/ubuntu/qic_ubuntu_1804_gcc7:latest bash
```

## Deploy LLaMA-7B

In the docker container, run:

```sh
cd /home/join/model/example/llama-7b
bash ./run.sh ../tmp/llama-7b/7B/ ../tmp/llama-7b/tokenizer.model
```

## Do inference

Infer with the python script.

In the LLMA/examples/llama-7b directory, run the command:

```sh
python3 client.py -u 'http://localhost:7999/chat' -p 'I believe the meaning of life'
```

The results will be as follows:

```sh
{
  "code": 1, 
  "error": "", 
  "finish_reason": "null", 
  "model_name": "LLaMA-7B", 
  "output": "I believe the meaning of life is to live to the fullest extent to help others."
}
```
