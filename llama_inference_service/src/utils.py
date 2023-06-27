# Copyright 2023 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple
import os
import json
from pathlib import Path
from itertools import chain
import torch

from ..model_parallel import initialize_model_parallel

from ..llama import ModelArgs, Transformer, Tokenizer, LLaMA
from ..llama import is_torch_gcu_available  # pylint: disable=no-name-in-module

torch.autograd.set_detect_anomaly(True)


if is_torch_gcu_available():
    import torch_gcu   # pylint: disable=import-error
else:
    import torch as torch_gcu  # pylint: disable=reimported


def setup_model_parallel() -> Tuple[int, int]:

    if not is_torch_gcu_available():
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        world_size = int(os.environ.get("WORLD_SIZE", -1))
        torch_gcu.distributed.init_process_group("nccl")
    else:
        local_rank = int(os.environ['LOCAL_RANK'])
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        torch_gcu.distributed.init_process_group("eccl", world_size=world_size, rank=rank)
    initialize_model_parallel(world_size)
    if not is_torch_gcu_available():
        torch_gcu.cuda.set_device(local_rank)

    torch.manual_seed(1)
    return local_rank, world_size


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    checkpoints = sorted(Path(ckpt_dir).glob("*.pt*"))
    checkpoint_num = len(checkpoints)
    indices = world_size // checkpoint_num
    ckpt_path = checkpoints[local_rank//indices]
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    with open(Path(ckpt_dir) / "params.json", "r") as file:
        params = json.loads(file.read())

    model_args: ModelArgs = ModelArgs(max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params)
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    if not is_torch_gcu_available():
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        model = Transformer(model_args)
    else:
        torch.set_default_tensor_type(torch.HalfTensor)
        model = Transformer(model_args)
        gcu_device = torch_gcu.gcu_device(local_rank)
        model = model.to(gcu_device)

    parallel_embeddings = ['tok_embeddings']
    parallel_column = ['attention.wq', 'attention.wk', 'attention.wv',
                       'feed_forward.w1', 'feed_forward.w3', 'output']
    parallel_row = ['attention.wo', 'feed_forward.w2']

    for name, state in chain(model.named_parameters(), model.named_buffers()):
        if name in checkpoint.keys():
            if any(ele in name for ele in parallel_embeddings):
                tensor_index = torch.tensor_split(checkpoint[name], indices, dim=1)[local_rank % indices]
                state.data = tensor_index.type_as(state.data).to(state.device)
            elif any(ele in name for ele in parallel_column):
                tensor_index = torch.tensor_split(checkpoint[name], indices, dim=0)[local_rank % indices]
                state.data = tensor_index.type_as(state.data).to(state.device)
            elif any(ele in name for ele in parallel_row):
                tensor_index = torch.tensor_split(checkpoint[name], indices, dim=1)[local_rank % indices]
                state.data = tensor_index.type_as(state.data).to(state.device)
            else:
                state.data = checkpoint[name].type_as(state.data).to(state.device)

    generator = LLaMA(model, tokenizer)
    return generator
