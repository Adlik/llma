# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import List
import time
import numpy as np
import os 
import torch

from llama.tokenizer import Tokenizer
from llama.model import Transformer
from llama import is_torch_gcu_available

torch.autograd.set_detect_anomaly(True)

if is_torch_gcu_available():
    import torch_gcu
    import torch_gcu.distributed as dist
    local_rank = int(os.environ['LOCAL_RANK']) if 'LOCAL_RANK' in os.environ else dist.get_rank()
    #gcu_device = torch_gcu.gcu_device(local_rank * int(os.getenv("LEO_CLUSTER_NUM", '1')))
    gcu_device = torch_gcu.gcu_device(local_rank)
else:
    import torch as torch_gcu


class LLaMA:
    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.max_prompts_len = 32
        self.max_seq_len = 512

    def gen_mask_stage_0(self, tokens: torch.Tensor, pad_id: int):
        temp = torch.full((1, 1, self.max_prompts_len, self.max_prompts_len), -65500.0, device="cpu")
        temp = torch.triu(temp, diagonal=1)
        expand_tokens = tokens[:, None, None, :].expand(1, 1, self.max_prompts_len, self.max_prompts_len)
        temp.masked_fill_(expand_tokens == pad_id, -65500.0)
        temp[0,0,:,:].fill_diagonal_(fill_value = 0., wrap = False).reshape(1,1,self.max_prompts_len,self.max_prompts_len)
        mask = torch.full((1, 1, self.max_prompts_len, self.max_seq_len), -65500.0, device="cpu")
        mask[0, 0, :, -self.max_prompts_len:] = temp
        return mask.to(gcu_device)

    def gen_mask_stage_1(self, cur_pos: int):
        mask = torch.full((1, 1, 1, self.max_seq_len), -65500.0, device="cpu")
        mask[:, :, :, self.max_seq_len-cur_pos:] = 0
        return mask.to(gcu_device)

    def generate(
        self,
        prompts: List[str],
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> List[str]:
        bsz = len(prompts)
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)
        total_padding_len = params.max_seq_len
        if not is_torch_gcu_available():
            tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()

        else:
            tokens = torch.full((bsz, total_padding_len), 0,device="cpu")
            tokens = tokens.long()

        for k, t in enumerate(prompt_tokens):
            assert len(t) <= self.max_prompts_len, \
                f"prompt size of {prompts[k]}({len(t)}) is greater than max_prompts_len: {self.max_prompts_len}"
            if not is_torch_gcu_available():
                tokens[k, : len(t)] = torch.tensor(t).long()
            else:
                tokens[k,  -len(t):] = torch.tensor(t).long()
        start_pos = min_prompt_size
        prev_pos = 0
        token_time_list = list()
        for cur_pos in range(start_pos, total_len):
            start_time = time.time()
            if prev_pos == 0:
                mask = self.gen_mask_stage_0(tokens[:, -self.max_prompts_len:], 0);
                logits = self.model.forward(tokens[:, -self.max_prompts_len:].to(gcu_device), start_pos = prev_pos, mask=mask)
            else:
                mask = self.gen_mask_stage_1(cur_pos)
                logits = self.model.forward(tokens[:, -1:].to(gcu_device), start_pos = prev_pos, mask=mask)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)

            next_token = next_token.reshape(tokens.shape[0],-1).cpu()
            # only replace token if prompt has already been generated
            tokens = torch.cat([tokens,next_token],dim = 1)
            tokens = tokens[:, 1:]
            prev_pos = cur_pos
            end_time = time.time()
            token_time_list.append(end_time - start_time)

        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[-len(prompt_tokens[i]) - max_gen_len :]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))
        return decoded, token_time_list


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    # call sync_lived_tensor to avoid repeat computing in different subgraphs
    torch_gcu.sync_lived_tensor()
    itemp = probs_sort.cpu()
    probs_sum = torch.cumsum(itemp, dim=-1)
    probs_sum = probs_sum.to(gcu_device)
    mask = probs_sum - probs_sort > p
    #probs_sort[mask] = 0.0
    probs_sort.masked_fill_(mask, 0.0)
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
