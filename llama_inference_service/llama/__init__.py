# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import importlib.util
import sys
import os 

from .generation import LLaMA
from .model import ModelArgs, Transformer
from .tokenizer import Tokenizer

def is_torch_gcu_available():
    if importlib.util.find_spec("torch_gcu") is None:
        return False
    if importlib.util.find_spec("torch_gcu.core") is None:
        return False
    return importlib.util.find_spec("torch_gcu.core.model") is not None
