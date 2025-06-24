import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import trimesh
from meto.ss_engine import Engine
import logging


def get_tokenizer_silksong(resolution=128, ss_mode=4, meta_init_data={}, structure_limit={}, debugging=False):

    tokenizer = Engine(discrete_bins=resolution, mode=ss_mode, debugging=debugging, meta_init_data=meta_init_data, structure_limit=structure_limit)
    vocab_size = tokenizer.num_word_table # C U E I0 I1 Out B O
    
    return tokenizer, vocab_size

def tokenize_mesh_ss(tokenizer, vertices, faces, non_mani_process=True):
    tokens_ori = tokenizer.encode(vertices, faces, non_mani_process)
    tokens, meta_data = tokenizer.token_encode(tokens_ori, tokenizer.mode)
    return tokens, meta_data

def detokenize_mesh_ss(tokenizer, tokens, colorful=False, mani_fix=False):
    vertices, faces = tokenizer.decode(tokens, tokenizer.discrete_bins, tokenizer.mode, colorful, mani_fix)
    return vertices, faces
    
def quantize_num_faces_ss(n):
    # 0: <=0, un cond
    # 1: 0-1000, low-poly
    # 2: 1000-2000, mid-poly
    # 3: 2000-4000, high-poly
    # 4: 4000-8000, ultra-poly
    
    if isinstance(n, int):
        if n <= 0:
            return 0
        elif n <= 1000:
            return 1
        elif n <= 2000:
            return 2
        elif n <= 4000:
            return 3
        elif n <= 8000:
            return 4
        elif n <= 12000:
            return 5
        elif n <= 16000:
            return 6
        else:
            return 7
    else: # torch tensor
        results = torch.zeros_like(n)
        # results[n <= 0] = 0
        results[(n > 0) & (n <= 1000)] = 1
        results[(n > 1000) & (n <= 2000)] = 2
        results[(n > 2000) & (n <= 4000)] = 3
        results[(n > 4000) & (n <= 8000)] = 4
        results[(n > 8000) & (n <= 12000)] = 5
        results[(n > 12000) & (n <= 16000)] = 6
        results[n > 16000] = 7
        return results
    
def quantize_num_CC_ss(n):
    # 0: <=0, un cond
    # 1: 0-1000, low-poly
    # 2: 1000-2000, mid-poly
    # 3: 2000-4000, high-poly
    # 4: 4000-8000, ultra-poly
    
    if isinstance(n, int):
        if n <= 0:
            return 0
        elif n <= 1:
            return 1
        elif n <= 5:
            return 2
        elif n <= 10:
            return 3
        elif n <= 30:
            return 4
        elif n <= 50:
            return 5
        elif n <= 100:
            return 6
        else:
            return 7
    else: # torch tensor
        results = torch.zeros_like(n)
        # results[n <= 0] = 0
        results[(n > 0) & (n <= 1)] = 1
        results[(n > 1) & (n <= 5)] = 2
        results[(n > 5) & (n <= 10)] = 3
        results[(n > 10) & (n <= 30)] = 4
        results[(n > 30) & (n <= 50)] = 5
        results[(n > 50) & (n <= 100)] = 6
        results[n > 100] = 7
        return results