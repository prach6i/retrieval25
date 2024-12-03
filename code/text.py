

from contextlib import nullcontext
from functools import partial
import logging
from typing import List, Union
import matplotlib.pyplot as plt

import torch
from torch import Tensor as T
import torch.nn.functional as F
from tqdm import tqdm
import json
from torch import nn
from typing import Tuple, List, Union
import numpy as np
#device = "cuda:0" if torch.cuda.is_available() else "cpu"

rel = lambda x: F.relu(x)

def build_topk_mask(embs, k , dim: int = -1):
    if isinstance(embs, np.ndarray):
        embs = torch.Tensor(embs)
    values, indices = torch.topk(embs, k, dim=dim)
    topk_mask = torch.zeros_like(embs)
    topk_mask.scatter_(dim=-1, index=indices, value=1)
    return topk_mask

def topk_sparsify(embs: torch.Tensor, k: int, dim: int = -1):
    topk_mask = build_topk_mask(embs, k=k, dim=dim)
    embs *= topk_mask
    return embs

class CSRTextEncoder(torch.nn.Module):
    def __init__(self):
        super(CSRTextEncoder,self).__init__()
        self.linear1 = nn.Linear(512, 1000)
        self.batch_norm1 = nn.BatchNorm1d(1000)
        self.linear2 = nn.Linear(768, 1000)
        self.batch_norm2 = nn.BatchNorm1d(1000)
        self.linear3 = nn.Linear(1000, 768)
        self.linear4 = nn.Linear(1000, 512)
        self.dropout = nn.Dropout(p=0.5)
        self.getReconstructionLoss = nn.MSELoss()
        self.rel = lambda x: F.relu(x)
        
    def forward(self, precomputed_embeddings,text_names, mask, training, device, k, verbose: bool = False):
        vocab_emb1 = self.linear1(precomputed_embeddings)
        vocab_emb2 = self.rel(vocab_emb1)
        batch_emb=vocab_emb2
        #print("batch_emb",batch_emb[:10][:10])
        if training:
            topk_mask = build_topk_mask(batch_emb, k).to(device)
            bow_mask = mask.to(device)
            mask = torch.logical_or(bow_mask, topk_mask).to(device)
            batch_emb1 = batch_emb.to(device) * mask
            #print("batch_emb1",batch_emb1[:10][:10])
            #batch_emb1 = self.rel(batch_emb1)
        else:
            batch_emb1 = batch_emb
        
        denominator = torch.max(batch_emb1, dim=0)[0] + 0.000000001
        norm_emb = batch_emb1 / denominator
        proj_emb = self.linear4(norm_emb)
        reconstruction_loss = self.getReconstructionLoss(proj_emb, precomputed_embeddings)

        if training:
            outputs = (norm_emb, reconstruction_loss, text_names)
        else:
            outputs = (batch_emb1, reconstruction_loss, text_names)
        
        
        return outputs
