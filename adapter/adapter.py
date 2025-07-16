# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor, nn
import math
from typing import Tuple, Type
import numpy as np

class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        # print(q.shape)
        # print("#"*10)
        # print(q.dtype)
        # print(self.q_proj.weight.dtype)
        # print("#"*10)
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out

class Adapter(nn.Module):
    def __init__(self, embedding_dim: int = 512, num_heads: int = 8, ffn_dim: int = None):
        super().__init__()
        self.attention = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        
        # FFN
        if ffn_dim is None:
            ffn_dim = 4 * embedding_dim 
        
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, ffn_dim),
            nn.GELU(),  # or nn.ReLU()
            nn.Linear(ffn_dim, embedding_dim)
        )

        self.linear = nn.Linear(embedding_dim, embedding_dim)
        # self.linear2 = nn.Linear(embedding_dim, embedding_dim)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, x: Tensor, y: Tensor, train=True) -> Tensor:
        if train:
            residual = x.clone()
            x = self.attention(x, x, x)
            x = self.norm1(x + residual)
            
            residual2 = x.clone()
            x = self.ffn(x)
            x = self.norm2(x + residual2)
            
            residual3 = y.clone()
            y = self.attention(y, y, y)
            y = self.norm1(y + residual3)
            
            residual4 = y.clone()
            y = self.ffn(y)
            y = self.norm2(y + residual4)
            
            # GAP (Global Average Pooling)
            image_features1 = torch.mean(x, dim=1)  # [batch_size, embedding_dim]
            image_features2 = torch.mean(y, dim=1)  # [batch_size, embedding_dim]
            
            # Linear 층 통과
            image_features1 = self.linear(image_features1)
            image_features2 = self.linear(image_features2)
            
            # 정규화
            image_features1 = image_features1 / image_features1.norm(dim=1, keepdim=True)
            image_features2 = image_features2 / image_features2.norm(dim=1, keepdim=True)
            
            # cosine similarity as logits
            logit_scale = self.logit_scale.exp()
            # logits_per_image = logit_scale * image_features1 @ image_features2.t()
            # logits_per_image2 = logits_per_image.t()
            
            return image_features1, image_features2, logit_scale
        else:
            residual = x.clone()
            x = self.attention(x, x, x)
            x = self.norm1(x + residual)
            
            residual2 = x.clone()
            x = self.ffn(x)
            x = self.norm2(x + residual2)
            
            # GAP (Global Average Pooling)
            image_features1 = torch.mean(x, dim=1)  # [batch_size, embedding_dim]
            
            # Linear 층 통과
            image_features1 = self.linear(image_features1)
            
            # 정규화
            image_features1 = image_features1 / image_features1.norm(dim=1, keepdim=True)
            
            return image_features1
        # return logits_per_image, logits_per_image2,
        