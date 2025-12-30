import torch
import math
import pytorch_lightning as pl 
import torch.nn as nn
from .utils import create_copy



def scaled_dot_product(q,k,v,mask=None,dropout=None):
    d_k = q.size()[-1]
    attention_scores = torch.matmul(q,k.transpose(-2,-1))/math.sqrt(d_k)

    if mask is not None:
        attention_scores = attention_scores.masked_fill(mask == 0, -9e15)

    ### Softmax
    stable_scores = attention_scores - torch.max(attention_scores, dim=-1, keepdim=True).values
    exponentiated_values = torch.exp(stable_scores)
    attention_softmaxed = exponentiated_values/ torch.sum(exponentiated_values, dim=-1, keepdim=True)
    
    ### Dropout on attention weights
    if dropout is not None:
        attention_softmaxed = dropout(attention_softmaxed)

    values = torch.matmul(attention_softmaxed,v)

    return values,attention_softmaxed


class MultiHeadAttention(nn.Module):
    def __init__(self,h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.linears = create_copy(nn.Linear(d_model, d_model), 4)
        self.dropout = dropout
        self.d_k = d_model // h
        self.h = h
    
    def forward(self,query,key,value,mask=None):

        batches = query.size(0)

        query = self.linears[0](query)
        key   = self.linears[1](key)
        value = self.linears[2](value)

        query = query.view(batches, -1, self.h, self.d_k).transpose(1, 2)
        key   = key.view(batches, -1, self.h, self.d_k).transpose(1, 2)
        value = value.view(batches, -1, self.h, self.d_k).transpose(1, 2)

        x, _ = scaled_dot_product(
            query, key, value, mask=mask, dropout=self.dropout
        )

        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(batches, -1, self.h * self.d_k)
        )
        
        return self.linears[-1](x)

