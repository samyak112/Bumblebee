import torch
import math
import pytorch_lightning as pl 
import torch.nn as nn
from .utils import create_copy

def scaled_dot_product(q,k,v,mask=None,dropout=None):
    d_k = q.size()[-1]

    '''
        Scale QKᵀ by sqrt(d_k) to control variance of the dot product. Without scaling, logits grow with dimension, causing softmax saturation and very small gradients.
    '''
    attention_scores = torch.matmul(q,k.transpose(-2,-1))/math.sqrt(d_k)

    if mask is not None:
        attention_scores = attention_scores.masked_fill(mask == 0, -9e15)

    ### Softmax
    '''
        Scaling by sqrt(d_k) rescales the attention logits to control variance
        so softmax does not saturate and gradients remain stable.

        Subtracting the max is a separate step: it shifts all logits equally
        to avoid overflow in exp during softmax. This does not change relative
        differences or the softmax result.
    '''
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

        '''
            Need to project the embedding in 4 different subspaces because we cant just slice a raw embedding as 
            an attribute is not present in an embedding in a continous space. It is scattrered across all dimensions.
            So we use weighted matrix which helps us in bringing those attributes in line so that after slicing each slice
            contains meaningful and complete attributes

            It would still be a single or multiple attribute and wont be the complete representation of the whole embedding
        '''

        query = self.linears[0](query)
        key   = self.linears[1](key)
        value = self.linears[2](value)


        '''
            Split the projected embeddings into multiple attention heads.
            This is equivalent to manually slicing the embedding
            (e.g., Q[:, :, 0:d_k], Q[:, :, d_k:2*d_k], ...)
            but reshaping + transpose allows all heads to be processed in parallel.
        '''
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

