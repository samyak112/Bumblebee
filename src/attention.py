import torch
import math
import pytorch_lightning as pl 

def scaled_dot_product(q,k,v):
    try:
        d_k = q.size()[-1]
        numerator = torch.matmul(q,k.transpose(-2,-1))
        attention_scores = numerator/math.sqrt(d_k)

        ### Softmax
        stable_scores = attention_scores - torch.max(attention_scores, dim=-1, keepdim=True).values
        exponentiated_values = torch.exp(stable_scores)
        attention_softmaxed = exponentiated_values/ torch.sum(exponentiated_values, dim=-1, keepdim=True)
        values = torch.matmul(attention_softmaxed,v)

        return values,attention_softmaxed

    except Exception as e:
        print(e)


### Test

seq_len, d_k = 3, 2
pl.seed_everything(42)
q = torch.randn(seq_len, d_k)
k = torch.randn(seq_len, d_k)
v = torch.randn(seq_len, d_k)
values, attention = scaled_dot_product(q, k, v)
print("Q\n", q)
print("K\n", k)
print("V\n", v)
print("Values\n", values)
print("Attention\n", attention)