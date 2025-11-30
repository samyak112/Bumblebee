import torch
import math
import pytorch_lightning as pl 

def scaled_dot_product(q,k,v,mask=None):
    try:
        d_k = q.size()[-1]
        numerator = torch.matmul(q,k.transpose(-2,-1))
        attention_scores = numerator/math.sqrt(d_k)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -9e15)

        ### Softmax
        stable_scores = attention_scores - torch.max(attention_scores, dim=-1, keepdim=True).values
        exponentiated_values = torch.exp(stable_scores)
        attention_softmaxed = exponentiated_values/ torch.sum(exponentiated_values, dim=-1, keepdim=True)
        values = torch.matmul(attention_softmaxed,v)

        return values,attention_softmaxed

    except Exception as e:
        print(e)


### Test
d_model = 12
heads = 1
seq_len, d_k = 3, d_model//heads
pl.seed_everything(42)

X = torch.randn(seq_len,d_model)

W_q = torch.randn(d_model,d_k)
W_k = torch.randn(d_model,d_k)
W_v = torch.randn(d_model,d_k)

# Linear projection of X seq into q,k,v
q = X @ W_q
k = X @ W_k
v = X @ W_v
values, attention = scaled_dot_product(q, k, v)
print("Q\n", q)
print("K\n", k)
print("V\n", v)
print("Values\n", values)
print("Attention\n", attention)