import torch.nn as nn

'''
Still figuring out what is the need to use a feed forward network 
when we already have non linearities in attention mechanism.

One explanation I came across is that softmax is applied to the attention weights (the token-mixing coefficients), not to the token representations themselves. Attention is basically computing a weighted average of value vectors, which is still a linear operation in value space.
But still confirming
'''
class FeedForwardNetwork(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForwardNetwork, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        '''
            Position-wise feed-forward network.
            Applied independently to each token.

            This block does not mix information across tokens.
            Instead, it mixes and nonlinearly transforms the dimensions
            of a single token embedding, allowing feature–feature
            interactions that attention alone cannot create.

            The expansion (d_model -> d_ff) creates a larger feature space,
            the nonlinearity introduces new features, and the projection
            back to d_model compresses them for the next layer.

            MHA decides what information to gather from other tokens, and FFN decides how to transform and interpret that gathered information inside each token.
        '''
        return self.w_2(self.dropout(self.w_1(x).relu()))