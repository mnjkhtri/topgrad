import torch
import torch.nn as nn
import torch.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, prenorm=False, act=nn.ReLU, dropout=0.1):
        super().__init__(self)
        assert embed_dim % num_heads == 0, "embed_dim must be divisble by num_heads"

        self.num_heads = num_heads
        self.head_size = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key   = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        fc_out = nn.Linear(embed_dim, embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            act(),
            nn.Linear(ff_dim, embed_dim),
        )

        self.dropout = nn.Dropout(dropout, inplace=True)

        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def attn(self, query, key, value):

        batch_size = Q.shape[0]

        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)

        Q = Q.view(batch_size, -1, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.num_heads, self.head_size).permute(0, 2, 1, 3)

        energy = torch.einsum('bhqd,bhkd->bhqk', Q, K)

        attention = F.softmax(energy/(self.head_size ** 0.5), dim=-1)

        out = torch.einsum('bhal,bhlv->bhav', [attention, V]).permute(0,2,1,3).contiguous()

        out = out.view(batch_size, -1, self.num_heads * self.head_dim)

        out = self.fc_out(out)
        return out
    
    def forward(self, x):

        if self.prenorm:
            raise NotImplementedError 
    
        else:
            x = self.ln1(x + self.dropout(self.attn(x, x, x)))
            x = self.ln2(x + self.dropoutk(self.feed_forward(x)))
        return x
