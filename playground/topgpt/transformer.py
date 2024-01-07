import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class NewGELU(nn.Module): #necessary
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class TransformerBlock(nn.Module):
    """
    Sort of an autoencoder: [B, T<=ML, Emb --> B, T, Emb].
    Equips cross-attention (decoder part)
    """
    def __init__(self, max_length, embed_dim, ff_dim, num_heads, prenorm=True, act=NewGELU, dropout=0.1):

        super(TransformerBlock, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisble by num_heads"

        self.max_length = max_length
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.prenorm = prenorm
        self.act = act
        self.dp= dropout

        #derv:
        self.head_size = self.embed_dim // self.num_heads

        #attention blocks
        self.c_attn = nn.Linear(self.embed_dim, 3*self.embed_dim)
        self.register_buffer(
            "mask", 
            torch.tril(torch.ones(self.max_length, self.max_length))
            .view(1, 1, self.max_length, self.max_length), persistent=False)
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim)

        #feedforward blocks
        self.mlpf = nn.Sequential(
            nn.Linear(self.embed_dim, self.ff_dim),
            self.act(),
            nn.Linear(self.ff_dim, self.embed_dim),
        )
        
        #after attn and ff blocks
        self.dropout = nn.Dropout(self.dp) #applied to two places (effective during training only)

        #depends
        self.ln1 = nn.LayerNorm(self.embed_dim)
        self.ln2 = nn.LayerNorm(self.embed_dim)

    def attn(self, x):

        batch_size, seq_length = x.shape[:2]

        Q, K, V = self.c_attn(x).split(self.embed_dim, dim=2)

        Q = Q.view(batch_size, -1, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.num_heads, self.head_size).permute(0, 2, 1, 3)

        att = torch.einsum('bhqd,bhkd->bhqk', [Q, K])/(self.head_size ** 0.5) #scaled dot produt attention
        att = att.masked_fill(self.mask[:,:,:seq_length,:seq_length] == 0, float('-inf'))
        att = self.dropout(F.softmax(att, dim=-1)) #just after probabilities, why tho? (attn dropout)

        #Attain to the probabilities
        out = torch.einsum('bhal,bhlv->bhav', [att, V]).permute(0,2,1,3).contiguous()
        out = out.view(batch_size, -1, self.num_heads * self.head_size)
        out = self.dropout(self.c_proj(out)) #projection after attending to tokens (residue dropout)
        return out
    
    def forward(self, x):

        if self.prenorm:
            x = x + self.attn(self.ln1(x))
            x = x + self.mlpf(self.ln2(x))
        else:
            x = self.ln1(x + self.attn(x))
            x = self.ln2(x + self.mlpf(x))
        return x
    

class Transformer(nn.Module):
    def __init__(self, vocab_size, max_length, embed_dim, ff_dim, num_heads, layers, emb_dropout=0.1):

        super(Transformer, self).__init__()
        self.max_length = max_length

        self.pos_embed = nn.Embedding(max_length, embed_dim)
        self.register_buffer("pos", torch.arange(max_length), persistent=False)
        self.tok_embed = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(emb_dropout)
    
        self.tbs = nn.Sequential(*[TransformerBlock(max_length, embed_dim, ff_dim, num_heads) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(embed_dim)

        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        self.apply(self._init_weight) #initialization for training

    def _init_weight(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def configure_optimizers(self, lr):
        """
        Separate out parameters by their types for AdamW 
        """
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f'{mn}.{pn}' if mn else pn
        
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        #validate that we considered every parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"parameters {inter_params} made it into both decay and no_decay sets"
        assert len(param_dict.keys() - union_params) == 0, f"parameters {param_dict.keys() - union_params} were note separated into either decay or no_decay sets"

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0},
        ]

        from torch.optim import AdamW
        optimizer = AdamW(optim_groups, lr=lr)
        return optimizer

    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in ('gpt2', 'gpt2-medium'), "please use only gpt2 or gpt2-medium, we poor"
        vocab_size = 50257
        max_length = 1024

        config = {
            'gpt2':         dict(embed_dim=768,  ff_dim=768*4,  num_heads=12, layers=12),
            'gpt2-medium':  dict(embed_dim=1024, ff_dim=1024*4, num_heads=16, layers=24)
        }[model_type]

        from transformers import GPT2LMHeadModel
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        model = cls(vocab_size, max_length, **config)
        sd = model.state_dict()

        assert len(sd_hf) == len(sd), "mismatch state dict, maybe you forgot to non-persist buffers"

        up = lambda i: {
            f'transformer.h.{i}.ln_1.weight':           f'tbs.{i}.ln1.weight',
            f'transformer.h.{i}.ln_1.bias':             f'tbs.{i}.ln1.bias',
            f'transformer.h.{i}.attn.c_attn.weight':    f'tbs.{i}.c_attn.weight',   #conv
            f'transformer.h.{i}.attn.c_attn.bias':      f'tbs.{i}.c_attn.bias',       
            f'transformer.h.{i}.attn.c_proj.weight':    f'tbs.{i}.c_proj.weight',   #conv
            f'transformer.h.{i}.attn.c_proj.bias':      f'tbs.{i}.c_proj.bias',
            f'transformer.h.{i}.ln_2.weight':           f'tbs.{i}.ln2.weight',
            f'transformer.h.{i}.ln_2.bias':             f'tbs.{i}.ln2.bias',
            f'transformer.h.{i}.mlp.c_fc.weight':       f'tbs.{i}.mlpf.0.weight',   #conv
            f'transformer.h.{i}.mlp.c_fc.bias':         f'tbs.{i}.mlpf.0.bias',
            f'transformer.h.{i}.mlp.c_proj.weight':     f'tbs.{i}.mlpf.2.weight',   #conv
            f'transformer.h.{i}.mlp.c_proj.bias':       f'tbs.{i}.mlpf.2.bias',
        }
    
        mapping = {
            'transformer.wpe.weight': 'pos_embed.weight',
            'transformer.wte.weight': 'tok_embed.weight',
            'transformer.ln_f.weight': 'ln_f.weight',
            'transformer.ln_f.bias': 'ln_f.bias',
            'lm_head.weight': 'lm_head.weight',
        }

        for i in range(config['layers']): mapping.update(up(i))
        assert len(mapping.keys()) == len(sd_hf.keys()), "mismatch mapping between the models"

        #conv1d checkpoints
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        from tqdm import tqdm
        print("Importing GPT")
        for k in tqdm(sd_hf):
            kn = mapping[k]
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[kn].shape;
                with torch.no_grad():
                    sd[kn].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[kn].shape;
                with torch.no_grad():
                    sd[kn].copy_(sd_hf[k])

        return model

    def forward(self, x, target=None):

        embed = self.ln_f(self.tbs(self.dropout(self.pos_embed(self.pos[:x.shape[1]]) + self.tok_embed(x))))
        logits = self.lm_head(embed)
        loss = None
        if target is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), target.view(-1), ignore_index=-1)
        return logits, loss
        
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, do_sample=False):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.shape[1] <= self.max_length else idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]/temperature

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)

            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx