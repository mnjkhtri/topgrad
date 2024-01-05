import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW, Adam

class TransformerBlock(nn.Module):
    """
    Sort of an autoencoder: [B, T<=ML, Emb --> B, T, Emb]. Supports both encoder and decoder blocks;
    """
    def __init__(self, max_length, embed_dim, ff_dim, num_heads, cross=True, prenorm=True, act=nn.ReLU, dropout=0.1):

        super(TransformerBlock, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisble by num_heads"

        self.num_heads = num_heads
        self.cross = cross
        self.prenorm = prenorm

        self.head_size = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key   = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        if cross:
            self.register_buffer("mask", torch.tril(torch.ones(max_length, max_length))
                                        .view(1, 1, max_length, max_length))

        self.fc_out = nn.Linear(embed_dim, embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            act(),
            nn.Linear(ff_dim, embed_dim),
        )

        self.dropout = nn.Dropout(dropout, inplace=True)

        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def attn(self, query, key, value):

        batch_size, seq_length = query.shape[:2]

        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)

        Q = Q.view(batch_size, -1, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.num_heads, self.head_size).permute(0, 2, 1, 3)

        att = torch.einsum('bhqd,bhkd->bhqk', [Q, K])/(self.head_size ** 0.5)
        if self.cross:
            att = att.masked_fill(self.mask[:,:,:seq_length,:seq_length] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        out = torch.einsum('bhal,bhlv->bhav', [att, V]).permute(0,2,1,3).contiguous()
        out = out.view(batch_size, -1, self.num_heads * self.head_size)
        out = self.fc_out(out)
        return out
    
    def forward(self, x):

        if self.prenorm:
            x = self.ln1(x)
            x = x + self.dropout(self.attn(x, x, x))
            x = self.ln2(x)
            x = x + self.dropout(self.feed_forward(x))
        else:
            x = self.ln1(x + self.dropout(self.attn(x, x, x)))
            x = self.ln2(x + self.dropout(self.feed_forward(x)))
        return x
    

class Transformer(nn.Module):
    def __init__(self, vocab_size, max_length, embed_dim, ff_dim, num_heads, layers):

        super(Transformer, self).__init__()
        self.max_length = max_length

        self.pos_embed = nn.Embedding(max_length, embed_dim)
        self.tok_embed = nn.Embedding(vocab_size, embed_dim)
        self.register_buffer("pos", torch.arange(max_length))

        self.tbs = nn.Sequential(*[TransformerBlock(max_length, embed_dim, ff_dim, num_heads) for _ in range(layers)])

        self.final = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, target=None):

        logits = self.final(self.tbs(self.pos_embed(self.pos[:x.shape[1]]) + self.tok_embed(x)))
        loss = None
        if target is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), target.view(-1), ignore_index=-1)
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, do_sample=False,):
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


class AdditionDataset(torch.utils.data.Dataset):

    def __init__(self, ndigit, split):
        super(AdditionDataset, self).__init__()
        assert ndigit <= 3, "warning!!! memory inefficiency"
        self.ndigit = ndigit
        nums = (10**self.ndigit)**2
        rng = torch.Generator()
        rng.manual_seed(1337)
        perm = torch.randperm(nums, generator=rng)
        num_test = min(int(nums*0.2), 1000)
        self.ixes = perm[:num_test] if split == 'valid' else perm[num_test:]

    def __len__(self):
        return self.ixes.shape[0]

    def __getitem__(self, idx):
        num = self.ixes[idx].detach().item()
        nd = 10**self.ndigit
        a = num // nd
        b = num % nd
        c = a + b
        astr = f'{a:0{self.ndigit}}' 
        bstr = f'{b:0{self.ndigit}}' 
        cstr = f'{c:0{self.ndigit+1}}'[::-1]
        render = astr + bstr + cstr
        dix = [int(s) for s in render]
        x, y = torch.tensor(dix[:-1]), torch.tensor(dix[1:])
        y[:self.ndigit*2-1] = -1
        return x, y


class Trainer:

    def __init__(self, model, train_dataset, batch_size, lr, max_iters):
        self.model = model
        self.train_dataset = train_dataset
        self.batch_size = batch_size
        self.lr = lr
        self.max_iters = max_iters

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.callbacks = defaultdict(list)

    def add_callback(self, event, callback):
        self.callbacks[event].append(callback)

    def trigger_callbacks(self, onevent):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self):

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=None,
            pin_memory=True,
        )

        optimizer = Adam(self.model.parameters(), lr=self.lr)

        self.model.to(self.device)
        self.model.train()
        data_iter = iter(train_loader)

        from tqdm import tqdm
        for self.iter_num in tqdm(range(self.max_iters)):
            try: batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = [t.to(self.device) for t in batch]
            input, target = batch 
            optimizer.zero_grad()
            _, self.loss = self.model(input, target)
            self.loss.backward()
            optimizer.step()
            self.trigger_callbacks('on_batch_end')

 
if __name__ == "__main__":
    #model:
    vocab_size = 10
    max_length = 10
    embed_dim = 128
    ff_dim = 256
    num_heads = 4
    layers = 2
    model = Transformer(vocab_size, max_length, embed_dim, ff_dim, num_heads, layers)

    #dataset:
    train_dataset = AdditionDataset(3, 'train')
    valid_dataset = AdditionDataset(3, 'valid')

    #batch size:
    batch_size = 32

    #lr:
    lr = 3e-4

    #max iters:
    max_iters = 5000

    #evaluate:
    def evaluate(trainer, split, max_batches=None):
        dataset = {'train':train_dataset, 'valid':valid_dataset}[split]
        loader = DataLoader(dataset, batch_size=100, drop_last=False)

        count = 0
        correct = 0
        mistakes_printed_already = 0

        factors = torch.tensor([[10**i for i in range(dataset.ndigit+1)][::-1]]).to(trainer.device)
        
        for b, (x, y) in enumerate(loader):
            x, y = x.to(trainer.device), y.to(trainer.device)
            d1d2 = x[:, :dataset.ndigit*2]
            d1d2d3 = model.generate(d1d2, dataset.ndigit+1, do_sample=False)
            d3 = d1d2d3[:, -(dataset.ndigit+1):]
            d3 = d3.flip(1)
            
            d1i = (d1d2[:,:dataset.ndigit] * factors[:,1:]).sum(1)
            d2i = (d1d2[:,dataset.ndigit:dataset.ndigit*2] * factors[:,1:]).sum(1)
            d3i_pred = (d3 * factors).sum(1)
            
            d3i_gt = d1i + d2i
            state = (d3i_pred == d3i_gt)

            for i in range(x.shape[0]): #for each item in the batch:
                count += 1
                if state[i]: correct += 1

                if not state[i] and mistakes_printed_already < 10:
                    mistakes_printed_already += 1
                    print(f"GPT claims that {d1i[i]:5d} + {d2i[i]:5d} = {d3i_pred[i]:5d} but gt is {d3i_gt[i]:5d}")

            if max_batches is not None and b >= 5: break

        print(f'For {split} total count: {count}, Correct: {correct}, Accuracy {correct/count}')
        print("-------------------------------------------------------------------")
        return correct

    # callback:
    top_score = 0
    def batch_end_callback(trainer):
        global top_score
        if trainer.iter_num % 500 == 0:
            model.eval()
            with torch.no_grad():
                train_score = evaluate(trainer, split='train', max_batches=5)
                valid_score = evaluate(trainer, split='valid')
            score = train_score + valid_score
            if score > top_score:
                top_score = score
                print(f"Saving model with new top score of {score}")
                import os
                ckpth_path = os.path.join('./', "mode.pt")
                torch.save(model.state_dict(), ckpth_path)
            model.train()

    #train
    trainer = Trainer(model, train_dataset, batch_size, lr, max_iters)
    trainer.add_callback('on_batch_end', batch_end_callback) 
    trainer.run()
