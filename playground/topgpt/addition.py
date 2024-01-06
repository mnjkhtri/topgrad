import torch
from torch.utils.data import DataLoader
from transformer import Transformer
from trainer import Trainer

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
    batch_size = 64

    #lr:
    lr = 3e-4

    #max iters:
    max_iters = 3000

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
            # print([i.device for i in model.parameters()])
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
        if trainer.iter_num % 1000 == 0:
            model.eval()
            with torch.no_grad():
                train_score = evaluate(trainer, split='train', max_batches=5)
                valid_score = evaluate(trainer, split='valid')
            score = train_score + valid_score
            if score > top_score:
                top_score = score
                print(f"Saving model with new top score of {score}")
                import os
                ckpth_path = os.path.join('./', "model.pt")
                torch.save(model.state_dict(), ckpth_path)
            model.train()

    #train
    trainer = Trainer(model, train_dataset, batch_size, lr, max_iters)
    trainer.add_callback('on_batch_end', batch_end_callback)
    trainer.run()