from collections import defaultdict

import torch
from torch.utils.data import DataLoader

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

        optimizer = self.model.configure_optimizers(lr=3e-4)

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