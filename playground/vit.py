import math
import argparse
import io
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
from transformers import ViTModel
from topgpt.transformer import TransformerBlock

#torch gelu is different
class NewGELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

#minor changes from the gpt version
class TransformerBlock(nn.Module):
    def __init__(self, max_length, embed_dim, ff_dim, num_heads, dropout=0.1):

        super(TransformerBlock, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisble by num_heads"

        self.max_length = max_length
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.dp = dropout

        #derv:
        self.head_size = self.embed_dim // self.num_heads

        #attention blocks
        self.query = nn.Linear(self.embed_dim, self.embed_dim)
        self.key = nn.Linear(self.embed_dim, self.embed_dim)
        self.value = nn.Linear(self.embed_dim, self.embed_dim)
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim)

        #feedforward blocks
        self.mlpf = nn.Sequential(
            nn.Linear(self.embed_dim, self.ff_dim),
            NewGELU(),
            nn.Linear(self.ff_dim, self.embed_dim),
        )
        
        #after attn and ff blocks
        self.dropout = nn.Dropout(self.dp, inplace=True)

        #depends
        self.ln1 = nn.LayerNorm(self.embed_dim)
        self.ln2 = nn.LayerNorm(self.embed_dim)

    def attn(self, x):

        batch_size, seq_length = x.shape[:2]

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        Q = Q.view(batch_size, -1, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.num_heads, self.head_size).permute(0, 2, 1, 3)

        att = torch.einsum('bhqd,bhkd->bhqk', [Q, K])/(self.head_size ** 0.5) #scaled dot produt attention
        att = F.softmax(att, dim=-1)

        out = torch.einsum('bhal,bhlv->bhav', [att, V]).permute(0,2,1,3).contiguous()
        out = out.view(batch_size, -1, self.num_heads * self.head_size)
        out = self.dropout(self.c_proj(out)) #projection after attending to tokens
        return out
    
    def forward(self, x):

        x = x + self.attn(self.ln1(x))
        x = x + self.mlpf(self.ln2(x))
        return x

class ViT(nn.Module):
    def __init__(self, image_size, patch_size, embed_dim, ff_dim, num_heads, layers):
        super(ViT, self).__init__()
        assert image_size % patch_size == 0

        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.layers = layers
    
        #derive:
        self.num_channels = 3
        self.num_patches = (image_size//patch_size)**2
        
        #embed for cls this does not come from embedding fn:
        self.register_buffer("cls_token", torch.ones(1, 1, embed_dim))

        #Store position embedding as a parameter:
        self.position_embeddings = nn.Parameter(torch.randn(1, self.num_patches+1, self.embed_dim))

        #For patch embedding fn:
        self.projection = nn.Conv2d(
            self.num_channels, 
            self.embed_dim, 
            kernel_size=self.patch_size, 
            stride=self.patch_size
        )

        self.dropout = nn.Dropout(0.1, inplace=True) #just after embeddings - for training

        self.tbs = nn.Sequential(*[TransformerBlock(
            self.num_patches, 
            self.embed_dim, 
            self.ff_dim, 
            self.num_heads,
            ) for _ in range(self.layers)]
        )

        self.ln_f = nn.LayerNorm(embed_dim) #layer norm after every block passes on

        self.head = nn.Linear(self.embed_dim, 1000) #classification head


    def patch_embeddings(self, x):
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1) #batch_size, 1, embed_dim

        x = self.projection(x) #batch_size, embed_dim, patch_num, patch_num
        x = x.view(x.shape[0], self.embed_dim, -1).permute(0, 2, 1) #batch_size, no_of_patches, embed_dim

        x = torch.cat((cls_tokens, x), dim=1)
        return x

    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in ('google/vit-base-patch16-224')
        image_size = 224
        patch_size = 16
        
        config = {
            'google/vit-base-patch16-224' : dict(embed_dim=768,  ff_dim=768*4,  num_heads=12, layers=12),
        }[model_type]

        #vanilla vit has not classifier head instead has pooler;
        from transformers import ViTForImageClassification
        model_hf = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224") #finetuned on imagenet
        sd_hf = model_hf.state_dict()
        #The original vit was trained on 21k imagenet, whose head is not released, only body;

        model = cls(image_size, patch_size, **config)
        sd = model.state_dict()

        assert len(sd) == len(sd_hf), "mismatch state dict, maybe you forgot to consider something"

        up = lambda i: {
            f'vit.encoder.layer.{i}.attention.attention.query.weight':      f'tbs.{i}.query.weight',
            f'vit.encoder.layer.{i}.attention.attention.query.bias':        f'tbs.{i}.query.bias',
            f'vit.encoder.layer.{i}.attention.attention.key.weight':        f'tbs.{i}.key.weight',
            f'vit.encoder.layer.{i}.attention.attention.key.bias':          f'tbs.{i}.key.bias',
            f'vit.encoder.layer.{i}.attention.attention.value.weight':      f'tbs.{i}.value.weight',
            f'vit.encoder.layer.{i}.attention.attention.value.bias':        f'tbs.{i}.value.bias',
            f'vit.encoder.layer.{i}.attention.output.dense.weight':         f'tbs.{i}.c_proj.weight',
            f'vit.encoder.layer.{i}.attention.output.dense.bias':           f'tbs.{i}.c_proj.bias',
            f'vit.encoder.layer.{i}.intermediate.dense.weight':             f'tbs.{i}.mlpf.0.weight',
            f'vit.encoder.layer.{i}.intermediate.dense.bias':               f'tbs.{i}.mlpf.0.bias',
            f'vit.encoder.layer.{i}.output.dense.weight':                   f'tbs.{i}.mlpf.2.weight',
            f'vit.encoder.layer.{i}.output.dense.bias':                     f'tbs.{i}.mlpf.2.bias',
            f'vit.encoder.layer.{i}.layernorm_before.weight':               f'tbs.{i}.ln1.weight',
            f'vit.encoder.layer.{i}.layernorm_before.bias':                 f'tbs.{i}.ln1.bias',
            f'vit.encoder.layer.{i}.layernorm_after.weight':                f'tbs.{i}.ln2.weight',
            f'vit.encoder.layer.{i}.layernorm_after.bias':                  f'tbs.{i}.ln2.bias',
        }

        mapping = {
            'vit.embeddings.cls_token': 'cls_token',
            'vit.embeddings.position_embeddings': 'position_embeddings',
            'vit.embeddings.patch_embeddings.projection.weight': 'projection.weight',
            'vit.embeddings.patch_embeddings.projection.bias': 'projection.bias',
            'vit.layernorm.weight': 'ln_f.weight',
            'vit.layernorm.bias': 'ln_f.bias',
            'classifier.weight': 'head.weight',
            'classifier.bias': 'head.bias'
        }

        for i in range(config['layers']): mapping.update(up(i))
        assert len(mapping.keys()) == len(sd_hf.keys()), "mismatch mapping between the models"

        from tqdm import tqdm
        print("Importing ViT")
        for k in tqdm(sd_hf):
            kn = mapping[k]
            assert sd_hf[k].shape == sd[kn].shape;
            with torch.no_grad():
                sd[kn].copy_(sd_hf[k])

        return model

    def forward(self, x):
        x = self.dropout(self.position_embeddings + self.patch_embeddings(x))
        print("After each transformer block:")
        print(x.shape)
        print("------------")
        x = self.ln_f(self.tbs(x))
        print("After taking out cls token through head:")
        return self.head(x[:,0,:])


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description="Inference using ViT model with an image URL")
    parser.add_argument("image_url", help="URL of the image to perform inference on")
    args = parser.parse_args()

    import requests
    response = requests.get(args.image_url)
    img = Image.open(io.BytesIO(response.content)).convert('RGB')

    preprocess = transforms.Compose([
        transforms.Resize(256),             
        transforms.CenterCrop(224),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    img = preprocess(img).unsqueeze(0).to(device)
    vit = ViT.from_pretrained("google/vit-base-patch16-224").to(device)

    with torch.no_grad():
        vit.to(device)
        vit.eval()
        logits = vit(img)

    index = torch.argmax(F.softmax(logits, dim=-1))

    with open('./imagenet1000_clsidx_to_labels.txt') as f:
        labels = eval(f.read())
        cat = labels.get(index.item(), "Label not found")
    print("The image is of", cat)