import torch
from transformer import Transformer
 
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    prompt = "The answer to life, universe and everything is"
    model_type = 'gpt2-medium'
    
    model = Transformer.from_pretrained(model_type)
    model.to(device)
    model.eval()
    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")
    indices = torch.tensor(tokenizer.encode(prompt)).view(1, -1).to(device)
    out = model.generate(indices, 100, do_sample=True)
    print(tokenizer.decode(out.detach().tolist()[0]))