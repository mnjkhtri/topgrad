import unittest
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformer import Transformer
# -----------------------------------------------------------------------------

class TestHuggingFaceImport(unittest.TestCase):

    def test_gpt2(self):
        model_type = 'gpt2'
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        prompt = "The answer to universe is"

        # create a minGPT and a huggingface/transformers model
        model = Transformer.from_pretrained(model_type)
        model_hf = GPT2LMHeadModel.from_pretrained(model_type) #init a HF model too

        # ship both to device
        model.to(device)
        model_hf.to(device)

        # set both to eval mode
        model.eval()
        model_hf.eval()

        # tokenize input prompt
        tokenizer = GPT2Tokenizer.from_pretrained(model_type)
        model_hf.config.pad_token_id = model_hf.config.eos_token_id # suppress a warning
        indices = tokenizer(prompt, return_tensors='pt').to(device)['input_ids']

        # ensure the logits match exactly
        logits1, loss = model(indices)
        logits2 = model_hf(indices).logits
        print(logits1, logits2)
        self.assertTrue(torch.allclose(logits1, logits2))

if __name__ == '__main__':
    unittest.main()