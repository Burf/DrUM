import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, *args, function = None):
        self.args = list(args)
        self.function = function

    def __len__(self):
        return len(self.args[0])
    
    def __getitem__(self, idx):
        out = [self.function(arg[idx]) if self.function is not None else arg[idx] for arg in self.args]
        return out[0] if len(out) == 1 else tuple(out)

def collate_fn(sample, processor = "openai/clip-vit-large-patch14", return_tensors = "pt", padding = "max_length", max_token_size = 256, truncation = True):
    if not callable(processor):
        key = "collate_processor"
        store = globals()
        if key in store:
            processor = store[key]
        else:
            from transformers import CLIPProcessor
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
            store[key] = processor
    max_length = min(processor.tokenizer.model_max_length if hasattr(processor, "tokenizer") else processor.model_max_length, max_token_size)
    return processor(text = sample, return_tensors = return_tensors, max_length = max_length, padding = padding, truncation = truncation)