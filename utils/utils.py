import torch
import pandas as pd
import re

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"  # Dingbats
    "\U000024C2-\U0001F251"
    "]+"
)

def preprocess_tweet(text):
    text = re.sub(r'(https://\S+)', '<URL>', text)
    #     text = text.replace('THREAD: ', '')
    text = EMOJI_PATTERN.sub(r'', text)
    encoded_string = text.encode("ascii", "ignore")
    text = encoded_string.decode()
    text = text.replace('#', '')
    text = text.replace('&amp;', '&')

    return text

def generate_random_mask(ids, out_dim, in_dim, p=0.5, device='cpu'):
    probas = torch.tensor([1-p, p], device=device)
    weights = torch.tensor([0, 1 / p], device=device)
    rand_masks = []
    for id in ids:
        # Set seed
        torch.random.manual_seed(id)
        torch.cuda.random.manual_seed(id)

        # Generate Mask
        indices = torch.multinomial(probas, num_samples=out_dim * in_dim, replacement=True)
        rand_mask = weights[indices].view(out_dim, in_dim)
        rand_masks.append(rand_mask)
    return torch.stack(rand_masks, dim=0)

def generate_grouped_mask(ids, out_dim, in_dim, p=0.5, device='cpu'):
    probas = torch.tensor([1-p, p], device=device)
    weights = torch.tensor([0, 1 / p], device=device)
    rand_masks = []
    for id in ids:
        # Set seed
        torch.random.manual_seed(id)
        torch.cuda.random.manual_seed(id)

        # Generate Mask
        indices = torch.multinomial(probas, num_samples=out_dim * in_dim, replacement=True)
        rand_mask = weights[indices].view(out_dim, in_dim)
        rand_masks.append(rand_mask)
    return torch.stack(rand_masks, dim=0)
