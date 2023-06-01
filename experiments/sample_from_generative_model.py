import torch
import torch.nn.functional as F
from transformers import AutoModelWithLMHead, AutoTokenizer, GPT2Tokenizer, GPT2LMHeadModel
import argparse

import re
import numpy as np


# Gets the score for the top-k logits to improve quality of samples.
def top_k_logits(logits, k):
    if k == 0:
        return logits
    values = torch.topk(logits, k)[0]
    batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
    return torch.where(logits < batch_mins, torch.ones_like(logits) * -1e10, logits)


# Generates from the model using optional top-k sampling
def sample_sequence(model, length, batch_size=1, context=None, temperature=1, top_k=10, sample=True, device='cuda'):
    context = torch.tensor(context, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    prev = context
    output = context
    past = None
    with torch.no_grad():
        for i in range(length):
            logits = model(prev).logits
            logits = logits[:, -1, :] / temperature
            logits = top_k_logits(logits, k=top_k)
            log_probs = F.softmax(logits, dim=-1)
            if sample:
                prev = torch.multinomial(log_probs, num_samples=1)
            else:
                _, prev = torch.topk(log_probs, k=1, dim=-1)
            output = torch.cat((output, prev), dim=1)
    return output


def convert_to_bert_format(bias_context_to_generations, bert_file, generation_only=False):
    """Convert to BERT regard classifier format."""
    with open(bert_file, 'w') as f:
        sample_idx = 0
        samples = []
        for bias_context, gen in bias_context_to_generations.items():
            for sample in gen:
                if generation_only:
                    samples.append('\t'.join([str(sample_idx), sample]))
                else:
                    samples.append('\t'.join([str(sample_idx), bias_context + sample]))
                sample_idx += 1
        f.write('\n'.join(samples) + '\n')


def filter_first_sentence(text):
    """Heuristic to only keep the first `sentence` in text."""
    # Cut off the line when we see the first period.
    text = text.replace('\n', '. ').replace('\t', '. ')
    if '! ' in text:
        period_idx = text.index('! ')
    elif '? ' in text:
        period_idx = text.index('? ')
    elif '. ' in text:
        period_idx = text.index('. ')
    else:
        period_idx = len(text)
    sample_end = min(period_idx + 1, len(text))
    text = text[:sample_end]
    return text

