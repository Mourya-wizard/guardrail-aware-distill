import gd.env  # noqa
import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

STUDENT, TEACHER = "Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-9B"


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def load_tok(name=STUDENT):
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    return tok


def load_model(name, train=False):
    m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).cuda()
    return m.train() if train else m.eval()


def prompt_ids(tok, user, system=None):
    """Token ids of the chat prompt up to (and including) the assistant-turn start; thinking disabled."""
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tok(text, add_special_tokens=False).input_ids


@torch.no_grad()
def generate(model, tok, users, system=None, prefixes=None, max_new_tokens=160, temperature=0.0, top_p=1.0, batch_size=32):
    """-> list of dicts {text, ids, prefix_ids, prompt_len}; `ids` is the continuation only (EOS removed),
    the forced prefix (if any) is tokenized separately and kept in `prefix_ids`."""
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    pad = tok.pad_token_id if tok.pad_token_id is not None else eos
    prefixes = prefixes or [None] * len(users)
    res = [None] * len(users)
    order = sorted(range(len(users)), key=lambda i: len(users[i]))  # length-bucket batches
    for b in range(0, len(order), batch_size):
        idx = order[b:b + batch_size]
        seqs, plens = [], []
        for i in idx:
            p = prompt_ids(tok, users[i], system)
            plens.append(len(p))
            seqs.append(p + (tok(prefixes[i], add_special_tokens=False).input_ids if prefixes[i] else []))
        L = max(map(len, seqs))
        inp = torch.tensor([[pad] * (L - len(s)) + s for s in seqs]).cuda()
        att = torch.tensor([[0] * (L - len(s)) + [1] * len(s) for s in seqs]).cuda()
        kw = dict(do_sample=True, temperature=temperature, top_p=top_p, top_k=0) if temperature > 0 else dict(do_sample=False)
        out = model.generate(input_ids=inp, attention_mask=att, max_new_tokens=max_new_tokens, eos_token_id=eos, pad_token_id=pad, **kw)[:, L:]
        for k, i in enumerate(idx):
            ids = out[k].tolist()
            ended = eos in ids
            if ended:
                ids = ids[:ids.index(eos)]
            ids = [t for t in ids if t != pad]
            pre = tok(prefixes[i], add_special_tokens=False).input_ids if prefixes[i] else []
            res[i] = {"text": tok.decode(ids, skip_special_tokens=True), "ids": ids, "ended": ended, "prefix_ids": pre, "prompt_len": plens[k]}
    return res
