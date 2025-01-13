import numpy as np
import torch
import datasets

from parcelmate.util import stderr

def get_input_data(
        input_data,
        tokenizer,
        n_tokens,
        seq_len,
        split='train',
        take=100000,
        wrap=True,
        shuffle=True,
        verbose=True,
        indent=0,
        **kwargs
):
    if verbose:
        stderr('%sGetting input data\n' % (' ' * indent))
    assert seq_len > 0, 'seq_len must be positive'
    try:
        dataset = datasets.load_dataset(input_data, split=split, streaming=True, **kwargs)
        if take:
            dataset = dataset.take(take)
        key = None
        _dataset = []
        for input_data in dataset:
            if key is None:
                if 'text' in input_data:
                    key = 'text'
                elif 'content' in input_data:
                    key = 'content'
                else:
                    raise ValueError('no known content key found in dataset')
            _dataset.append(input_data[key])
        dataset = _dataset
        if shuffle:
            np.random.shuffle(dataset)
        assert split, 'split must be specified when loading a HuggingFace dataset'
    except (AssertionError, datasets.exceptions.DatasetNotFoundError):
        if kwargs:
            stderr('%sWARNING: Unused keyword arguments: %s\n' % (kwargs, ' ' * indent))
        if isinstance(input_data, str):
            dataset = [input_data]
        else:
            dataset = input_data
        assert isinstance(dataset, list), 'x must be a string or list of strings'

    _n_tokens = 0
    input_ids = None
    attention_mask = None
    for instance in dataset:
        toks = tokenizer(instance)
        ids, mask = toks['input_ids'], toks['attention_mask']
        if _n_tokens + len(ids) > n_tokens:
            ids = ids[:n_tokens - _n_tokens]
            mask = mask[:n_tokens - _n_tokens]
        __n_tokens = len(ids)
        if wrap:
            if input_ids is None:
                input_ids = [[]]
                attention_mask = [[]]
            elif len(input_ids[-1]) == seq_len:  # Start a new batch item
                input_ids.append([])
                attention_mask.append([])
            n = seq_len - len(input_ids[-1])
            assert n > 0, 'non-positive n found when wrapping input data. len(input_ids[-1]) = %d' % len(input_ids[-1])
            input_ids[-1].extend(ids[:n])
            attention_mask[-1].extend(mask[:n])
            ids = ids[n:]
            mask = mask[n:]
            while ids:  # Wrap
                n = min(len(ids), seq_len)
                input_ids.append(ids[:n])
                attention_mask.append(mask[:n])
                ids = ids[n:]
                mask = mask[n:]
        else:
            if input_ids is None:
                input_ids = []
                attention_mask = []
            input_ids.append(ids)
            attention_mask.append(mask)
        _n_tokens += __n_tokens
        if _n_tokens >= n_tokens:
            break

    assert n_tokens == _n_tokens, ('%d tokens requested but the dataset only contains %d tokens.'
                                   ' Consider increasing the value of `take`.'
                                   % (n_tokens, _n_tokens))

    input_ids = torch.as_tensor(pad(input_ids))
    attention_mask = torch.as_tensor(pad(attention_mask))

    return input_ids, attention_mask

def pad(arr, max_len=None, pad_value=0, right=True):
    if max_len is None:
        max_len = max(len(x) for x in arr)
    if right:
        out = [x + [pad_value] * (max_len - len(x)) for x in arr]
    else:
        out = [[pad_value] * (max_len - len(x)) + x for x in arr]
    return out

def correlate(X, rowvar=True):
    if rowvar:
        X = X.T
    X -= X.mean(axis=0, keepdims=True)
    X /= np.linalg.norm(X, axis=0, keepdims=True)
    R = X.T @ X

    return R

def fisher(arr, eps=1e-3):
    return np.arctanh(np.multiply(arr, 1 - eps, out=arr), out=arr)

def fisher_average(*arrs, eps=1e-3):
    out = None
    for arr in arrs:
        if out is None:
            out = fisher(arr, eps=eps)
        else:
            out += fisher(arr, eps=eps)
    out /= len(arrs)
    out = np.tanh(out, out=out)

    return out

