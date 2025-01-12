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
        wrap=True,
        shuffle=True,
        verbose=True,
        **kwargs
):
    if verbose:
        stderr('Getting input data\n')
    assert seq_len > 0, 'seq_len must be positive'
    try:
        dataset = datasets.load_dataset(input_data, split=split, streaming=True, **kwargs)
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
            stderr('WARNING: Unused keyword arguments: %s\n' % kwargs)
        if isinstance(input_data, str):
            dataset = [input_data]
        else:
            dataset = input_data
        assert isinstance(dataset, list), 'x must be a string or list of strings'

    _n_tokens = 0
    input_ids = None
    attention_mask = None
    for input_data in dataset:
        toks = tokenizer(input_data)
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
    k = X.shape[1]
    m = X.mean(axis=0, keepdims=True)
    s = np.linalg.norm(X, axis=0, keepdims=True)
    np.subtract(X, m, out=X)
    np.divide(X, s, out=X)
    R = np.zeros((k, k), dtype=np.float32)
    R = np.dot(X.T, X, out=R)

    return R

def fisher(arr, eps=1e-3):
    return np.arctanh((1 - eps) * arr)

def fisher_average(*arrs, eps=1e-3, dtype=np.float32):
    out = None
    for arr in arrs:
        if out is None:
            out = np.zeros(arr.shape, dtype=dtype)
        out += fisher(arr, eps=eps)
    out /= len(arrs)
    out = np.tanh(out, dtype=dtype)

    return out

