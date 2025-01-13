import math
import os
import numpy as np
import h5py
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import torch
from transformers import AutoModel, AutoTokenizer

from parcelmate.constants import *
from parcelmate.data import *
from parcelmate.util import *

def get_model_and_tokenizer(model_name):
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def get_timecourses(model, input_ids, attention_mask, batch_size=8, verbose=True, **kwargs):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    if verbose:
        stderr('Getting timecourses\n')
    out = None
    t = 0
    T = int(attention_mask.detach().numpy().sum())
    B = int(math.ceil(input_ids.size(0) / batch_size))
    for i in range(0, input_ids.size(0), batch_size):
        if verbose:
            stderr('\r  Batch %d/%d' % (i // batch_size + 1, B))
        _input_ids = input_ids[i:i + batch_size].to(device)
        _attention_mask = attention_mask[i:i + batch_size].to(device)
        states = model(
            input_ids=_input_ids,
            attention_mask=_attention_mask,
            output_hidden_states=True,
            **kwargs
        ).hidden_states
        mask = _attention_mask.detach().cpu().numpy().astype(bool)
        _t = int(mask.sum())
        if out is None:
            out_shape = (sum(x.shape[-1] for x in states), T)
            out = np.zeros(out_shape, dtype=np.float32)
        h = 0
        for state in states:
            _h = state.size(-1)
            out[h:h + _h, t:t + _t] = state.detach().cpu().numpy()[mask].T
            h += _h
        t += _t
    if verbose:
        stderr('\n')

    return out  # <n_neurons, n_tokens>

def get_connectivity(timecourses, n_components=None):
    X = timecourses
    if n_components:
        m = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=n_components))
        ])
        X = m.fit_transform(X)
    R = correlate(X, rowvar=True)

    return R

def dump_connectivity(R, outpath):
    dirpath = os.path.dirname(outpath)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    with open(outpath, 'wb') as f:
        pickle.dump(R, f)

def run_connectivity(
        model_name='gpt2',
        results_dir='results',
        n_iterates=3,
        input_data_names=('wikitext', 'codeparrot'),
        seq_len=1024,
        n_tokens=None,
        split='train',
        wrap=True,
        shuffle=True,
        batch_size=8,
        n_components=None,
        eps=1e-3,
        verbose=True,
        data_kwargs=None,
        model_kwargs=None
):
    if data_kwargs is None:
        data_kwargs = {}
    if model_kwargs is None:
        model_kwargs = {}
    if n_tokens is None:
        n_tokens = (N_TOKENS // (seq_len * batch_size)) * seq_len * batch_size

    model, tokenizer = get_model_and_tokenizer(model_name)

    if isinstance(input_data_names, str):
        input_data_names = (input_data_names,)

    for _input_data_name in input_data_names:
        if _input_data_name == 'wikitext':
            input_data_kwargs = dict(
                input_data='wikitext',
                tokenizer= tokenizer,
                name='wikitext-103-raw-v1',
            )
        elif _input_data_name == 'codeparrot':
            input_data_kwargs = dict(
                input_data='codeparrot/codeparrot-clean',
                tokenizer=tokenizer,
            )
        else:
            raise ValueError('Unrecognized input data name: %s' % _input_data_name)
        input_data_kwargs.update(data_kwargs)

        input_ids, attention_mask = get_input_data(
            n_tokens=n_tokens * n_iterates,
            split=split,
            seq_len=seq_len,
            wrap=wrap,
            shuffle=shuffle,
            verbose=verbose,
            **input_data_kwargs
        )

        n = round(len(input_ids) // n_iterates)
        R = []
        for i in range(0, len(input_ids), n):
            _input_ids = input_ids[i:i+n]
            _attention_mask = attention_mask[i:i+n]

            states = get_timecourses(model, _input_ids, _attention_mask, batch_size=batch_size, **model_kwargs)
            _R = get_connectivity(states, n_components=n_components)
            R.append(_R)
            if n_iterates > 1:
                dump_connectivity(_R, os.path.join(results_dir, 'R_%s_i%d.obj' % (_input_data_name, i + 1)))
        if n_iterates > 1:
            R = fisher_average(R, eps=eps)
        else:
            R = R[0]
        dump_connectivity(R, os.path.join(results_dir, 'R_%s_avg.obj' % _input_data_name))
