import math
import os
import numpy as np
import h5py
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
    timecourses = None
    coordinates = None
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
        if timecourses is None:
            out_shape = (sum(x.shape[-1] for x in states), T)
            timecourses = np.zeros(out_shape, dtype=np.float32)
        if coordinates is None:
            coordinates = np.zeros((sum(x.shape[-1] for x in states),), dtype=np.int32)
        h = 0
        for state in states:
            _h = state.size(-1)
            timecourses[h:h + _h, t:t + _t] = state.detach().cpu().numpy()[mask].T
            coordinates[h:h + _h] = i // B
            h += _h
        t += _t
    if verbose:
        stderr('\n')

    return dict(
        timecourses=timecourses,  # <n_neurons, n_tokens>
        coordinates=coordinates  # <n_neurons>
    )


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


def dump_connectivity(connectivity, path, coordinates=None):
    dirpath = os.path.dirname(path)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    with h5py.File(path, 'w') as f:
        f.create_dataset('connectivity', data=connectivity)
        f.create_dataset('coordinates', data=coordinates)


def read_connectivity(path, coordinates=None):
    with h5py.File(path, 'r') as f:
        connectivity = f['connectivity'][:]
        if coordinates in f:
            coordinates = f['coordinates'][:]
        else:
            coordinates = None

    return dict(
        connectivity=connectivity,  # <n_neurons, n_neurons>
        coordinates=coordinates  # <n_neurons>
    )


def run_connectivity(
        model_name='gpt2',
        output_dir='results',
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
        model_kwargs=None,
        overwrite=False
):
    if data_kwargs is None:
        data_kwargs = {}
    if model_kwargs is None:
        model_kwargs = {}
    if n_tokens is None:
        n_tokens = (N_TOKENS // (seq_len * batch_size)) * seq_len * batch_size

    connectivity_dir = os.path.join(output_dir, 'connectivity')
    if os.path.exists(os.path.join(connectivity_dir, 'finished.txt')) and not overwrite:
        stderr('Connectivity already computed\n')
        return

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

        if not os.path.exists(connectivity_dir):
            os.makedirs(connectivity_dir)

        n = round(len(input_ids) // n_iterates)
        R = []
        for i in range(0, len(input_ids), n):
            _input_ids = input_ids[i:i+n]
            _attention_mask = attention_mask[i:i+n]

            out = get_timecourses(model, _input_ids, _attention_mask, batch_size=batch_size, **model_kwargs)
            timecourses = out['timecourses']
            coordinates = out['coordinates']
            _R = get_connectivity(timecourses, n_components=n_components)
            R.append(_R)
            if n_iterates > 1:
                dump_connectivity(
                    _R,
                    os.path.join(connectivity_dir, 'R_%s_i%d.obj' % (_input_data_name, i // n + 1)),
                    coordinates=coordinates
                )
        R = np.stack(R, axis=-1)
        if n_iterates > 1:
            R = fisher_average(R, eps=eps)
        else:
            R = R[0]
        dump_connectivity(R, os.path.join(connectivity_dir, 'R_%s_avg.obj' % _input_data_name))

    with open(os.path.join(connectivity_dir, 'finished.txt'), 'w') as f:
        f.write('Done\n')
