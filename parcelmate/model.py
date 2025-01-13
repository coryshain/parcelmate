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


def get_timecourses(model, input_ids, attention_mask, batch_size=8, verbose=True, indent=0, **kwargs):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    if verbose:
        stderr('%sGetting timecourses\n' % (' ' * indent))
    timecourses = None
    coordinates = None
    t = 0
    T = int(attention_mask.detach().numpy().sum())
    B = int(math.ceil(input_ids.size(0) / batch_size))
    indent += 2
    for i in range(0, input_ids.size(0), batch_size):
        if verbose:
            stderr('\r%sBatch %d/%d' % (' ' * indent, i // batch_size + 1, B))
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


def save_connectivity(connectivity, path, coordinates=None, verbose=True, indent=0):
    dirpath = os.path.dirname(path)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    if verbose:
        stderr('%sSaving connectivity to %s\n' % (' ' * indent, path))
    with h5py.File(path, 'w') as f:
        f.create_dataset('connectivity', data=connectivity)
        if coordinates is not None:
            f.create_dataset('coordinates', data=coordinates)


def load_connectivity(path, coordinates=None, verbose=True, indent=0):
    if verbose:
        stderr('%sLoading connectivity from %s\n' % (' ' * indent, path))
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
        n_samples=3,
        input_data_names=('wikitext', 'codeparrot'),
        seq_len=1024,
        n_tokens=None,
        split='train',
        take=100000,
        wrap=True,
        shuffle=True,
        batch_size=8,
        n_components=None,
        eps=1e-3,
        data_kwargs=None,
        model_kwargs=None,
        overwrite=False,
        verbose=True,
        indent=0
):
    if data_kwargs is None:
        data_kwargs = {}
    if model_kwargs is None:
        model_kwargs = {}
    if n_tokens is None:
        n_tokens = (N_TOKENS // (seq_len * batch_size)) * seq_len * batch_size

    connectivity_dir = os.path.join(output_dir, 'connectivity')
    if os.path.exists(os.path.join(connectivity_dir, 'finished.txt')) and not overwrite:
        stderr('%sConnectivity already computed\n' % (' ' * indent))
        return

    model, tokenizer = get_model_and_tokenizer(model_name)

    if isinstance(input_data_names, str):
        input_data_names = (input_data_names,)

    indent = 0

    for _input_data_name in input_data_names:
        if verbose:
            stderr('%sRunning connectivity for %s\n' % (' ' * indent, _input_data_name))
        indent += 2
        if _input_data_name == 'wikitext':
            input_data_kwargs = dict(
                input_data='wikitext',
                tokenizer= tokenizer,
                name='wikitext-103-raw-v1',
            )
        elif _input_data_name == 'codeparrot':
            input_data_kwargs = dict(
                input_data='codeparrot/codeparrot-clean',
                tokenizer=tokenizer
            )
        else:
            raise ValueError('Unrecognized input data name: %s' % _input_data_name)
        input_data_kwargs.update(data_kwargs)

        input_ids, attention_mask = get_input_data(
            n_tokens=n_tokens * n_samples,
            split=split,
            take=take,
            seq_len=seq_len,
            wrap=wrap,
            shuffle=shuffle,
            verbose=verbose,
            indent=indent,
            **input_data_kwargs
        )

        if not os.path.exists(connectivity_dir):
            os.makedirs(connectivity_dir)

        if verbose:
            stderr('%sQuerying model\n' % (' ' * indent))
        n = int(np.ceil(len(input_ids) // n_samples))
        connectivity = []
        coordinates = None
        indent += 2
        for i in range(0, len(input_ids), n):
            if verbose:
                stderr('%sSample %d/%d\n' % (' ' * indent, i // n + 1, n_samples))
            _input_ids = input_ids[i:i+n]
            _attention_mask = attention_mask[i:i+n]

            indent += 2

            out = get_timecourses(
                model,
                _input_ids,
                _attention_mask,
                batch_size=batch_size,
                verbose=verbose,
                indent=indent,
                **model_kwargs
            )
            timecourses = out['timecourses']
            coordinates = out['coordinates']
            _timecourses = get_connectivity(timecourses, n_components=n_components)
            connectivity.append(_timecourses)
            if n_samples > 1:
                save_connectivity(
                    _timecourses,
                    os.path.join(
                        connectivity_dir,
                        '%s_%s_%s%d%s' % (
                            CONNECTIVITY_PREFIX,
                            _input_data_name,
                            SAMPLE_PREFIX,
                            i // n + 1,
                            EXTENSION
                        )
                    ),
                    coordinates=coordinates,
                    verbose=verbose,
                    indent=indent
                )
            indent -= 2
        indent -= 2
        connectivity = np.stack(connectivity, axis=0)
        if n_samples > 1:
            connectivity = fisher_average(connectivity, eps=eps)
        else:
            connectivity = connectivity[0]
        save_connectivity(
            connectivity,
            os.path.join(
                connectivity_dir,
                '%s_%s_avg%s' % (
                    CONNECTIVITY_PREFIX,
                    _input_data_name,
                    EXTENSION
                ),
            ),
            coordinates=coordinates,
            verbose=verbose,
            indent=indent
        )
        indent -= 2

    with open(os.path.join(connectivity_dir, 'finished.txt'), 'w') as f:
        f.write('Done\n')
