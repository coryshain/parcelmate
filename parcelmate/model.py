import math
import os
import copy
import numpy as np
import pandas as pd
import h5py
from matplotlib import pyplot as plt
import seaborn as sns
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
        for s, state in enumerate(states):
            _h = state.size(-1)
            timecourses[h:h + _h, t:t + _t] = state.detach().cpu().numpy()[mask].T
            coordinates[h:h + _h] = s
            h += _h
        t += _t
    if verbose:
        stderr('\n')

    model.to('cpu')
    torch.cuda.empty_cache()

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


def load_connectivity(path, verbose=True, indent=0):
    if verbose:
        stderr('%sLoading connectivity from %s\n' % (' ' * indent, path))
    with h5py.File(path, 'r') as f:
        connectivity = f['connectivity'][:]
        if 'coordinates' in f:
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
        domains=('wikitext', 'bookcorpus', 'agnews', 'tldr17', 'codeparrot', 'random', 'whitespace'),
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

    connectivity_dir = os.path.join(output_dir, CONNECTIVITY_DIR)
    if os.path.exists(os.path.join(connectivity_dir, 'finished.txt')) and not overwrite:
        stderr('%sConnectivity already computed\n' % (' ' * indent))
        return

    model, tokenizer = get_model_and_tokenizer(model_name)

    if isinstance(domains, str):
        domains = (domains,)

    indent = 0

    for domain in domains:
        if verbose:
            stderr('%sRunning connectivity for %s\n' % (' ' * indent, domain))
        indent += 2
        _data_kwargs = copy.deepcopy(data_kwargs)
        if domain == 'wikitext':
            _data_kwargs.update(dict(
                dataset='wikitext',
                tokenizer= tokenizer,
                name='wikitext-103-raw-v1',
            ))
        elif domain == 'bookcorpus':
            _data_kwargs.update(dict(
                dataset='bookcorpus'
            ))
        elif domain == 'agnews':
            _data_kwargs.update(dict(
                dataset='fancyzhx/ag_news'
            ))
        elif domain == 'codeparrot':
            _data_kwargs.update(dict(
                dataset='codeparrot/codeparrot-clean'
            ))
        elif domain == 'tldr17':
            _data_kwargs.update(dict(
                dataset='webis/tldr-17'
            ))
        elif domain == 'random':
            _data_kwargs.update(dict(
                dataset='random'
            ))
        elif domain == 'whitespace':
            _data_kwargs.update(dict(
                dataset='whitespace'
            ))
        else:
            raise ValueError('Unrecognized input data name: %s' % domain)
        _data_kwargs['trust_remote_code'] = True
        _data_kwargs['tokenizer'] = tokenizer

        input_ids, attention_mask = get_dataset(
            n_tokens=n_tokens * n_samples,
            split=split,
            take=take,
            seq_len=seq_len,
            wrap=wrap,
            shuffle=shuffle,
            verbose=verbose,
            indent=indent,
            **_data_kwargs
        )

        if not os.path.exists(connectivity_dir):
            os.makedirs(connectivity_dir)

        if verbose:
            stderr('%sQuerying model\n' % (' ' * indent))
        n = int(np.ceil(len(input_ids) / n_samples))
        connectivity = []
        coordinates = None
        indent += 2
        for i in range(0, len(input_ids), n):
            t0 = time.time()
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
            if n_components:
                n_components_ = min(n_components, timecourses.shape[1])
            else:
                n_components_ = None
            _connectivity = get_connectivity(timecourses, n_components=n_components_)
            connectivity.append(_connectivity)
            if n_samples > 1:
                save_connectivity(
                    _connectivity,
                    os.path.join(
                        connectivity_dir,
                        '%s_%s_%s%d%s' % (
                            CONNECTIVITY_PREFIX,
                            domain,
                            SAMPLE_PREFIX,
                            i // n + 1,
                            EXTENSION
                        )
                    ),
                    coordinates=coordinates,
                    verbose=verbose,
                    indent=indent
                )
            if verbose:
                stderr('%sElapsed time: %.2f s\n' % (' ' * indent, time.time() - t0))
            indent -= 2
        indent -= 2
        if n_samples > 1:
            connectivity = fisher_average(*connectivity, eps=eps)
        else:
            connectivity = connectivity[0]
        save_connectivity(
            connectivity,
            os.path.join(
                connectivity_dir,
                '%s_%s_avg%s' % (
                    CONNECTIVITY_PREFIX,
                    domain,
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

def check_stability(
        output_dir='results',
        verbose=True,
        indent=0
):
    connectivity_dir = os.path.join(output_dir, CONNECTIVITY_DIR)
    stability_dir = os.path.join(output_dir, STABILITY_DIR)

    samples_by_domain = {}
    averages_by_domain = {}
    for path in os.listdir(connectivity_dir):
        match = INPUT_NAME_RE.match(path)
        if match:
            domain = match.group(1)
        else:
            continue
        key = match.group(2)
        if key == 'avg':
            R_by_domain = averages_by_domain
        else:
            key = int(key[len(SAMPLE_PREFIX):])
            R_by_domain = samples_by_domain
        if not domain in R_by_domain:
            R_by_domain[domain] = {}
        filepath = os.path.join(connectivity_dir, path)
        data = load_connectivity(filepath, verbose=verbose, indent=indent)
        R = data['connectivity']
        R_by_domain[domain][key] = R

    for domain in samples_by_domain:
        n = len(samples_by_domain[domain])
        R = np.zeros((n, n))
        labels = sorted(list(samples_by_domain[domain].keys()))
        for i, key1 in enumerate(labels):
            if key1 == 'avg':
                continue
            R1 = samples_by_domain[domain][key1]
            ix = np.tril_indices(R1.shape[0], k=-1)
            R1 = R1[ix]
            for j, key2 in enumerate(labels):
                if key2 == 'avg':
                    continue
                R2 = samples_by_domain[domain][key2]
                R2 = R2[ix]
                R[i, j] = np.corrcoef(R1, R2)[0, 1]

        R = pd.DataFrame(R, index=labels, columns=labels)

        if not os.path.exists(stability_dir):
            os.makedirs(stability_dir)

        filepath = os.path.join(stability_dir, 'withindomains_%s.png' % domain)
        ax = sns.heatmap(
            R,
            cmap='coolwarm',
            vmin=-1,
            vmax=1,
            xticklabels=True,
            yticklabels=True,
            annot=True
        )
        fig = ax.get_figure()
        fig.savefig(filepath, dpi=150)
        plt.close('all')

    labels = sorted(list(averages_by_domain.keys()))
    n = len(labels)
    R = np.zeros((n, n))
    for i, domain1 in enumerate(labels):
        R1 = averages_by_domain[domain1]['avg']
        ix = np.tril_indices(R1.shape[0], k=-1)
        R1 = R1[ix]
        for j, domain2 in enumerate(labels):
            R2 = averages_by_domain[domain2]['avg']
            R2 = R2[ix]
            R[i, j] = np.corrcoef(R1, R2)[0, 1]

    R = pd.DataFrame(R, index=labels, columns=labels)

    if not os.path.exists(stability_dir):
        os.makedirs(stability_dir)

    filepath = os.path.join(stability_dir, 'between_domains.png')
    ax = sns.heatmap(
        R,
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        xticklabels=True,
        yticklabels=True,
        annot=True
    )
    ax.tick_params(axis='x', rotation=45)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close('all')



