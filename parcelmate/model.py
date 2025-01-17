import math
import os
import copy
import numpy as np
import h5py
from scipy import optimize
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import MiniBatchKMeans
import torch
from transformers import AutoModel, AutoTokenizer

from parcelmate.constants import *
from parcelmate.data import *
from parcelmate.util import *


def get_model_and_tokenizer(model_name):
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def get_timecourses(
        model,
        input_ids,
        attention_mask,
        batch_size=8,
        highpass=None,
        lowpass=None,
        step=0.2,
        verbose=True,
        indent=0,
        **kwargs
):
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
            timecourses[h:h + _h, t:t + _t] = bandpass(
                state.detach().cpu().numpy()[mask].T,
                step=step,
                lower=highpass,
                upper=lowpass
            )
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


def save_h5_data(
        data,
        path,
        verbose=True,
        indent=0
):
    dirpath = os.path.dirname(path)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    if verbose:
        stderr('%sSaving to %s\n' % (' ' * indent, path))
    with h5py.File(path, 'w') as f:
        for key in data:
            f.create_dataset(key, data=data[key])


def load_h5_data(path, verbose=True, indent=0):
    if verbose:
        stderr('%sLoading from %s\n' % (' ' * indent, path))
    out = {}
    with h5py.File(path, 'r') as f:
        for key in f.keys():
            out[key] = f[key][:]

    return out


def sample_parcellations(
        connectivity,
        n_networks=50,
        n_samples=100,
        binarize_connectivity=True,
        n_components_pca=200,
        n_components_ica=None,
        clustering_kwargs=None,
        verbose=True,
        indent=0
):
    if verbose:
        stderr('%sSampling (n_networks=%d)\n' % (' ' * indent, n_networks))
    indent += 2

    if clustering_kwargs is None:
        clustering_kwargs = {}
    X = connectivity
    if binarize_connectivity:
        X = (X > np.quantile(X, 0.9, axis=1)).astype(int)
    if n_components_pca:
        n_components = n_components_pca
        if n_components == 'auto':
            n_components = n_networks - 1
        if verbose:
            stderr('%sPCA transforming (n components = %s)' % (' ' * indent, n_components))
        t1 = time.time()
        n_components = min(n_components, X.shape[-1])
        m = PCA(n_components=n_components, svd_solver='auto', whiten=True)
        X = m.fit_transform(X)
        stderr(' (%0.2fs)\n' % (time.time() - t1))
    if n_components_ica:
        n_components = n_components_ica
        if n_components == 'auto':
            n_components = n_networks - 1
        n_components = min(n_components, X.shape[-1])
        if verbose:
            stderr('%sICA transforming (n components = %s)' % (' ' * indent, n_components))
        t1 = time.time()
        m = FastICA(n_components=n_components, whiten='unit-variance')
        X = m.fit_transform(X)
        stderr(' (%0.2fs)\n' % (time.time() - t1))

    if verbose:
        stderr('%sDrawing samples\n' % (' ' * indent))
    indent += 2
    n_units = X.shape[0]
    samples = np.zeros((n_samples, n_units))
    scores = np.zeros(n_samples)
    for i in range(n_samples):
        if verbose and n_samples > 1:
            stderr('\r%sSample %d/%d' % (' ' * indent, i + 1, n_samples))
        m = MiniBatchKMeans(n_clusters=n_networks, **clustering_kwargs)
        _sample = m.fit_predict(X)
        _score = m.inertia_
        samples[i, :] = _sample
        scores[i] = _score

    if n_samples > 1:
        stderr('\n')

    return dict(
        samples=samples,  # <n_samples, n_units>
        scores=scores  # <n_samples>
    )


def _align_samples(
        samples,
        w=None,
        n_alignments=None,
        shuffle=False,
        greedy=True,
        verbose=True,
        indent=0
):
    if w is None:
        _w = 1
    else:
        _w = w[0]
    n_samples = samples.shape[0]
    n_units = samples.shape[1]
    n_networks = samples.max() + 1
    reference = (samples[0][None, ...] == np.arange(n_networks)[..., None]).astype(float)
    parcellation = None
    C = 0

    # Align subsequent samples
    if shuffle:
        s_ix = np.random.permutation(n_samples)
        samples = samples[s_ix]
    n = n_alignments
    if n is None:
        n = n_samples
    i = 0
    for i_cum in range(n):
        if verbose:
            stderr('\r%sAlignment %d/%d' % (' ' * indent, i_cum + 1, n))

        if w is not None:
            _w = w[i]
        else:
            _w = 1
        if _w == 0:
            continue

        if len(samples.shape) == 2:
            s = (samples[i][None, ...] == np.arange(n_networks)[..., None])
        else:
            s = samples[i].T
        s = s.astype(float)
        _reference = standardize_array(reference)
        _s = standardize_array(s)
        scores = np.dot(
            _reference,
            _s.T,
        ) / n_units

        _, ix_r = optimize.linear_sum_assignment(scores, maximize=True)
        s = s[ix_r]
        if parcellation is None:
            parcellation = s * _w
        else:
            parcellation = parcellation + s * _w
        if greedy:
            reference = parcellation
        C += _w
        i += 1
        if i >= n_samples:
            i = 0
            if shuffle:
                s_ix = np.random.permutation(n_samples)
                samples = samples[s_ix]

    if verbose and n > 0:
        stderr('\n')

    parcellation = parcellation / C

    return parcellation

def align_samples(
        samples,
        scores,
        n_alignments=None,
        weight_samples=False,
        verbose=True,
        indent=0
):
    if verbose:
        stderr('%sAligning samples\n' % (' ' * indent))
    indent += 1

    s_ix = np.argsort(scores)
    samples = samples[s_ix]
    scores = scores[s_ix]
    if weight_samples:
        w = 1 - scores  # Flip to upweight lower inertia
    else:
        w = None

    parcellation = _align_samples(
        samples,
        w=w,
        n_alignments=n_alignments,
        shuffle=False,
        greedy=True,
        verbose=verbose,
        indent=indent + 2
    ).T

    indent -= 1

    return parcellation


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
        highpass=None,
        lowpass=None,
        step=0.2,
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

    connectivity_dir = os.path.join(output_dir, CONNECTIVITY_NAME)
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
                highpass=highpass,
                lowpass=lowpass,
                step=step,
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
                save_h5_data(
                    dict(
                        connectivity=_connectivity,
                        coordinates=coordinates
                    ),
                    os.path.join(
                        connectivity_dir,
                        '%s_%s_%s%d%s' % (
                            CONNECTIVITY_NAME,
                            domain,
                            SAMPLE_NAME,
                            i // n + 1,
                            EXTENSION
                        )
                    ),
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
        save_h5_data(
            dict(
                connectivity=connectivity,
                coordinates=coordinates
            ),
            os.path.join(
                connectivity_dir,
                '%s_%s_avg%s' % (
                    CONNECTIVITY_NAME,
                    domain,
                    EXTENSION
                ),
            ),
            verbose=verbose,
            indent=indent
        )
        indent -= 2

    with open(os.path.join(connectivity_dir, 'finished.txt'), 'w') as f:
        f.write('Done\n')



def run_parcellation(
        output_dir='results',
        n_networks=50,
        n_samples=100,
        binarize_connectivity=True,
        n_components_pca=None,
        n_components_ica=None,
        clustering_kwargs=None,
        n_alignments=None,
        weight_samples=False,
        overwrite=False,
        verbose=True,
        indent=0
):
    connectivity_dir = os.path.join(output_dir, CONNECTIVITY_NAME)

    for path in os.listdir(connectivity_dir):
        match = INPUT_NAME_RE.match(path)
        if not match:
            continue
        inpath = os.path.join(connectivity_dir, path)
        data = load_h5_data(inpath, verbose=verbose, indent=indent)

        if overwrite or not 'parcellation' in data:
            R = np.nan_to_num(data['connectivity'])
            R = np.abs(R)

            sample = sample_parcellations(
                R,
                n_networks=n_networks,
                n_samples=n_samples,
                binarize_connectivity=binarize_connectivity,
                n_components_pca=n_components_pca,
                n_components_ica=n_components_ica,
                clustering_kwargs=clustering_kwargs,
                verbose=verbose,
                indent=indent + 2
            )
            parcellation = align_samples(
                sample['samples'],
                sample['scores'],
                n_alignments=n_alignments,
                weight_samples=weight_samples,
                verbose=verbose,
                indent=indent + 2
            )
            data['parcellation'] = parcellation

            save_h5_data(
                data,
                inpath,
                verbose=verbose,
                indent=indent + 2
            )






