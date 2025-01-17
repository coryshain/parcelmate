import re

CONNECTIVITY_NAME = 'connectivity'
PARCELLATION_NAME = 'parcellation'
STABILITY_NAME = 'stability'
SAMPLE_NAME = 'sample'

N_TOKENS = 1000000
EXTENSION = '.h5'
INPUT_NAME_RE = re.compile('(%s|%s)_(.+)_(%s\d+|avg)%s' % (
    CONNECTIVITY_NAME, PARCELLATION_NAME, SAMPLE_NAME, EXTENSION)
)

PLOT_DIR = 'plots'
