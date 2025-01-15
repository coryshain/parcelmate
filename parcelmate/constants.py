import re

N_TOKENS = 1000000
CONNECTIVITY_PREFIX = 'R'
SAMPLE_PREFIX = 'sample'
EXTENSION = '.h5'
INPUT_NAME_RE = re.compile('%s_(.+)_(%s\d+|avg)%s' % (CONNECTIVITY_PREFIX, SAMPLE_PREFIX, EXTENSION))

CONNECTIVITY_DIR = 'connectivity'
STABILITY_DIR = 'stability'