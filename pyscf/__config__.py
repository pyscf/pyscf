import os
import sys
import tempfile

#
# All parameters initialized before loading pyscf_conf.py will be overwritten
# by the dynamic importing procedure.
#

DEBUG = False

MAX_MEMORY = int(os.environ.get('PYSCF_MAX_MEMORY', 4000)) # MB
TMPDIR = os.environ.get('PYSCF_TMPDIR', tempfile.gettempdir())
ARGPARSE = bool(os.getenv('PYSCF_ARGPARSE', False))

VERBOSE = 3  # default logger level (logger.NOTE)
UNIT = 'angstrom'

#
# Loading pyscf_conf.py and overwriting above parameters
#
for conf_file in (os.environ.get('PYSCF_CONFIG_FILE', None),
                  os.path.join(os.path.abspath('.'), '.pyscf_conf.py'),
                  os.path.join(os.environ.get('HOME', '.'), '.pyscf_conf.py')):
    if conf_file is not None and os.path.isfile(conf_file):
        break
else:
    conf_file = None

if conf_file is not None:
    with open(conf_file, 'r') as f:
        exec(f.read())
    del f
del (os, sys, tempfile)

#
# All parameters initialized after loading pyscf_conf.py will be kept in the
# program.
#
