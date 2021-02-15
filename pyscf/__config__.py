import os, sys

#
# All parameters initialized before loading pyscf_conf.py will be overwritten
# by the dynamic importing procedure.
#
DEBUG = False

MAX_MEMORY = int(os.environ.get('PYSCF_MAX_MEMORY', 4000)) # MB
TMPDIR = os.environ.get('TMPDIR', '.')
TMPDIR = os.environ.get('PYSCF_TMPDIR', TMPDIR)

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
    if sys.version_info < (3,0):
        import imp
        imp.load_source('pyscf.__config__', conf_file)
        del(imp)
    else:
        from importlib import machinery
        machinery.SourceFileLoader('pyscf.__config__', conf_file).load_module()
        del(machinery)
del(os, sys)

#
# All parameters initialized after loading pyscf_conf.py will be kept in the
# program.
#
