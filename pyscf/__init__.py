# Copyright 2014-2022 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
*****************************************************
PySCF Python-based simulations of chemistry framework
*****************************************************

How to use
----------
There are two ways to access the documentation: the docstrings come with
the code, and an online program reference, available from
http://www.sunqm.net/pyscf/index.html

We recommend the enhanced Python interpreter `IPython <http://ipython.org>`_
and the web-based Python IDE `Ipython notebook <http://ipython.org/notebook.html>`_
to try out the package::

    >>> import pyscf
    >>> mol = pyscf.M(atom='H 0 0 0; H 0 0 1.2', basis='cc-pvdz')
    >>> mol.RHF().run()
    converged SCF energy = -1.06111199785749
    -1.06111199786

'''

__version__ = '2.3.0'

import os
import sys

# Load modules which are developed as plugins of the namespace package
PYSCF_EXT_PATH = os.getenv('PYSCF_EXT_PATH')
if PYSCF_EXT_PATH:
    for p in PYSCF_EXT_PATH.split(':'):
        if os.path.isdir(p):
            submodules = os.listdir(p)
            if 'pyscf' in submodules:
                # PYSCF_EXT_PATH points to the root directory of a submodule
                __path__.append(os.path.join(p, 'pyscf'))
            else:
                # Load all modules in PYSCF_EXT_PATH if it's a folder that
                # contains all extended modules
                for f in submodules:
                    __path__.append(os.path.join(p, f, 'pyscf'))
                del f
        elif os.path.exists(p):
            # Load all moduels defined inside the file PYSCF_EXT_PATH
            with open(p, 'r') as f:
                __path__.extend(f.read().splitlines())
            del f
    del p
elif '/site-packages/' in __file__ or '/dist-packages/' in __file__:
    # If pyscf has been installed in the standard runtime path (typically
    # under site-packages), and plugins are installed with the pip editable
    # mode, load namespace plugins. In this case, it is likely all modules are
    # managed by pip/conda or their virtual environments. It is safe to search
    # namespace plugins according to the old style of PEP 420.
    __path__ = __import__('pkgutil').extend_path(__path__, __name__)
else:
    # We need a better way to load plugins if pyscf is imported by the
    # environment variable PYTHONPATH. Current treatment may mix installed
    # plugins (e.g.  through pip install) with the developing plugins which
    # were accidentally placed under PYTHONPATH. When PYTHONPATH mechanism is
    # taken, an explicit list of extended paths (using environment
    # PYSCF_EXT_PATH) is recommended.
    __path__ = __import__('pkgutil').extend_path(__path__, __name__)
    if not all('/site-packages/' in p for p in __path__[1:]):
        sys.stderr.write('pyscf plugins found in \n%s\n'
                         'When PYTHONPATH is set, it is recommended to load '
                         'these plugins through the environment variable '
                         'PYSCF_EXT_PATH\n' % '\n'.join(__path__[1:]))

import numpy
if numpy.__version__[:5] in ('1.16.', '1.17.'):
    # Numpy memory leak bug https://github.com/numpy/numpy/issues/13808
    import ctypes
    from numpy.core import _internal
    def _get_void_ptr(arr):
        simple_arr = numpy.asarray(_internal._unsafe_first_element_pointer(arr))
        c_arr = (ctypes.c_char * 0).from_buffer(simple_arr)
        return ctypes.cast(ctypes.byref(c_arr), ctypes.c_void_p)
    _internal._get_void_ptr = _get_void_ptr

from pyscf import __config__
from pyscf import lib
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo

# Whether to enable debug mode. When this flag is set, some modules may run
# extra debug code.
DEBUG = __config__.DEBUG

def M(**kwargs):
    '''Main driver to create Molecule object (mol) or Material crystal object (cell)'''
    from pyscf import __all__
    if kwargs.get('a') is not None:  # a is crystal lattice parameter
        return __all__.pbc.gto.M(**kwargs)
    else:  # Molecule
        return gto.M(**kwargs)

del os, sys
