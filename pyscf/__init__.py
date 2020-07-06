# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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

    >>> from pyscf import gto, scf
    >>> mol = gto.M(atom='H 0 0 0; H 0 0 1.2', basis='cc-pvdz')
    >>> mol.apply(scf.RHF).run()
    converged SCF energy = -1.06111199785749
    -1.06111199786

'''

__version__ = '1.7.3'

import os
# Avoid too many threads being created in OMP loops.
# See issue https://github.com/pyscf/pyscf/issues/317
if 'OPENBLAS_NUM_THREADS' not in os.environ:
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
if 'MKL_NUM_THREADS' not in os.environ:
    os.environ['MKL_NUM_THREADS'] = '1'

import sys
from distutils.version import LooseVersion
import numpy
if LooseVersion(numpy.__version__) <= '1.8.0':
    raise SystemError("You're using an old version of Numpy (%s). "
                      "It is recommended to upgrade numpy to 1.8.0 or newer. \n"
                      "You still can use all features of PySCF with the old numpy by removing this warning msg. "
                      "Some modules (DFT, CC, MRPT) might be affected because of the bug in old numpy." %
                      numpy.__version__)
elif '1.16.2' <= LooseVersion(numpy.__version__) < '1.18':
    #sys.stderr.write('Numpy 1.16 has memory leak bug  '
    #                 'https://github.com/numpy/numpy/issues/13808\n'
    #                 'It is recommended to downgrade to numpy 1.15 or older\n')
    import ctypes
    from numpy.core import _internal
    def _get_void_ptr(arr):
        simple_arr = numpy.asarray(_internal._unsafe_first_element_pointer(arr))
        c_arr = (ctypes.c_char * 0).from_buffer(simple_arr)
        return ctypes.cast(ctypes.byref(c_arr), ctypes.c_void_p)
    # patch _get_void_ptr as a workaround to numpy issue #13808
    _internal._get_void_ptr = _get_void_ptr

from pyscf import __config__
from pyscf import lib
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo

#__path__.append(os.path.join(os.path.dirname(__file__), 'future'))
__path__.append(os.path.join(os.path.dirname(__file__), 'tools'))

DEBUG = __config__.DEBUG

def M(**kwargs):
    '''Main driver to create Molecule object (mol) or Material crystal object (cell)'''
    from pyscf import __all__
    if kwargs.get('a') is not None:  # a is crystal lattice parameter
        return __all__.pbc.gto.M(**kwargs)
    else:  # Molecule
        return gto.M(**kwargs)

del(os, sys, LooseVersion)
