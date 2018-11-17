# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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

__version__ = '1.5.4'

import os
import sys
from distutils.version import LooseVersion
import numpy
if LooseVersion(numpy.__version__) <= LooseVersion('1.8.0'):
    raise SystemError("You're using an old version of Numpy (%s). "
                      "It is recommended to upgrad numpy to 1.8.0 or newer. \n"
                      "You still can use all features of PySCF with the old numpy by removing this warning msg. "
                      "Some modules (DFT, CC, MRPT) might be affected because of the bug in old numpy." %
                      numpy.__version__)

from pyscf import __config__
from pyscf import lib
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo

#__path__.append(os.path.join(os.path.dirname(__file__), 'future'))
__path__.append(os.path.join(os.path.dirname(__file__), 'tools'))

DEBUG = __config__.DEBUG

del(os, sys, LooseVersion, numpy)
