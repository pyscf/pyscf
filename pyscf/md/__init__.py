#!/usr/bin/env python
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
#
# Author: Matthew Hennefarth <matthew.hennefarth@gmail.com>

'''
Molecular Dynamics
==================

Simple usage::

    >>> from pyscf import gto, dft
    >>> import pyscf.md as md
    >>> mol = gto.M(atom='N 0 0 0; N 0 0 1', basis='def2-tzvp')
    >>> mf = dft.RKS(mol)
    >>> mf.xc = 'pbe,pbe'
    >>> integrator = md.NVE(mf, dt=5, time=10).run()
'''

import numpy as np

from pyscf import __config__

# Grabs the global SEED variable and creates the random number generator
SEED = getattr(__config__, 'SEED', None)
rng = np.random.Generator(np.random.PCG64(SEED))

def set_seed(seed):
    '''Sets the seed for the random number generator used by the md module'''
    global rng
    rng = np.random.Generator(np.random.PCG64(seed))

from pyscf.md import integrators, distributions

NVE = integrators.VelocityVerlet

