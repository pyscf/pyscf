#!/usr/bin/env python
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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''DMRG program interface.

There are two DMRG program interfaces available:

    * `Block <https://github.com/sanshar/Block>`_ interface provided the
      features including the DMRG-CASCI, the 1-step and 2-step DMRG-CASSCF, second
      order pertubation for dynamic correlation.  1-, 2- and 3-particle density
      matrices.

    * `CheMPS2 <https://github.com/SebWouters/CheMPS2>`_ interface provided the
      DMRG-CASCI and 2-step DMRG-CASSCF.

Simple usage::

    >>> from pyscf import gto, scf, mcscf, dmrgscf, mrpt
    >>> mol = gto.M(atom='C 0 0 0; C 0 0 1', basis='631g')
    >>> mf = scf.RHF(mol).run()
    >>> mc = dmrgscf.DMRGSCF(mf, 4, 4)
    >>> mc.kernel()
    -75.3374492511669
    >>> mrpt.NEVPT(mc).compress_approx().kernel()
    -0.10474250075684

    >>> mc = mcscf.CASSCF(mf, 4, 4)
    >>> mc.fcisolver = dmrgscf.CheMPS2(mol)
    >>> mc.kernel()
    -75.3374492511669

Note a few configurations in ``/path/to/dmrgscf/settings.py`` needs to be made
before using the DMRG interface code.

Block
-----
:class:`DMRGCI` is the main object to hold Block input parameters and results.
:func:`DMRGSCF` is a shortcut function quickly setup DMRG-CASSCF calculation.
:func:`compress_approx` initializes the compressed MPS perturber for NEVPT2
calculation.

In DMRGCI object, you can set the following attributes to control Block program:

    outputlevel : int
        Noise level for Block program output.
    maxIter : int
        Max DMRG sweeps
    approx_maxIter : int
        To control the DMRG-CASSCF approximate DMRG solver accuracy.
    twodot_to_onedot : int
        When to switch from two-dot algroithm to one-dot algroithm.
    nroots : int
        Number of states in the same irreducible representation to compute.
    weights : list of floats
        Use this attribute with "nroots" attribute to set state-average calculation.
    restart : bool
        To control whether to restart a DMRG calculation.
    tol : float
        DMRG convergence tolerence
    maxM : int
        Bond dimension
    scheduleSweeps, scheduleMaxMs, scheduleTols, scheduleNoises : list
        DMRG sweep scheduler.  See also Block documentation
    wfnsym : str or int
        Wave function irrep label or irrep ID
    orbsym : list of int
        irrep IDs of each orbital
    groupname : str
        groupname, orbsym together can control whether to employ symmetry in
        the calculation.  "groupname = None and orbsym = []" requires the
        Block program using C1 symmetry.

CheMPS2
-------
In :class:`CheMPS2`, DMRG calculation can be controlled by:

    | wfn_irrep
    | dmrg_states
    | dmrg_noise
    | dmrg_e_convergence
    | dmrg_noise_factor
    | dmrg_maxiter_noise
    | dmrg_maxiter_silent

See http://sebwouters.github.io/CheMPS2/index.html for more detail usages of
these keywords.
'''

from pyscf.dmrgscf import dmrgci
from pyscf.dmrgscf.dmrgci import DMRGCI, DMRGSCF, dryrun

from pyscf.dmrgscf import chemps2
from pyscf.dmrgscf.chemps2 import CheMPS2

