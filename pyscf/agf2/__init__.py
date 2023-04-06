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
#
# Author: Oliver Backhouse <olbackhouse@gmail.com>
#         George Booth <george.booth@kcl.ac.uk>
#

'''
Auxiliary second-order Green's function perturbation therory
============================================================

The AGF2 method permits the computation of quasiparticle excitations and 
ground-state properties at the AGF2(None,0) level. 

When using results of this code for publications, please cite the follow papers:
"Wave function perspective and efficient truncation of renormalized second-order perturbation theory", O. J. Backhouse, M. Nusspickel and G. H. Booth, J. Chem. Theory Comput., 16, 1090 (2020).
"Efficient excitations and spectra within a perturbative renormalization approach", O. J. Backhouse and G. H. Booth, J. Chem. Theory Comput., 16, 6294 (2020).


Simple usage::

    >>> from pyscf import gto, scf, agf2
    >>> mol = gto.M(atom='H 0 0 0; H 0 0 1')
    >>> mf = scf.RHF(mol).run()
    >>> gf2 = agf2.AGF2(mf).run()
    >>> gf2.ipagf2()

:func:`agf2.AGF2` returns an instance of AGF2 class.  Valid parameters to
control the AGF2 method are:

    verbose : int
        Print level. Default value equals to :class:`Mole.verbose`
    max_memory : float or int
        Allowed memory in MB. Default value equals to :class:`Mole.max_memory`
    conv_tol : float
        Convergence threshold for AGF2 energy. Default value is 1e-7
    conv_tol_rdm1 : float
        Convergence threshold for first-order reduced density matrix.
        Default value is 1e-6.
    conv_tol_nelec : float
        Convergence threshold for the number of electrons. Default 
        value is 1e-6.
    max_cycle : int
        Maximum number of AGF2 iterations. Default value is 50.
    max_cycle_outer : int
        Maximum number of outer Fock loop iterations. Default 
        value is 20.
    max_cycle_inner : int
        Maximum number of inner Fock loop iterations. Default
        value is 50.
    weight_tol : float
        Threshold in spectral weight of auxiliaries to be considered
        zero. Default 1e-11.
    diis_space : int
        DIIS space size for Fock loop iterations. Default value is 6.
    diis_min_space : 
        Minimum space of DIIS. Default value is 1.

Saved result

    e_corr : float
        AGF2 correlation energy
    e_tot : float
        Total energy (HF + correlation)
    e_1b : float
        One-body part of :attr:`e_tot`
    e_2b : float
        Two-body part of :attr:`e_tot`
    e_init : float
        Initial correlation energy (truncated MP2)
    converged : bool
        Whether convergence was successful
    se : SelfEnergy
        Auxiliaries of the self-energy
    gf : GreensFunction
        Auxiliaries of the Green's function
'''

from pyscf import scf, lib
from pyscf.agf2 import aux_space, ragf2, uagf2, dfragf2, dfuagf2, ragf2_slow, uagf2_slow
from pyscf.agf2.aux_space import AuxiliarySpace, GreensFunction, SelfEnergy

# Backwards compatibility:
aux = aux_space


def AGF2(mf, nmom=(None,0), frozen=None, mo_energy=None, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.uhf.UHF):
        return UAGF2(mf, nmom, frozen, mo_energy, mo_coeff, mo_occ)

    elif isinstance(mf, scf.rohf.ROHF):
        lib.logger.warn(mf, 'RAGF2 method does not support ROHF reference. '
                            'Converting to UHF and using UAGF2.')
        mf = scf.addons.convert_to_uhf(mf)
        return UAGF2(mf, nmom, frozen, mo_energy, mo_coeff, mo_occ)

    elif isinstance(mf, scf.rhf.RHF):
        return RAGF2(mf, nmom, frozen, mo_energy, mo_coeff, mo_occ)

    else:
        raise RuntimeError('AGF2 code only supports RHF, ROHF and UHF references')

AGF2.__doc__ = ragf2.RAGF2.__doc__


def RAGF2(mf, nmom=(None,0), frozen=None, mo_energy=None, mo_coeff=None, mo_occ=None):
    if nmom != (None,0): # redundant
        if nmom[1] == 0 and nmom[0] != 0:
            nmom = (None,0)
    
    if nmom != (None,0) and getattr(mf, 'with_df', None) is not None:
        raise RuntimeError('AGF2 with custom moment orders does not '
                           'density fitting.')

    elif nmom != (None,0):
        lib.logger.warn(mf, 'AGF2 called with custom moment orders - '
                            'falling back on _slow implementations.')
        return ragf2_slow.RAGF2(mf, nmom, frozen, mo_energy, mo_coeff, mo_occ)

    elif getattr(mf, 'with_df', None) is not None:
        return dfragf2.DFRAGF2(mf, frozen, mo_energy, mo_coeff, mo_occ)

    else:
        return ragf2.RAGF2(mf, frozen, mo_energy, mo_coeff, mo_occ)

RAGF2.__doc__ = ragf2.RAGF2.__doc__


def UAGF2(mf, nmom=(None,0), frozen=None, mo_energy=None, mo_coeff=None, mo_occ=None):
    if nmom != (None,0): # redundant
        if nmom[1] == 0 and nmom[0] != 0:
            nmom = (None,0)
    
    if nmom != (None,0) and getattr(mf, 'with_df', None) is not None:
        raise RuntimeError('AGF2 with custom moment orders does not '
                           'density fitting.')

    elif nmom != (None,0):
        lib.logger.warn(mf, 'AGF2 called with custom moment orders - '
                            'falling back on _slow implementations.')
        return uagf2_slow.UAGF2(mf, nmom, frozen, mo_energy, mo_coeff, mo_occ)

    elif getattr(mf, 'with_df', None) is not None:
        return dfuagf2.DFUAGF2(mf, frozen, mo_energy, mo_coeff, mo_occ)

    else:
        return uagf2.UAGF2(mf, frozen, mo_energy, mo_coeff, mo_occ)

UAGF2.__doc__ = uagf2.UAGF2.__doc__
