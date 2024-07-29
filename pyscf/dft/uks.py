#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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

'''
Non-relativistic Unrestricted Kohn-Sham
'''


import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import uhf
from pyscf.dft import rks

def get_veff(ks, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
    '''Coulomb + XC functional for UKS.  See pyscf/dft/rks.py
    :func:`get_veff` fore more details.
    '''
    if mol is None: mol = ks.mol
    if dm is None: dm = ks.make_rdm1()
    if not isinstance(dm, numpy.ndarray):
        dm = numpy.asarray(dm)
    if dm.ndim == 2:  # RHF DM
        dm = numpy.asarray((dm*.5,dm*.5))
    ks.initialize_grids(mol, dm)

    t0 = (logger.process_clock(), logger.perf_counter())

    ground_state = (dm.ndim == 3 and dm.shape[0] == 2)

    ni = ks._numint
    if hermi == 2:  # because rho = 0
        n, exc, vxc = (0,0), 0, 0
    else:
        max_memory = ks.max_memory - lib.current_memory()[0]
        n, exc, vxc = ni.nr_uks(mol, ks.grids, ks.xc, dm, max_memory=max_memory)
        logger.debug(ks, 'nelec by numeric integration = %s', n)
        if ks.nlc or ni.libxc.is_nlc(ks.xc):
            if ni.libxc.is_nlc(ks.xc):
                xc = ks.xc
            else:
                assert ni.libxc.is_nlc(ks.nlc)
                xc = ks.nlc
            n, enlc, vnlc = ni.nr_nlc_vxc(mol, ks.nlcgrids, xc, dm[0]+dm[1],
                                          max_memory=max_memory)
            exc += enlc
            vxc += vnlc
            logger.debug(ks, 'nelec with nlc grids = %s', n)
        t0 = logger.timer(ks, 'vxc', *t0)

    if not ni.libxc.is_hybrid_xc(ks.xc):
        vk = None
        if (ks._eri is None and ks.direct_scf and
            getattr(vhf_last, 'vj', None) is not None):
            ddm = numpy.asarray(dm) - numpy.asarray(dm_last)
            vj = ks.get_j(mol, ddm[0]+ddm[1], hermi)
            vj += vhf_last.vj
        else:
            vj = ks.get_j(mol, dm[0]+dm[1], hermi)
        vxc += vj
    else:
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(ks.xc, spin=mol.spin)
        if (ks._eri is None and ks.direct_scf and
            getattr(vhf_last, 'vk', None) is not None):
            ddm = numpy.asarray(dm) - numpy.asarray(dm_last)
            vj, vk = ks.get_jk(mol, ddm, hermi)
            vk *= hyb
            if omega != 0:
                vklr = ks.get_k(mol, ddm, hermi, omega)
                vklr *= (alpha - hyb)
                vk += vklr
            vj = vj[0] + vj[1] + vhf_last.vj
            vk += vhf_last.vk
        else:
            vj, vk = ks.get_jk(mol, dm, hermi)
            vj = vj[0] + vj[1]
            vk *= hyb
            if omega != 0:
                vklr = ks.get_k(mol, dm, hermi, omega)
                vklr *= (alpha - hyb)
                vk += vklr
        vxc += vj - vk

        if ground_state:
            exc -=(numpy.einsum('ij,ji', dm[0], vk[0]).real +
                   numpy.einsum('ij,ji', dm[1], vk[1]).real) * .5
    if ground_state:
        ecoul = numpy.einsum('ij,ji', dm[0]+dm[1], vj).real * .5
    else:
        ecoul = None

    vxc = lib.tag_array(vxc, ecoul=ecoul, exc=exc, vj=vj, vk=vk)
    return vxc

def get_vsap(ks, mol=None):
    '''Superposition of atomic potentials

    S. Lehtola, Assessment of initial guesses for self-consistent
    field calculations. Superposition of Atomic Potentials: simple yet
    efficient, J. Chem. Theory Comput. 15, 1593 (2019). DOI:
    10.1021/acs.jctc.8b01089. arXiv:1810.11659.

    This function evaluates the effective charge of a neutral atom,
    given by exchange-only LDA on top of spherically symmetric
    unrestricted Hartree-Fock calculations as described in

    S. Lehtola, L. Visscher, E. Engel, Efficient implementation of the
    superposition of atomic potentials initial guess for electronic
    structure calculations in Gaussian basis sets, J. Chem. Phys., in
    press (2020).

    The potentials have been calculated for the ground-states of
    spherically symmetric atoms at the non-relativistic level of theory
    as described in

    S. Lehtola, "Fully numerical calculations on atoms with fractional
    occupations and range-separated exchange functionals", Phys. Rev. A
    101, 012516 (2020). DOI: 10.1103/PhysRevA.101.012516

    using accurate finite-element calculations as described in

    S. Lehtola, "Fully numerical Hartree-Fock and density functional
    calculations. I. Atoms", Int. J. Quantum Chem. e25945 (2019).
    DOI: 10.1002/qua.25945

    .. note::
        This function will modify the input ks object.

    Args:
        ks : an instance of :class:`RKS`
            XC functional are controlled by ks.xc attribute.  Attribute
            ks.grids might be initialized.

    Returns:
        matrix Vsap = Vnuc + J + Vxc.
    '''
    Vsap = rks.get_vsap(ks, mol)
    return numpy.asarray([Vsap, Vsap])

def energy_elec(ks, dm=None, h1e=None, vhf=None):
    if dm is None: dm = ks.make_rdm1()
    if h1e is None: h1e = ks.get_hcore()
    if vhf is None or getattr(vhf, 'ecoul', None) is None:
        vhf = ks.get_veff(ks.mol, dm)
    if not (isinstance(dm, numpy.ndarray) and dm.ndim == 2):
        dm = dm[0] + dm[1]
    return rks.energy_elec(ks, dm, h1e, vhf)


class UKS(rks.KohnShamDFT, uhf.UHF):
    '''Unrestricted Kohn-Sham
    See pyscf/dft/rks.py RKS class for document of the attributes'''
    def __init__(self, mol, xc='LDA,VWN'):
        uhf.UHF.__init__(self, mol)
        rks.KohnShamDFT.__init__(self, xc)

    def dump_flags(self, verbose=None):
        uhf.UHF.dump_flags(self, verbose)
        rks.KohnShamDFT.dump_flags(self, verbose)
        return self

    def initialize_grids(self, mol=None, dm=None):
        ground_state = (isinstance(dm, numpy.ndarray)
                        and dm.ndim == 3 and dm.shape[0] == 2)
        if ground_state:
            super().initialize_grids(mol, dm[0]+dm[1])
        else:
            super().initialize_grids(mol)
        return self

    get_veff = get_veff
    get_vsap = get_vsap
    energy_elec = energy_elec

    init_guess_by_vsap = rks.init_guess_by_vsap

    def nuc_grad_method(self):
        from pyscf.grad import uks
        return uks.Gradients(self)

    def to_hf(self):
        '''Convert to UHF object.'''
        return self._transfer_attrs_(self.mol.UHF())
