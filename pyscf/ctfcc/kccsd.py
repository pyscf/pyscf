#!/usr/bin/env python
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
# Author: Yang Gao <younggao1994@gmail.com>
#         Qiming Sun <osirpt.sun@gmail.com>

'''
KGCCSD with CTF
'''

import numpy
import ctf
import time
from functools import reduce
from pyscf.lib import logger
from pyscf import lib
from pyscf.pbc import df
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.cc.kccsd_rhf import _get_epq
import pyscf.pbc.tools.pbc as tools
from pyscf.pbc.mp.kmp2 import (get_frozen_mask, get_nocc, get_nmo,
                               padded_mo_coeff, padding_k_idx)
from pyscf.ctfcc.integrals import ao2mo
from pyscf.ctfcc import kccsd_rhf, gccsd, mpi_helper
from symtensor.sym_ctf import tensor, einsum, zeros

comm = mpi_helper.comm
rank = mpi_helper.rank
size = mpi_helper.size

def energy(mycc, t1, t2, eris):
    return gccsd.energy(mycc, t1, t2, eris).real / mycc.nkpts

class KGCCSD(kccsd_rhf.KRCCSD):

    update_amps = gccsd.update_amps
    energy = energy
    get_nocc = get_nocc
    get_nmo = get_nmo
    get_frozen_mask = get_frozen_mask
    padding_k_idx = padding_k_idx

    def init_amps(self, eris=None):
        nocc = self.nocc
        nmo = self.nmo
        nvir = nmo - nocc
        if eris is None:
            eris = self.ao2mo(self.mo_coeff)
        t1 = zeros([nocc,nvir], sym=self._sym1)
        t2 = eris.oovv.conj() / eris.eijab
        self.emp2 = 0.25*einsum('ijab,ijab', t2, eris.oovv).real / self.nkpts
        logger.info(self, 'Init t2, MP2 energy = %.15g', self.emp2)
        return self.emp2, t1, t2

    def ao2mo(self, mo_coeff=None):
        return _PhysicistsERIs(self, mo_coeff)

    def ipccsd(self, nroots=1, koopmans=True, guess=None, left=False,
               eris=None, imds=None, partition=None, kptlist=None,
               dtype=None, **kwargs):
        from pyscf.ctfcc.eom_kccsd_ghf import EOMIP
        return EOMIP(self).ipccsd(nroots, koopmans, guess, left, \
                                  eris, imds, partition, kptlist, dtype, **kwargs)

    def eaccsd(self, nroots=1, koopmans=True, guess=None, left=False,
               eris=None, imds=None, partition=None, kptlist=None,
               dtype=None, **kwargs):
        from pyscf.ctfcc.eom_kccsd_ghf import EOMEA
        return EOMEA(self).eaccsd(nroots, koopmans, guess, left, \
                                  eris, imds, partition, kptlist, dtype, **kwargs)

class _PhysicistsERIs:
    def __init__(self, mycc, mo_coeff=None):
        from pyscf.pbc.cc.ccsd import _adjust_occ
        cput0 = (time.clock(), time.time())
        nocc, nmo, nkpts = mycc.nocc, mycc.nmo, mycc.nkpts
        nvir = nmo - nocc
        cell, kpts = mycc._scf.cell, mycc.kpts
        nao = cell.nao_nr()
        symlib = mycc.symlib
        gvec = cell.reciprocal_vectors()
        sym2 = ['+-+-', [kpts,]*4, None, gvec]
        nonzero_opadding, nonzero_vpadding = padding_k_idx(mycc, kind="split")
        if mo_coeff is None: mo_coeff = mycc._scf.mo_coeff
        nao = mo_coeff[0].shape[0]
        dtype = mo_coeff[0].dtype
        moidx = mycc.get_frozen_mask()
        nocc_per_kpt = numpy.asarray(mycc.get_nocc(per_kpoint=True))
        nmo_per_kpt  = numpy.asarray(mycc.get_nmo(per_kpoint=True))

        padded_moidx = []
        for k in range(nkpts):
            kpt_nocc = nocc_per_kpt[k]
            kpt_nvir = nmo_per_kpt[k] - kpt_nocc
            kpt_padded_moidx = numpy.concatenate((numpy.ones(kpt_nocc, dtype=numpy.bool),
                                                  numpy.zeros(nmo - kpt_nocc - kpt_nvir, dtype=numpy.bool),
                                                  numpy.ones(kpt_nvir, dtype=numpy.bool)))
            padded_moidx.append(kpt_padded_moidx)

        self.mo_coeff = []
        self.orbspin = []
        self.kconserv = mycc.khelper.kconserv
        # Generate the molecular orbital coefficients with the frozen orbitals masked.
        # Each MO is tagged with orbspin, a list of 0's and 1's that give the overall
        # spin of each MO.
        #
        # Here we will work with two index arrays; one is for our original (small) moidx
        # array while the next is for our new (large) padded array.
        for k in range(nkpts):
            kpt_moidx = moidx[k]
            kpt_padded_moidx = padded_moidx[k]

            mo = numpy.zeros((nao, nmo), dtype=dtype)
            mo[:, kpt_padded_moidx] = mo_coeff[k][:, kpt_moidx]
            if getattr(mo_coeff[k], 'orbspin', None) is not None:
                orbspin_dtype = mo_coeff[k].orbspin[kpt_moidx].dtype
                orbspin = numpy.zeros(nmo, dtype=orbspin_dtype)
                orbspin[kpt_padded_moidx] = mo_coeff[k].orbspin[kpt_moidx]
                mo = lib.tag_array(mo, orbspin=orbspin)
                self.orbspin.append(orbspin)
            else:  # guess orbital spin - assumes an RHF calculation
                assert (numpy.count_nonzero(kpt_moidx) % 2 == 0)
                orbspin = numpy.zeros(mo.shape[1], dtype=int)
                orbspin[1::2] = 1
                mo = lib.tag_array(mo, orbspin=orbspin)
                self.orbspin.append(orbspin)
            self.mo_coeff.append(mo)
        self.mo_coeff = comm.bcast(self.mo_coeff, root=0)
        self.orbspin = comm.bcast(self.orbspin, root=0)

        # Re-make our fock MO matrix elements from density and fock AO
        if rank==0:
            dm = mycc._scf.make_rdm1(mycc.mo_coeff, mycc.mo_occ)
            with lib.temporary_env(mycc._scf, exxdiv=None):
                # _scf.exxdiv affects eris.fock. HF exchange correction should be
                # excluded from the Fock matrix.
                vhf = mycc._scf.get_veff(cell, dm)
            fockao = mycc._scf.get_hcore() + vhf
            fock = numpy.asarray([reduce(numpy.dot, (mo.T.conj(), fockao[k], mo))
                                       for k, mo in enumerate(self.mo_coeff)])
            e_hf = mycc._scf.energy_tot(dm=dm, vhf=vhf)
            mo_energy = [fock[k].diagonal().real for k in range(nkpts)]
        else:
            fock = e_hf = mo_energy = None
        fock = comm.bcast(fock, root=0)
        self.e_hf = comm.bcast(e_hf, root=0)
        self.mo_energy = comm.bcast(mo_energy, root=0)
        # Add HFX correction in the eris.mo_energy to improve convergence in
        # CCSD iteration. It is useful for the 2D systems since their occupied and
        # the virtual orbital energies may overlap which may lead to numerical
        # issue in the CCSD iterations.
        # FIXME: Whether to add this correction for other exxdiv treatments?
        # Without the correction, MP2 energy may be largely off the correct value.
        madelung = tools.madelung(cell, kpts)
        self.mo_energy = [_adjust_occ(mo_e, nocc, -madelung)
                          for k, mo_e in enumerate(self.mo_energy)]

        # Get location of padded elements in occupied and virtual space.
        nocc_per_kpt = mycc.get_nocc(per_kpoint=True)
        nonzero_padding = mycc.padding_k_idx(kind="joint")

        # Check direct and indirect gaps for possible issues with CCSD convergence.
        mo_e = [self.mo_energy[kp][nonzero_padding[kp]] for kp in range(nkpts)]
        mo_e = numpy.sort([y for x in mo_e for y in x])  # Sort de-nested array
        gap = mo_e[numpy.sum(nocc_per_kpt)] - mo_e[numpy.sum(nocc_per_kpt)-1]
        if gap < 1e-5:
            logger.warn(mycc, 'HOMO-LUMO gap %s too small for KCCSD. '
                            'May cause issues in convergence.', gap)
        fock =  ctf.astensor(fock)
        self.foo = tensor(fock[:,:nocc,:nocc], mycc._sym1, symlib=mycc.symlib, verbose=mycc.SYMVERBOSE)
        self.fov = tensor(fock[:,:nocc,nocc:], mycc._sym1, symlib=mycc.symlib, verbose=mycc.SYMVERBOSE)
        self.fvv = tensor(fock[:,nocc:,nocc:], mycc._sym1, symlib=mycc.symlib, verbose=mycc.SYMVERBOSE)
        nonzero_opadding, nonzero_vpadding = mycc.padding_k_idx(kind='split')
        mo_e_o = [e[:nocc] for e in self.mo_energy]
        mo_e_v = [e[nocc:] + mycc.level_shift for e in self.mo_energy]
        eia = numpy.zeros([nkpts,nocc,nvir])
        for ki in range(nkpts):
            eia[ki] = _get_epq([0,nocc,ki,mo_e_o,nonzero_opadding],
                        [0,nvir,ki,mo_e_v,nonzero_vpadding],
                        fac=[1.0,-1.0])
        self.eia = ctf.astensor(eia)
        foo_ = numpy.asarray([numpy.diag(e) for e in mo_e_o])
        fvv_ = numpy.asarray([numpy.diag(e) for e in mo_e_v])
        self._foo = ctf.astensor(foo_)
        self._fvv = ctf.astensor(fvv_)
        self.eijab = ctf.zeros([nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir])
        tasks = mpi_helper.static_partition(range(nkpts**3))
        ntasks = max(comm.allgather(len(tasks)))
        kconserv = mycc.khelper.kconserv
        for itask in range(ntasks):
            if itask >= len(tasks):
                self.eijab.write([], [])
                continue
            ki, kj, ka = mpi_helper.unpack_idx(tasks[itask], nkpts, nkpts, nkpts)
            kb = kconserv[ki,ka,kj]
            eia = _get_epq([0,nocc,ki,mo_e_o,nonzero_opadding],
                           [0,nvir,ka,mo_e_v,nonzero_vpadding],
                           fac=[1.0,-1.0])
            ejb = _get_epq([0,nocc,kj,mo_e_o,nonzero_opadding],
                           [0,nvir,kb,mo_e_v,nonzero_vpadding],
                           fac=[1.0,-1.0])
            eijab = eia[:,None,:,None] + ejb[None,:,None,:]
            off = ki * nkpts**2 + kj * nkpts + ka
            self.eijab.write(off*eijab.size+numpy.arange(eijab.size), eijab.ravel())

        if type(mycc._scf.with_df) is df.FFTDF:
            ao2mo.make_fftdf_eris_ghf(mycc, self)
        else:
            from pyscf.ctfcc.integrals import mpigdf
            if type(mycc._scf.with_df) is mpigdf.GDF:
                ao2mo.make_df_eris_ghf(mycc, self)
            elif type(mycc._scf.with_df) is df.GDF:
                logger.warn(mycc, "GDF converted to an MPIGDF object, \
                                   one process used for reading from disk")
                mycc._scf.with_df = mpigdf.from_serial(mycc._scf.with_df)
                ao2mo.make_df_eris_ghf(mycc, self)
            else:
                raise NotImplementedError("DF object not recognized")
        logger.timer(mycc, "ao2mo transformation", *cput0)

if __name__ == '__main__':
    from pyscf.pbc import gto, scf, cc
    cell = gto.Cell()
    cell.atom='''
    H 0.000000000000   0.000000000000   0.000000000000
    H 1.685068664391   1.685068664391   1.685068664391
    '''
    cell.basis = 'gth-dzv'
    cell.pseudo = 'gth-pade'
    cell.a = '''
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000'''
    cell.unit = 'B'
    cell.mesh = [7,7,7]
    cell.verbose = 4
    cell.build()

    kpts = cell.make_kpts([1,1,3])
    mf = scf.KGHF(cell,kpts, exxdiv=None)

    if rank==0:
        mf.kernel()

    mycc = KGCCSD(mf)
    ecc = mycc.kernel()[0]
    print(ecc - -0.09528576800989746)
