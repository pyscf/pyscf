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
KRCCSD with CTF
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
from pyscf.pbc.mp.kmp2 import (get_frozen_mask, get_nocc, get_nmo,
                               padded_mo_coeff, padding_k_idx)
from pyscf.ctfcc.integrals import ao2mo
import pyscf.pbc.tools.pbc as tools
from pyscf.ctfcc import rccsd, mpi_helper
from symtensor.sym_ctf import tensor, einsum, zeros
from symtensor.symlib import SYMLIB

comm = mpi_helper.comm
rank = mpi_helper.rank
size = mpi_helper.size

def energy(mycc, t1, t2, eris):
    nkpts = mycc.nkpts
    e = 2*einsum('ia,ia', eris.fov, t1)
    tau = einsum('ia,jb->ijab',t1,t1)
    tau += t2
    e += 2*einsum('ijab,iajb', tau, eris.ovov)
    e +=  -einsum('ijab,ibja', tau, eris.ovov)
    if abs(e.imag)>1e-4:
        logger.warn(mycc, 'Non-zero imaginary part found in KRCCSD energy %s', e)
    return e.real / nkpts

def init_amps(mycc, eris):
    time0 = time.clock(), time.time()
    nocc = mycc.nocc
    nvir = mycc.nmo - nocc
    nkpts = mycc.nkpts
    t1 = zeros([nocc,nvir], eris.dtype, mycc._sym1)
    t2 = eris.ovov.transpose(0,2,1,3).conj() / eris.eijab
    mycc.emp2  = 2*einsum('ijab,iajb', t2, eris.ovov)
    mycc.emp2 -=   einsum('ijab,ibja', t2, eris.ovov)
    mycc.emp2  =   mycc.emp2.real/nkpts
    logger.info(mycc, 'Init t2, MP2 energy (with fock eigenvalue shift) = %.15g', mycc.emp2)
    logger.timer(mycc, 'init mp2', *time0)
    return mycc.emp2, t1, t2

class KRCCSD(rccsd.RCCSD):
    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None, SYMVERBOSE=0):
        rccsd.RCCSD.__init__(self, mf, frozen, mo_coeff, mo_occ, SYMVERBOSE)
        self.kpts = mf.kpts
        self.khelper = kpts_helper.KptsHelper(mf.cell, mf.kpts)
        self.max_space = 20
        self._keys = self._keys.union(['max_space'])
        self.symlib = SYMLIB('ctf')
        self.make_symlib()

    energy = energy
    init_amps = init_amps
    get_nocc = get_nocc
    get_nmo = get_nmo
    get_frozen_mask = get_frozen_mask
    padding_k_idx

    @property
    def nkpts(self):
        return len(self.kpts)

    @property
    def _sym1(self):
        '''
        Descriptor of Kpoint symmetry in T1
        (ki - ka) mod G == 0
        '''
        kpts = self.kpts
        gvec = self._scf.cell.reciprocal_vectors()
        return ['+-',[kpts,]*2, None, gvec]

    @property
    def _sym2(self):
        '''
        Descriptor of Kpoint symmetry in T2
        (ki + kj - ka - kb) mod G == 0
        '''
        kpts = self.kpts
        gvec = self._scf.cell.reciprocal_vectors()
        return ['++--',[kpts,]*4, None, gvec]

    def vector_to_amplitudes(self, vec, nmo=None, nocc=None):
        if nocc is None: nocc = self.nocc
        if nmo is None: nmo = self.nmo
        nvir = nmo - nocc
        nkpts = self.nkpts
        nov = nkpts * nocc * nvir
        t1 = vec[:nov].reshape(nkpts,nocc,nvir)
        t2 = vec[nov:].reshape(nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir)
        kpts = self.kpts
        gvec = self._scf.cell.reciprocal_vectors()
        t1  = tensor(t1, self._sym1, symlib=self.symlib, verbose=self.SYMVERBOSE)
        t2  = tensor(t2, self._sym2, symlib=self.symlib, verbose=self.SYMVERBOSE)
        return t1, t2

    def make_symlib(self):
        '''
        Pre-compute all transformation deltas needed in KCCSD iterations
        '''
        kpts = self.kpts
        gvec = self._scf.cell.reciprocal_vectors()
        sym3 = ['++-',[kpts,]*3, None, gvec]
        self.symlib.update(self._sym1, self._sym2, sym3)

    def ao2mo(self, mo_coeff=None):
        return _ChemistsERIs(self, mo_coeff)

    def ipccsd(self, nroots=1, koopmans=True, guess=None, left=False,
               eris=None, imds=None, partition=None, kptlist=None,
               dtype=None, **kwargs):
        from pyscf.ctfcc.eom_kccsd_rhf import EOMIP
        return EOMIP(self).ipccsd(nroots, koopmans, guess, left, \
                                  eris, imds, partition, kptlist, dtype, **kwargs)

    def eaccsd(self, nroots=1, koopmans=True, guess=None, left=False,
               eris=None, imds=None, partition=None, kptlist=None,
               dtype=None, **kwargs):
        from pyscf.ctfcc.eom_kccsd_rhf import EOMEA
        return EOMEA(self).eaccsd(nroots, koopmans, guess, left, \
                                  eris, imds, partition, kptlist, dtype, **kwargs)

class _ChemistsERIs:
    def __init__(self, mycc, mo_coeff=None):
        from pyscf.pbc.cc.ccsd import _adjust_occ
        cput0 = (time.clock(), time.time())
        nocc, nmo, nkpts = mycc.nocc, mycc.nmo, mycc.nkpts
        nvir = nmo - nocc
        cell, kpts = mycc._scf.cell, mycc.kpts
        symlib = mycc.symlib
        gvec = cell.reciprocal_vectors()
        sym2 = ['+-+-', [kpts,]*4, None, gvec]

        nonzero_opadding, nonzero_vpadding = padding_k_idx(mycc, kind="split")
        madelung = tools.madelung(cell, kpts)
        if rank==0:
            if mo_coeff is None:
                mo_coeff = mycc.mo_coeff
            cell = mycc._scf.cell
            self.mo_coeff = mo_coeff = padded_mo_coeff(mycc, mo_coeff)
            dm = mycc._scf.make_rdm1(mycc.mo_coeff, mycc.mo_occ)
            with lib.temporary_env(mycc._scf, exxdiv=None):
                fockao = mycc._scf.get_hcore() + mycc._scf.get_veff(cell, dm)
            self.fock = numpy.asarray([reduce(numpy.dot, (mo.T.conj(), fockao[k], mo))
                                for k, mo in enumerate(mo_coeff)])
        else:
            self.fock = self.mo_coeff = mo_coeff = None

        comm.barrier()

        fock = comm.bcast(self.fock, root=0)
        mo_energy = [fock[k].diagonal().real for k in range(nkpts)]
        mo_energy = [_adjust_occ(mo_e, nocc, -madelung)
                          for k, mo_e in enumerate(mo_energy)]
        mo_e_o = [e[:nocc] for e in mo_energy]
        mo_e_v = [e[nocc:] + mycc.level_shift for e in mo_energy]
        eia = numpy.zeros([nkpts,nocc,nvir])
        for ki in range(nkpts):
            eia[ki] = _get_epq([0,nocc,ki,mo_e_o,nonzero_opadding],
                        [0,nvir,ki,mo_e_v,nonzero_vpadding],
                        fac=[1.0,-1.0])

        mo_coeff = self.mo_coeff = comm.bcast(self.mo_coeff, root=0)
        self.dtype = dtype = numpy.result_type(*(mo_coeff, fock)).char
        fock = ctf.astensor(fock)
        self.foo = tensor(fock[:,:nocc,:nocc], mycc._sym1, symlib=symlib, verbose=mycc.SYMVERBOSE)
        self.fov = tensor(fock[:,:nocc,nocc:], mycc._sym1, symlib=symlib, verbose=mycc.SYMVERBOSE)
        self.fvv = tensor(fock[:,nocc:,nocc:], mycc._sym1, symlib=symlib, verbose=mycc.SYMVERBOSE)

        self.eia = ctf.astensor(eia)
        foo_ = numpy.asarray([numpy.diag(e) for e in mo_e_o])
        fvv_ = numpy.asarray([numpy.diag(e) for e in mo_e_v])
        self._foo = tensor(ctf.astensor(foo_), mycc._sym1, symlib=symlib, verbose=mycc.SYMVERBOSE)
        self._fvv = tensor(ctf.astensor(fvv_), mycc._sym1, symlib=symlib, verbose=mycc.SYMVERBOSE)

        self.eijab = ctf.zeros([nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir])

        kconserv = mycc.khelper.kconserv
        khelper = mycc.khelper

        idx_oovv = numpy.arange(nocc*nocc*nvir*nvir)
        jobs = list(khelper.symm_map.keys())
        tasks = mpi_helper.static_partition(jobs)
        ntasks = max(comm.allgather(len(tasks)))
        nwrite = 0
        for itask in tasks:
            ikp, ikq, ikr = itask
            pqr = numpy.asarray(khelper.symm_map[(ikp,ikq,ikr)])
            nwrite += len(numpy.unique(pqr, axis=0))

        nwrite_max = max(comm.allgather(nwrite))
        write_count = 0
        for itask in range(ntasks):
            if itask >= len(tasks):
                continue
            ikp, ikq, ikr = tasks[itask]
            iks = kconserv[ikp,ikq,ikr]
            done = numpy.zeros([nkpts,nkpts,nkpts])
            for (kp, kq, kr) in khelper.symm_map[(ikp, ikq, ikr)]:
                if done[kp,kq,kr]: continue
                ks = kconserv[kp,kq,kr]
                eia = _get_epq([0,nocc,kp,mo_e_o,nonzero_opadding],
                               [0,nvir,kq,mo_e_v,nonzero_vpadding],
                               fac=[1.0,-1.0])
                ejb = _get_epq([0,nocc,kr,mo_e_o,nonzero_opadding],
                               [0,nvir,ks,mo_e_v,nonzero_vpadding],
                               fac=[1.0,-1.0])
                eijab = eia[:,None,:,None] + ejb[None,:,None,:]
                off = kp * nkpts**2 + kr * nkpts + kq
                self.eijab.write(off*idx_oovv.size+idx_oovv, eijab.ravel())
                done[kp,kq,kr] = 1
                write_count += 1

        for i in range(nwrite_max-write_count):
            self.eijab.write([], [])

        if type(mycc._scf.with_df) is df.FFTDF:
            ao2mo.make_fftdf_eris_rhf(mycc, self)
        else:
            from pyscf.ctfcc.integrals import mpigdf
            if type(mycc._scf.with_df) is mpigdf.GDF:
                ao2mo.make_df_eris_rhf(mycc, self)
            elif type(mycc._scf.with_df) is df.GDF:
                logger.warn(mycc, "GDF converted to an MPIGDF object, \
                                   one process used for reading from disk")
                mycc._scf.with_df = mpigdf.from_serial(mycc._scf.with_df)
                ao2mo.make_df_eris_rhf(mycc, self)
            else:
                raise NotImplementedError("DF object not recognized")
        logger.timer(mycc, "ao2mo transformation", *cput0)

if __name__ == '__main__':
    from pyscf.pbc import gto, scf
    cell = gto.Cell()
    cell.atom='''
    H 0.000000000000   0.000000000000   0.000000000000
    H 1.685068664391   1.685068664391   1.685068664391
    '''
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.a = '''
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000'''
    cell.unit = 'B'
    cell.mesh = [15,15,15]
    cell.verbose = 4
    cell.build()

    kpts = cell.make_kpts([1,1,3])
    mf = scf.KRHF(cell,kpts, exxdiv=None)

    if rank==0:
        mf.kernel()

    mycc = KRCCSD(mf)
    ecc = mycc.kernel()[0]
    print(numpy.amax(ecc - -0.016312367049186628))
