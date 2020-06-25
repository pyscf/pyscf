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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY ND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Yang Gao <younggao1994@gmail.com>
#     Qiming Sun <osirpt.sun@gmail.com>

'''
KUCCSD with CTF
'''
import time
import numpy
import ctf
from functools import reduce
from pyscf.lib import logger
from pyscf import lib
from pyscf.pbc.mp.kump2 import (get_frozen_mask, get_nocc, get_nmo,
                                padded_mo_coeff, padding_k_idx)
from pyscf.pbc.lib.kpts_helper import gamma_point
from pyscf.pbc.cc.kccsd_rhf import _get_epq
from pyscf.pbc import df
from pyscf.pbc.cc.ccsd import _adjust_occ
import pyscf.pbc.tools.pbc as tools
from pyscf.ctfcc import uccsd, kccsd_rhf, mpi_helper
from pyscf.ctfcc.integrals import ao2mo
from pyscf.pbc.cc import kccsd_uhf
from symtensor.sym_ctf import einsum, tensor, zeros


comm = mpi_helper.comm
rank = mpi_helper.rank
size = mpi_helper.size

def energy(mycc, t1, t2, eris):
    return uccsd.energy(mycc, t1, t2, eris) / mycc.nkpts

def vector_to_amplitudes(mycc, vec, nmo, nocc, nkpts):
    nmoa, nmob = nmo
    nocca, noccb = nocc
    nvira, nvirb = nmoa - nocca, nmob - noccb
    sec1a = nkpts*nocca*nvira
    sec1b = sec1a + nkpts*noccb*nvirb
    sec2a = sec1b + nkpts**3*nocca**2*nvira**2
    sec2ab = sec2a + nkpts**3*nocca*noccb*nvira*nvirb
    sym1, sym2 = mycc._sym1, mycc._sym2
    t1a = tensor(vec[:sec1a].reshape(nkpts,nocca,nvira), sym1)
    t1b = tensor(vec[sec1a:sec1b].reshape(nkpts,noccb,nvirb), sym1)
    t2aa = tensor(vec[sec1b:sec2a].reshape(nkpts,nkpts,nkpts,nocca,nocca,nvira,nvira), sym2)
    t2ab = tensor(vec[sec2a:sec2ab].reshape(nkpts,nkpts,nkpts,nocca,noccb,nvira,nvirb), sym2)
    t2bb = tensor(vec[sec2ab:].reshape(nkpts,nkpts,nkpts,noccb,noccb,nvirb,nvirb), sym2)
    return (t1a, t1b), (t2aa, t2ab, t2bb)


class KUCCSD(kccsd_rhf.KRCCSD):

    update_amps = uccsd.update_amps
    energy = energy
    get_nocc = get_nocc
    get_nmo = get_nmo
    get_frozen_mask = get_frozen_mask

    def init_amps(self, eris):
        nocca, noccb = self.nocc
        nmoa, nmob = self.nmo
        nvira, nvirb=  nmoa - nocca, nmob - noccb
        t1a = zeros([nocca,nvira], sym=self._sym1)
        t1b = zeros([noccb,nvirb], sym=self._sym1)
        
        t2aa = eris.ovov.conj().transpose(0,2,1,3) / eris.eijab
        t2aa-= eris.ovov.conj().transpose(2,0,1,3) / eris.eijab
        t2ab = eris.ovOV.conj().transpose(0,2,1,3) / eris.eiJaB
        t2bb = eris.OVOV.conj().transpose(0,2,1,3) / eris.eIJAB
        t2bb-= eris.OVOV.conj().transpose(2,0,1,3) / eris.eIJAB

        d = 0.0 + 0.j
        d += 0.25*(einsum('iajb,ijab->',eris.ovov,t2aa)
            - einsum('jaib,ijab->',eris.ovov,t2aa))
        d += einsum('iajb,ijab->',eris.ovOV,t2ab)
        d += 0.25*(einsum('iajb,ijab->',eris.OVOV,t2bb)
            - einsum('jaib,ijab->',eris.OVOV,t2bb))

        self.emp2 = d/self.nkpts
        return self.emp2.real, (t1a, t1b), (t2aa,t2ab,t2bb)

    def amplitudes_to_vector(self, t1, t2):
        return uccsd.amplitudes_to_vector(t1, t2)

    def vector_to_amplitudes(self, vector, nmo=None, nocc=None, nkpts=None):
        if nocc is None: nocc = self.nocc
        if nmo is None: nmo = self.nmo
        if nkpts is None: nkpts = self.nkpts
        return vector_to_amplitudes(self, vector, nmo, nocc, nkpts)

    def ao2mo(self, mo_coeff=None):
        return _ChemistsERIs(self, mo_coeff)

    def ipccsd(self, nroots=1, koopmans=True, guess=None, left=False,
               eris=None, imds=None, partition=None, kptlist=None,
               dtype=None, **kwargs):
        from pyscf.ctfcc.eom_kccsd_uhf import EOMIP
        return EOMIP(self).ipccsd(nroots, koopmans, guess, left, \
                                  eris, imds, partition, kptlist, dtype, **kwargs)

    def eaccsd(self, nroots=1, koopmans=True, guess=None, left=False,
               eris=None, imds=None, partition=None, kptlist=None,
               dtype=None, **kwargs):
        from pyscf.ctfcc.eom_kccsd_uhf import EOMEA
        return EOMEA(self).eaccsd(nroots, koopmans, guess, left, \
                                  eris, imds, partition, kptlist, dtype, **kwargs)

class _ChemistsERIs:
    def __init__(self, mycc, mo_coeff=None):
        cput0 = (time.clock(), time.time())
        if mo_coeff is None:
            mo_coeff = mycc.mo_coeff
        mo_coeff = kccsd_uhf.convert_mo_coeff(mo_coeff)
        mo_coeff = padded_mo_coeff(mycc, mo_coeff)

        self.mo_coeff = mo_coeff
        self.nocc = mycc.nocc
        symlib = mycc.symlib

        nkpts = mycc.nkpts
        kpts = mycc.kpts
        nocca, noccb = mycc.nocc
        nmoa, nmob = mycc.nmo
        nvira, nvirb = nmoa - nocca, nmob - noccb

        if gamma_point(mycc.kpts):
            dtype = numpy.double
        else:
            dtype = numpy.complex128
        if rank==0:
            dm = mycc._scf.make_rdm1(mycc.mo_coeff, mycc.mo_occ)
            hcore = mycc._scf.get_hcore()
            with lib.temporary_env(mycc._scf, exxdiv=None):
                vhf = mycc._scf.get_veff(mycc._scf.cell, dm)
            focka = [reduce(numpy.dot, (mo.conj().T, hcore[k]+vhf[0][k], mo))
                    for k, mo in enumerate(mo_coeff[0])]
            fockb = [reduce(numpy.dot, (mo.conj().T, hcore[k]+vhf[1][k], mo))
                    for k, mo in enumerate(mo_coeff[1])]
            e_hf = mycc._scf.energy_tot(dm=dm, vhf=vhf)
            focka = numpy.asarray(focka)
            fockb = numpy.asarray(fockb)
        else:
            focka = fockb = e_hf = None

        focka = comm.bcast(focka, root=0)
        fockb = comm.bcast(fockb, root=0)
        self.e_hf = comm.bcast(e_hf, root=0)
        madelung = tools.madelung(mycc._scf.cell, kpts)
        mo_ea = [focka[k].diagonal().real for k in range(nkpts)]
        mo_eb = [fockb[k].diagonal().real for k in range(nkpts)]
        mo_ea = [_adjust_occ(e, nocca, -madelung) for e in mo_ea]
        mo_eb = [_adjust_occ(e, noccb, -madelung) for e in mo_eb]
        self.mo_energy = (mo_ea, mo_eb)

        mo_e_oa = [e[:nocca] for e in mo_ea]
        mo_e_va = [e[nocca:] + mycc.level_shift for e in mo_ea]
        mo_e_ob = [e[:noccb] for e in mo_eb]
        mo_e_vb = [e[noccb:] + mycc.level_shift for e in mo_eb]

        nonzero_padding_a, nonzero_padding_b = padding_k_idx(mycc, kind="split")
        nonzero_opadding_a, nonzero_vpadding_a = nonzero_padding_a
        nonzero_opadding_b, nonzero_vpadding_b = nonzero_padding_b

        eia = numpy.zeros([nkpts,nocca,nvira])
        eIA = numpy.zeros([nkpts,noccb,nvirb])
        for ki in range(nkpts):
            eia[ki] = _get_epq([0,nocca,ki,mo_e_oa,nonzero_opadding_a],
                        [0,nvira,ki,mo_e_va,nonzero_vpadding_a],
                        fac=[1.0,-1.0])
            eIA[ki] = _get_epq([0,noccb,ki,mo_e_ob,nonzero_opadding_b],
                        [0,nvirb,ki,mo_e_vb,nonzero_vpadding_b],
                        fac=[1.0,-1.0])
        self.eia = ctf.astensor(eia)
        self.eIA = ctf.astensor(eIA)
        focka = ctf.astensor(focka)
        fockb = ctf.astensor(fockb)

        self.foo = tensor(focka[:,:nocca,:nocca], mycc._sym1, symlib=symlib, verbose=mycc.SYMVERBOSE)
        self.fov = tensor(focka[:,:nocca,nocca:], mycc._sym1, symlib=symlib, verbose=mycc.SYMVERBOSE)
        self.fvv = tensor(focka[:,nocca:,nocca:], mycc._sym1, symlib=symlib, verbose=mycc.SYMVERBOSE)
        foo_ = numpy.asarray([numpy.diag(e) for e in mo_e_oa])
        fvv_ = numpy.asarray([numpy.diag(e) for e in mo_e_va])
        self._foo = tensor(ctf.astensor(foo_), mycc._sym1, symlib=symlib, verbose=mycc.SYMVERBOSE)
        self._fvv = tensor(ctf.astensor(fvv_), mycc._sym1, symlib=symlib, verbose=mycc.SYMVERBOSE)

        self.fOO = tensor(fockb[:,:noccb,:noccb], mycc._sym1, symlib=symlib, verbose=mycc.SYMVERBOSE)
        self.fOV = tensor(fockb[:,:noccb,noccb:], mycc._sym1, symlib=symlib, verbose=mycc.SYMVERBOSE)
        self.fVV = tensor(fockb[:,noccb:,noccb:], mycc._sym1, symlib=symlib, verbose=mycc.SYMVERBOSE)
        fOO_ = numpy.asarray([numpy.diag(e) for e in mo_e_ob])
        fVV_ = numpy.asarray([numpy.diag(e) for e in mo_e_vb])
        self._fOO = tensor(ctf.astensor(fOO_), mycc._sym1, symlib=symlib, verbose=mycc.SYMVERBOSE)
        self._fVV = tensor(ctf.astensor(fVV_), mycc._sym1, symlib=symlib, verbose=mycc.SYMVERBOSE)

        tasks = mpi_helper.static_partition(range(nkpts**3))
        ntasks = max(comm.allgather(len(tasks)))
        self.eijab = ctf.zeros([nkpts,nkpts,nkpts,nocca,nocca,nvira,nvira])
        self.eiJaB = ctf.zeros([nkpts,nkpts,nkpts,nocca,noccb,nvira,nvirb])
        self.eIJAB = ctf.zeros([nkpts,nkpts,nkpts,noccb,noccb,nvirb,nvirb])
        kconserv = mycc.khelper.kconserv
        for itask in range(ntasks):
            if itask >= len(tasks):
                self.eijab.write([], [])
                self.eiJaB.write([], [])
                self.eIJAB.write([], [])
                continue
            ki, kj, ka = mpi_helper.unpack_idx(tasks[itask], nkpts, nkpts, nkpts)
            kb = kconserv[ki,ka,kj]
            eia = _get_epq([0,nocca,ki,mo_e_oa,nonzero_opadding_a],
                           [0,nvira,ka,mo_e_va,nonzero_vpadding_a],
                           fac=[1.0,-1.0])
            ejb = _get_epq([0,nocca,kj,mo_e_oa,nonzero_opadding_a],
                           [0,nvira,kb,mo_e_va,nonzero_vpadding_a],
                           fac=[1.0,-1.0])
            eIA = _get_epq([0,noccb,ki,mo_e_ob,nonzero_opadding_b],
                           [0,nvirb,ka,mo_e_vb,nonzero_vpadding_b],
                           fac=[1.0,-1.0])
            eJB = _get_epq([0,noccb,kj,mo_e_ob,nonzero_opadding_b],
                           [0,nvirb,kb,mo_e_vb,nonzero_vpadding_b],
                           fac=[1.0,-1.0])
            eijab = eia[:,None,:,None] + ejb[None,:,None,:]
            eiJaB = eia[:,None,:,None] + eJB[None,:,None,:]
            eIJAB = eIA[:,None,:,None] + eJB[None,:,None,:]
            off = ki * nkpts**2 + kj * nkpts + ka
            self.eijab.write(off*eijab.size+numpy.arange(eijab.size), eijab.ravel())
            self.eiJaB.write(off*eiJaB.size+numpy.arange(eiJaB.size), eiJaB.ravel())
            self.eIJAB.write(off*eIJAB.size+numpy.arange(eIJAB.size), eIJAB.ravel())

        if type(mycc._scf.with_df) is df.FFTDF:
            ao2mo.make_fftdf_eris_uhf(mycc, self)
        else:
            from pyscf.ctfcc.integrals import mpigdf
            if type(mycc._scf.with_df) is mpigdf.GDF:
                ao2mo.make_df_eris_uhf(mycc, self)
            elif type(mycc._scf.with_df) is df.GDF:
                logger.warn(mycc, "GDF converted to an MPIGDF object, \
                                   one process used for reading from disk")
                mycc._scf.with_df = mpigdf.from_serial(mycc._scf.with_df)
                ao2mo.make_df_eris_uhf(mycc, self)
            else:
                raise NotImplementedError("DF object not recognized")
        logger.timer(mycc, "ao2mo transformation", *cput0)

if __name__ == '__main__':
    from pyscf.pbc import gto, scf, cc
    cell = gto.Cell()
    cell.atom='''
    He 0.000000000000   0.000000000000   0.000000000000
    He 1.685068664391   1.685068664391   1.685068664391
    '''
    cell.basis = [[0, (1., 1.)], [0, (.5, 1.)]]
    cell.a = '''
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000'''
    cell.unit = 'B'
    cell.mesh = [13]*3
    cell.verbose= 4
    cell.build()
    kmf = scf.KUHF(cell, kpts=cell.make_kpts([1,1,3]), exxdiv=None)
    if rank==0:
        kmf.kernel()
    mycc = KUCCSD(kmf)
    eris = mycc.ao2mo()
    e, t1, t2 = mycc.kernel()
    print("Energy Error:", e--0.01031579333505543)
