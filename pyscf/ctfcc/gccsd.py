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
Mole GCCSD with CTF
'''

import numpy
import ctf
import time
from functools import reduce
from pyscf.cc import ccsd
from pyscf import lib, scf
from pyscf.lib import logger
import pyscf.ctfcc.gintermediates as imd
from pyscf.ctfcc.integrals.ao2mo import _make_ao_ints
from pyscf.ctfcc import rccsd, mpi_helper
from pyscf.ctfcc.mpi_helper import rank, size, comm
from symtensor.sym_ctf import tensor, einsum

def energy(mycc, t1, t2, eris):
    e = einsum('ia,ia', eris.fov, t1)
    e += 0.25*einsum('ijab,ijab', t2, eris.oovv)
    tmp = einsum('ia,ijab->jb', t1, eris.oovv)
    e += 0.5*einsum('jb,jb->', t1, tmp)
    return e

def update_amps(mycc, t1, t2, eris):

    tau = imd.make_tau(t2, t1, t1, eris)

    Fvv = imd.cc_Fvv(t1, t2, eris)
    Foo = imd.cc_Foo(t1, t2, eris)
    Fov = imd.cc_Fov(t1, t2, eris)

    Woooo = imd.cc_Woooo(t1, t2, eris)
    Wvvvv = imd.cc_Wvvvv(t1, t2, eris)
    Wovvo = imd.cc_Wovvo(t1, t2, eris)

    # Move energy terms to the other side
    Fvv -= eris._fvv
    Foo -= eris._foo

    # T1 equation
    t1new  =  einsum('ie,ae->ia', t1, Fvv)
    t1new += -einsum('ma,mi->ia', t1, Foo)
    t1new +=  einsum('imae,me->ia', t2, Fov)
    t1new += -einsum('nf,naif->ia', t1, eris.ovov)
    t1new += -0.5*einsum('imef,maef->ia', t2, eris.ovvv)
    t1new += -0.5*einsum('mnae,mnie->ia', t2, eris.ooov)
    t1new += eris.fov.conj()

    # T2 equation
    Ftmp = Fvv - 0.5*einsum('mb,me->be', t1, Fov)
    tmp = einsum('ijae,be->ijab', t2, Ftmp)
    t2new = tmp - tmp.transpose(0,1,3,2)
    Ftmp = Foo + 0.5*einsum('je,me->mj', t1, Fov)
    tmp = einsum('imab,mj->ijab', t2, Ftmp)
    t2new -= tmp - tmp.transpose(1,0,2,3)
    t2new += eris.oovv.conj()
    t2new += 0.5*einsum('mnab,mnij->ijab', tau, Woooo)
    t2new += 0.5*einsum('ijef,abef->ijab', tau, Wvvvv)
    tmp = einsum('imae,mbej->ijab', t2, Wovvo)
    tmp1 = einsum('ma,mbje->abje', t1, eris.ovov)
    tmp += einsum('ie,abje->ijab', t1, tmp1)
    tmp = tmp - tmp.transpose(1,0,2,3)
    tmp = tmp - tmp.transpose(0,1,3,2)
    t2new += tmp
    tmp = einsum('ie,jeba->ijab', t1, eris.ovvv.conj())
    t2new += (tmp - tmp.transpose(1,0,2,3))
    tmp = einsum('ma,ijmb->ijab', t1, eris.ooov.conj())
    t2new -= (tmp - tmp.transpose(0,1,3,2))

    t1new /= eris.eia
    t2new /= eris.eijab

    return t1new, t2new

class GCCSD(rccsd.RCCSD):

    energy = energy
    update_amps = update_amps

    def init_amps(self, eris=None):
        if eris is None:
            eris = self.ao2mo(self.mo_coeff)
        t1 = eris.fov / eris.eia
        t2 = eris.oovv / eris.eijab
        self.emp2 = 0.25*einsum('ijab,ijab', t2, eris.oovv.conj()).real
        logger.info(self, 'Init t2, MP2 energy = %.15g', self.emp2)
        return self.emp2, t1, t2

    def ao2mo(self, mo_coeff=None):
        if getattr(self._scf, 'with_df', None):
            raise NotImplementedError("DF integrals not supported")
        else:
            return _PhysicistsERIs(self, mo_coeff)

    def ipccsd(self, nroots=1, koopmans=False, guess=None, left=False,
               eris=None, imds=None, **kwargs):
        from pyscf.ctfcc import eom_gccsd
        return eom_gccsd.EOMIP(self).kernel(nroots, koopmans, guess, left, eris,
                                            imds, **kwargs)

    def eaccsd(self, nroots=1, koopmans=False, guess=None, left=False,
               eris=None, imds=None, **kwargs):
        from pyscf.ctfcc import eom_gccsd
        return eom_gccsd.EOMEA(self).kernel(nroots, koopmans, guess, left, eris,
                                            imds, **kwargs)

class _PhysicistsERIs:
    '''<pq||rs> = <pq|rs> - <pq|sr>'''
    def __init__(self, mycc, mo_coeff=None):
        cput1 = cput0 = (time.clock(), time.time())
        self.orbspin = None
        self.mol = mycc.mol
        self.nocc = nocc = mycc.nocc

        if mo_coeff is None:
            mo_coeff = mycc.mo_coeff
        mo_idx = ccsd.get_frozen_mask(mycc)

        if getattr(mo_coeff, 'orbspin', None) is not None:
            self.orbspin = mo_coeff.orbspin[mo_idx]
            mo_coeff = lib.tag_array(mo_coeff[:,mo_idx], orbspin=self.orbspin)
        else:
            orbspin = scf.ghf.guess_orbspin(mo_coeff)
            mo_coeff = mo_coeff[:,mo_idx]
            if not numpy.any(orbspin == -1):
                self.orbspin = orbspin[mo_idx]
                mo_coeff = lib.tag_array(mo_coeff, orbspin=self.orbspin)

        self.mo_coeff = mo_coeff = comm.bcast(mo_coeff, root=0)
        if rank==0:
            dm = mycc._scf.make_rdm1(mycc.mo_coeff, mycc.mo_occ)
            vhf = mycc._scf.get_veff(mycc.mol, dm)
            fockao = mycc._scf.get_fock(vhf=vhf, dm=dm)
            fock = reduce(numpy.dot, (mo_coeff.conj().T, fockao, mo_coeff))
        else:
            fock = None
        comm.barrier()
        fock = comm.bcast(fock, root=0)
        self.dtype = dtype = numpy.result_type(fock)
        mo_e = fock.diagonal().real
        eia  = mo_e[:nocc,None]- mo_e[None,nocc:]
        nmo = mycc.nmo
        nvir = nmo - nocc
        fock = ctf.astensor(fock)
        self.foo = tensor(fock[:nocc,:nocc], verbose=mycc.SYMVERBOSE)
        self.fov = tensor(fock[:nocc,nocc:], verbose=mycc.SYMVERBOSE)
        self.fvv = tensor(fock[nocc:,nocc:], verbose=mycc.SYMVERBOSE)
        self.eia = ctf.astensor(eia)
        gap = abs(eia).min()
        if gap < 1e-5:
            logger.warn(mycc, 'HOMO-LUMO gap %s too small for GCCSD', gap)
        self.e_hf = comm.bcast(mycc.e_hf, root=0)
        self._foo = self.foo.diagonal(preserve_shape=True)
        self._fvv = self.fvv.diagonal(preserve_shape=True)
        self.eijab = self.eia.reshape(nocc,1,nvir,1) + self.eia.reshape(1,nocc,1,nvir)

        assert(dtype==numpy.double)

        nao = mo_coeff.shape[0]
        mo_a = mo_coeff[:nao//2]
        mo_b = mo_coeff[nao//2:]

        def _force_sym(eri, sym_forbid_ij, sym_forbid_kl):
            eri_size = eri.size
            nij = sum(sym_forbid_ij)
            nkl = sum(sym_forbid_kl)
            task_ij = mpi_helper.static_partition(numpy.arange(nij))
            task_kl = mpi_helper.static_partition(numpy.arange(nkl))
            assert(eri_size==sym_forbid_ij.size*sym_forbid_kl.size)
            off_ij = numpy.arange(sym_forbid_ij.size)[sym_forbid_ij] * sym_forbid_kl.size
            off_kl = numpy.arange(sym_forbid_ij.size) * sym_forbid_kl.size
            idx_ij = off_ij[task_ij][:,None] + numpy.arange(sym_forbid_kl.size)
            idx_kl = off_kl[:,None] + numpy.arange(sym_forbid_kl.size)[sym_forbid_kl][task_kl]
            idx_full = numpy.append(idx_ij.ravel(), idx_kl.ravel())
            eri.write(idx_full, numpy.zeros(idx_full.size))
            return eri

        cput1 = cput0 = (time.clock(), time.time())
        ppoo, ppov, ppvv = _make_ao_ints(mycc.mol, mo_a+mo_b, nocc, dtype)
        orbspin = self.orbspin
        occspin = orbspin[:nocc]
        virspin = orbspin[nocc:]

        oosym_forbid = (occspin[:,None] != occspin).ravel()
        ovsym_forbid = (occspin[:,None] != virspin).ravel()
        vosym_forbid = (virspin[:,None] != occspin).ravel()
        vvsym_forbid = (virspin[:,None] != virspin).ravel()

        cput1 = logger.timer(mycc, 'making ao integrals', *cput1)
        mo = ctf.astensor(mo_a+mo_b)
        orbo, orbv = mo[:,:nocc], mo[:,nocc:]

        tmp = ctf.einsum('uvmn,ui->ivmn', ppoo, orbo)
        oooo = ctf.einsum('ivmn,vj->ijmn', tmp, orbo)
        _force_sym(oooo, oosym_forbid, oosym_forbid)
        oooo = oooo.transpose(0,2,1,3) - oooo.transpose(0,2,3,1)

        ooov = ctf.einsum('ivmn,va->mnia', tmp, orbv)
        _force_sym(ooov, oosym_forbid, ovsym_forbid)
        ooov = ooov.transpose(0,2,1,3) - ooov.transpose(2,0,1,3)

        tmp = ctf.einsum('uvma,vb->ubma', ppov, orbv)
        ovov = ctf.einsum('ubma,ui->ibma', tmp, orbo)
        _force_sym(ovov, ovsym_forbid, ovsym_forbid)
        oovv = ovov.transpose(0,2,1,3) - ovov.transpose(0,2,3,1)
        del ppoo, ovov, tmp

        tmp = ctf.einsum('uvma,ub->mabv', ppov, orbv)
        _ovvo = ctf.einsum('mabv,vi->mabi', tmp, orbo)
        _force_sym(_ovvo, ovsym_forbid, vosym_forbid)
        tmp = ctf.einsum('uvab,ui->ivab', ppvv, orbo)
        _oovv = ctf.einsum('ivab,vj->ijab', tmp, orbo)
        _force_sym(_oovv, oosym_forbid, vvsym_forbid)

        ovov = _oovv.transpose(0,2,1,3) - _ovvo.transpose(0,2,3,1)
        ovvo = _ovvo.transpose(0,2,1,3) - _oovv.transpose(0,2,3,1)
        del _ovvo, _oovv, ppov, tmp

        tmp = ctf.einsum('uvab,vc->ucab', ppvv, orbv)
        ovvv = ctf.einsum('ucab,ui->icab', tmp, orbo)
        _force_sym(ovvv, ovsym_forbid, vvsym_forbid)
        ovvv = ovvv.transpose(0,2,1,3) - ovvv.transpose(0,2,3,1)

        vvvv = ctf.einsum('ucab,ud->dcab', tmp, orbv)
        _force_sym(vvvv, vvsym_forbid, vvsym_forbid)
        vvvv = vvvv.transpose(0,2,1,3) - vvvv.transpose(0,2,3,1)
        del ppvv, tmp

        self.oooo = tensor(oooo, verbose=mycc.SYMVERBOSE)
        self.ooov = tensor(ooov, verbose=mycc.SYMVERBOSE)
        self.oovv = tensor(oovv, verbose=mycc.SYMVERBOSE)
        self.ovov = tensor(ovov, verbose=mycc.SYMVERBOSE)
        self.ovvo = tensor(ovvo, verbose=mycc.SYMVERBOSE)
        self.ovvv = tensor(ovvv, verbose=mycc.SYMVERBOSE)
        self.vvvv = tensor(vvvv, verbose=mycc.SYMVERBOSE)
        logger.timer(mycc, 'ao2mo transformation', *cput0)

if __name__ == '__main__':

    from pyscf import gto, scf
    mol = gto.Mole()
    mol.atom = [['O', (0.,   0., 0.)],
                ['O', (1.21, 0., 0.)]]
    mol.basis = 'cc-pvdz'
    mol.spin = 2
    mol.verbose=4
    mol.build()

    mf = scf.UHF(mol)
    if rank==0:
        mf.kernel()
        mf = scf.addons.convert_to_ghf(mf)

    # Freeze 1s electrons
    frozen = [0,1,2,3]

    gcc = GCCSD(mf, frozen=frozen)
    ecc0, t1, t2 = gcc.kernel()
    print("Error", abs(ecc0--0.34869874698026165))
