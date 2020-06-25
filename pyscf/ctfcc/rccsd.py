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
Mole RCCSD with CTF
'''

import numpy
import ctf
import time
from functools import reduce
from pyscf.cc import ccsd
from pyscf.lib import logger
from pyscf.ctfcc.mpi_helper import rank, size, comm
from pyscf.ctfcc.ccsd import CCSD
import pyscf.ctfcc.rintermediates as imd
from pyscf.ctfcc.integrals.ao2mo import _make_ao_ints
from symtensor.sym_ctf import tensor, einsum

# This is restricted (R)CCSD
# Ref: Hirata, et al., J. Chem. Phys. 120, 2581 (2004)

def energy(mycc, t1, t2, eris):
    e = 2*einsum('ia,ia', eris.fov, t1)
    tau = einsum('ia,jb->ijab', t1, t1)
    tau += t2
    e += 2*einsum('ijab,iajb', tau, eris.ovov)
    e +=  -einsum('ijab,ibja', tau, eris.ovov)
    return e.real

def init_amps(mycc, eris):
    t1 = eris.fov.conj() / eris.eia
    t2 = eris.ovov.transpose(0,2,1,3).conj() / eris.eijab
    mycc.emp2  = 2*einsum('ijab,iajb', t2, eris.ovov)
    mycc.emp2 -=   einsum('ijab,ibja', t2, eris.ovov)
    logger.info(mycc, 'Init t2, MP2 energy = %.15g', mycc.emp2.real)
    return mycc.emp2, t1, t2

def update_amps(mycc, t1, t2, eris):
    # Ref: Hirata et al., J. Chem. Phys. 120, 2581 (2004) Eqs.(35)-(36)

    fov = eris.fov.copy()
    foo = eris.foo.copy()
    fvv = eris.fvv.copy()
    Foo = imd.cc_Foo(t1,t2,eris)
    Fvv = imd.cc_Fvv(t1,t2,eris)
    Fov = imd.cc_Fov(t1,t2,eris)

    # Move energy terms to the other side
    Foo -= eris._foo
    Fvv -= eris._fvv

    # T1 equation
    t1new = fov.conj().copy()
    tmp    =   einsum('kc,ka->ac', fov, t1)
    t1new +=-2*einsum('ac,ic->ia', tmp, t1)
    t1new +=   einsum('ac,ic->ia', Fvv, t1)
    t1new +=  -einsum('ki,ka->ia', Foo, t1)
    t1new += 2*einsum('kc,kica->ia', Fov, t2)
    t1new +=  -einsum('kc,ikca->ia', Fov, t2)
    tmp    =   einsum('kc,ic->ki', Fov, t1)
    t1new +=   einsum('ki,ka->ia', tmp, t1)
    t1new += 2*einsum('kcai,kc->ia', eris.ovvo, t1)
    t1new +=  -einsum('kiac,kc->ia', eris.oovv, t1)
    t1new += 2*einsum('kdac,ikcd->ia', eris.ovvv, t2)
    t1new +=  -einsum('kcad,ikcd->ia', eris.ovvv, t2)
    tmp    =   einsum('kdac,kd->ac', eris.ovvv, t1)
    t1new += 2*einsum('ac,ic->ia', tmp, t1)
    tmp    =   einsum('kcad,kd->ca', eris.ovvv, t1)
    t1new +=  -einsum('ca,ic->ia', tmp, t1)
    t1new +=-2*einsum('kilc,klac->ia', eris.ooov, t2)
    t1new +=   einsum('likc,klac->ia', eris.ooov, t2)
    tmp    =   einsum('kilc,lc->ki', eris.ooov, t1)
    t1new +=-2*einsum('ki,ka->ia', tmp, t1)
    tmp    =   einsum('likc,lc->ik', eris.ooov, t1)
    t1new +=   einsum('ik,ka->ia', tmp, t1)

    # T2 equation
    t2new = eris.ovov.conj().transpose(0,2,1,3).copy()

    Loo = imd.Loo(t1, t2, eris)
    Lvv = imd.Lvv(t1, t2, eris)
    Loo -= eris._foo
    Lvv -= eris._fvv
    Woooo = imd.cc_Woooo(t1, t2, eris)
    Wvoov = imd.cc_Wvoov(t1, t2, eris)
    Wvovo = imd.cc_Wvovo(t1, t2, eris)
    Wvvvv = imd.cc_Wvvvv(t1, t2, eris)
    tau = t2 + einsum('ia,jb->ijab', t1, t1)
    t2new += einsum('klij,klab->ijab', Woooo, tau)
    t2new += einsum('abcd,ijcd->ijab', Wvvvv, tau)
    tmp = einsum('ac,ijcb->ijab', Lvv, t2)
    t2new += (tmp + tmp.transpose(1,0,3,2))
    tmp = einsum('ki,kjab->ijab', Loo, t2)
    t2new -= (tmp + tmp.transpose(1,0,3,2))
    tmp  = 2*einsum('akic,kjcb->ijab', Wvoov, t2)
    tmp -=   einsum('akci,kjcb->ijab', Wvovo, t2)
    t2new += (tmp + tmp.transpose(1,0,3,2))
    tmp = einsum('akic,kjbc->ijab', Wvoov, t2)
    t2new -= (tmp + tmp.transpose(1,0,3,2))
    tmp = einsum('bkci,kjac->ijab', Wvovo, t2)
    t2new -= (tmp + tmp.transpose(1,0,3,2))

    tmp2  = einsum('kibc,ka->abic', eris.oovv, -t1)
    tmp2 += eris.ovvv.conj().transpose(1,3,0,2)
    tmp = einsum('abic,jc->ijab', tmp2, t1)
    t2new += (tmp + tmp.transpose(1,0,3,2))
    tmp2  = einsum('kcai,jc->akij', eris.ovvo, t1)
    tmp2 += eris.ooov.transpose(3,1,2,0).conj()
    tmp = einsum('akij,kb->ijab', tmp2, t1)
    t2new -= (tmp + tmp.transpose(1,0,3,2))

    t1new /= eris.eia
    t2new /= eris.eijab

    return t1new, t2new


class RCCSD(CCSD):

    init_amps = init_amps
    energy = energy
    update_amps = update_amps

    def ao2mo(self, mo_coeff=None):
        return _ChemistsERIs(self, mo_coeff)

    def ipccsd(self, nroots=1, koopmans=False, guess=None, left=False,
               eris=None, imds=None, **kwargs):
        from pyscf.ctfcc import eom_rccsd
        return eom_rccsd.EOMIP(self).kernel(nroots, koopmans, guess, left, eris,
                                            imds, **kwargs)

    def eaccsd(self, nroots=1, koopmans=False, guess=None, left=False,
               eris=None, imds=None, **kwargs):
        from pyscf.ctfcc import eom_rccsd
        return eom_rccsd.EOMEA(self).kernel(nroots, koopmans, guess, left, eris,
                                            imds, **kwargs)

    def eeccsd(self, nroots=1, koopmans=False, guess=None, eris=None):
        raise NotImplementedError()

    def eomee_ccsd_singlet(self, nroots=1, koopmans=False, guess=None, eris=None):
        raise NotImplementedError()

    def eomee_ccsd_triplet(self, nroots=1, koopmans=False, guess=None, eris=None):
        raise NotImplementedError()

    def eomsf_ccsd(self, nroots=1, koopmans=False, guess=None, eris=None):
        raise NotImplementedError()

class _ChemistsERIs:
    def __init__(self, mycc, mo_coeff=None):
        nocc, nmo = mycc.nocc, mycc.nmo
        nvir = nmo - nocc
        cput0 = (time.clock(), time.time())

        if rank==0:
            if mo_coeff is None:
                self.mo_coeff = mo_coeff = ccsd._mo_without_core(mycc, mycc.mo_coeff)
            else:
                self.mo_coeff = mo_coeff = ccsd._mo_without_core(mycc, mo_coeff)
            dm = mycc._scf.make_rdm1(mycc.mo_coeff, mycc.mo_occ)
            fockao = mycc._scf.get_hcore() + mycc._scf.get_veff(mycc.mol, dm)
            fock = reduce(numpy.dot, (mo_coeff.T.conj(), fockao, mo_coeff))
        else:
            fock = self.mo_coeff = None

        comm.barrier()
        fock = comm.bcast(fock, root=0)
        mo_coeff = self.mo_coeff = comm.bcast(self.mo_coeff, root=0)
        self.dtype = dtype = numpy.result_type(fock)

        mo_e = fock.diagonal().real
        eia  = mo_e[:nocc,None]- mo_e[None,nocc:]

        fock = ctf.astensor(fock)
        self.foo = tensor(fock[:nocc,:nocc], verbose=mycc.SYMVERBOSE)
        self.fov = tensor(fock[:nocc,nocc:], verbose=mycc.SYMVERBOSE)
        self.fvv = tensor(fock[nocc:,nocc:], verbose=mycc.SYMVERBOSE)

        self.eia = ctf.astensor(eia)
        self._foo = self.foo.diagonal(preserve_shape=True)
        self._fvv = self.fvv.diagonal(preserve_shape=True)
        self.eijab = self.eia.reshape(nocc,1,nvir,1) + self.eia.reshape(1,nocc,1,nvir)

        cput1 = (time.clock(), time.time())
        ppoo, ppov, ppvv = _make_ao_ints(mycc.mol, mo_coeff, nocc, dtype)
        cput1 = logger.timer(mycc, 'making ao integrals', *cput1)
        mo = ctf.astensor(mo_coeff)
        orbo, orbv = mo[:,:nocc], mo[:,nocc:]

        tmp = ctf.einsum('uvmn,ui->ivmn', ppoo, orbo.conj())
        oooo = ctf.einsum('ivmn,vj->ijmn', tmp, orbo)
        ooov = ctf.einsum('ivmn,va->mnia', tmp, orbv)

        tmp = ctf.einsum('uvma,vb->ubma', ppov, orbv)
        ovov = ctf.einsum('ubma,ui->ibma', tmp, orbo.conj())
        tmp = ctf.einsum('uvma,ub->mabv', ppov, orbv.conj())
        ovvo = ctf.einsum('mabv,vi->mabi', tmp, orbo)

        tmp = ctf.einsum('uvab,ui->ivab', ppvv, orbo.conj())
        oovv = ctf.einsum('ivab,vj->ijab', tmp, orbo)

        tmp = ctf.einsum('uvab,vc->ucab', ppvv, orbv)
        ovvv = ctf.einsum('ucab,ui->icab', tmp, orbo.conj())
        vvvv = ctf.einsum('ucab,ud->dcab', tmp, orbv.conj())

        self.oooo = tensor(oooo, verbose=mycc.SYMVERBOSE)
        self.ooov = tensor(ooov, verbose=mycc.SYMVERBOSE)
        self.ovov = tensor(ovov, verbose=mycc.SYMVERBOSE)
        self.oovv = tensor(oovv, verbose=mycc.SYMVERBOSE)
        self.ovvo = tensor(ovvo, verbose=mycc.SYMVERBOSE)
        self.ovvv = tensor(ovvv, verbose=mycc.SYMVERBOSE)
        self.vvvv = tensor(vvvv, verbose=mycc.SYMVERBOSE)
        logger.timer(mycc, 'ao2mo transformation', *cput0)


if __name__ == '__main__':
    from pyscf import scf, gto

    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]
    mol.basis = 'cc-pvdz'
    mol.verbose = 4
    mol.spin = 0
    mol.build()

    mf = scf.RHF(mol)
    if rank==0:
        mf.kernel()

    mycc = RCCSD(mf)
    ecc = mycc.kernel()[0]
    print(ecc - -0.2133432465136908)
