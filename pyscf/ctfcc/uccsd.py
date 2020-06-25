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
UCCSD with CTF
'''

import numpy
import ctf
import time
from functools import reduce
from pyscf.lib import logger
from pyscf.mp import ump2
from pyscf.ctfcc import mpi_helper, rccsd
import pyscf.ctfcc.uintermediates as imd
from pyscf.ctfcc.integrals.ao2mo import _make_ao_ints
from symtensor.sym_ctf import tensor, zeros, einsum

rank = mpi_helper.rank
size = mpi_helper.size
comm = mpi_helper.comm

def energy(mycc, t1, t2, eris):
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    e = einsum('ia,ia->', eris.fov, t1a)
    e+= einsum('ia,ia->', eris.fOV, t1b)

    tauaa = t2aa + 2*einsum('ia,jb->ijab', t1a, t1a)
    tauab = t2ab +   einsum('ia,jb->ijab', t1a, t1b)
    taubb = t2bb + 2*einsum('ia,jb->ijab', t1b, t1b)

    e += 0.25*(einsum('iajb,ijab->',eris.ovov,tauaa)
             - einsum('jaib,ijab->',eris.ovov,tauaa))

    e += einsum('iajb,ijab->',eris.ovOV,tauab)
    e += 0.25*(einsum('iajb,ijab->',eris.OVOV,taubb)
             - einsum('jaib,ijab->',eris.OVOV,taubb))
    return e.real

def init_amps(mycc, eris):
    t1a = eris.fov.conj() / eris.eia
    t1b = eris.fOV.conj() / eris.eIA
    t2aa = eris.ovov.conj().transpose(0,2,1,3) / eris.eijab
    t2aa-= t2aa.transpose(0,1,3,2)
    t2ab = eris.ovOV.conj().transpose(0,2,1,3) / eris.eiJaB
    t2bb = eris.OVOV.conj().transpose(0,2,1,3) / eris.eIJAB
    t2bb-= t2bb.transpose(0,1,3,2)

    e  =      einsum('ijab,iajb->', t2ab, eris.ovOV)
    e += 0.25*einsum('ijab,iajb->', t2aa, eris.ovov)
    e -= 0.25*einsum('ijab,ibja->', t2aa, eris.ovov)
    e += 0.25*einsum('ijab,iajb->', t2bb, eris.OVOV)
    e -= 0.25*einsum('ijab,ibja->', t2bb, eris.OVOV)

    t1 = (t1a, t1b)
    t2 = (t2aa, t2ab, t2bb)
    return e, t1, t2

def update_amps(mycc, t1, t2, eris):
    time0 = time.clock(), time.time()
    log = logger.Logger(mycc.stdout, mycc.verbose)

    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2

    Fvv, FVV = imd.cc_Fvv(t1, t2, eris)
    Foo, FOO = imd.cc_Foo(t1, t2, eris)
    Fov, FOV = imd.cc_Fov(t1, t2, eris)

    # Move energy terms to the other side
    Fvv -= eris._fvv
    FVV -= eris._fVV
    Foo -= eris._foo
    FOO -= eris._fOO

    # T1 equation
    Ht1a = eris.fov.conj()
    Ht1b = eris.fOV.conj()

    Ht1a += einsum('imae,me->ia', t2aa, Fov)
    Ht1a += einsum('imae,me->ia', t2ab, FOV)
    Ht1b += einsum('imae,me->ia', t2bb, FOV)
    Ht1b += einsum('miea,me->ia', t2ab, Fov)

    Ht1a += einsum('ie,ae->ia', t1a, Fvv)
    Ht1b += einsum('ie,ae->ia', t1b, FVV)
    Ht1a -= einsum('ma,mi->ia', t1a, Foo)
    Ht1b -= einsum('ma,mi->ia', t1b, FOO)

    ovoo = eris.ooov.transpose(2,3,0,1) - eris.ooov.transpose(0,3,2,1)
    Ht1a += 0.5*einsum('mnae,meni->ia', t2aa, ovoo)
    OVOO = eris.OOOV.transpose(2,3,0,1) - eris.OOOV.transpose(0,3,2,1)
    Ht1b += 0.5*einsum('mnae,meni->ia', t2bb, OVOO)


    Ht1a -= einsum('nmae,nime->ia', t2ab, eris.ooOV)
    Ht1b -= einsum('mnea,nime->ia', t2ab, eris.OOov)

    Ht1a += einsum('mf,aimf->ia', t1a, eris.voov)
    Ht1a -= einsum('mf,miaf->ia', t1a, eris.oovv)
    Ht1a += einsum('mf,aimf->ia', t1b, eris.voOV)

    Ht1b += einsum('mf,aimf->ia', t1b, eris.VOOV)
    Ht1b -= einsum('mf,miaf->ia', t1b, eris.OOVV)
    Ht1b += einsum('mf,fmia->ia', t1a, eris.voOV.conj())

    Ht1a += einsum('imef,fmea->ia', t2aa, eris.vovv.conj())
    Ht1a += einsum('imef,fmea->ia', t2ab, eris.VOvv.conj())
    Ht1b += einsum('imef,fmea->ia', t2bb, eris.VOVV.conj())
    Ht1b += einsum('mife,fmea->ia', t2ab, eris.voVV.conj())

    Ftmpa = Fvv - 0.5 * einsum('mb,me->be', t1a, Fov)
    Ftmpb = FVV - 0.5 * einsum('mb,me->be', t1b, FOV)

    # T2 equation
    Ht2aa = einsum('ijae,be->ijab', t2aa, Ftmpa)
    Ht2bb = einsum('ijae,be->ijab', t2bb, Ftmpb)
    Ht2ab = einsum('ijae,be->ijab', t2ab, Ftmpb)
    Ht2ab += einsum('ijeb,ae->ijab', t2ab, Ftmpa)

    #P(ab)
    Ht2aa -= einsum('ijbe,ae->ijab', t2aa, Ftmpa)
    Ht2bb -= einsum('ijbe,ae->ijab', t2bb, Ftmpb)

    # Foo equation
    Ftmpa = Foo + 0.5 * einsum('je,me->mj', t1a, Fov)
    Ftmpb = FOO + 0.5 * einsum('je,me->mj', t1b, FOV)

    Ht2aa -= einsum('imab,mj->ijab', t2aa, Ftmpa)
    Ht2bb -= einsum('imab,mj->ijab', t2bb, Ftmpb)
    Ht2ab -= einsum('imab,mj->ijab', t2ab, Ftmpb)
    Ht2ab -= einsum('mjab,mi->ijab', t2ab, Ftmpa)

    #P(ij)
    Ht2aa += einsum('jmab,mi->ijab', t2aa, Ftmpa)
    Ht2bb += einsum('jmab,mi->ijab', t2bb, Ftmpb)

    Ht2aa += (eris.ovov.transpose(0,2,1,3) - eris.ovov.transpose(2,0,1,3)).conj()
    Ht2bb += (eris.OVOV.transpose(0,2,1,3) - eris.OVOV.transpose(2,0,1,3)).conj()
    Ht2ab += eris.ovOV.transpose(0,2,1,3).conj()

    tauaa, tauab, taubb = imd.make_tau(t2, t1, t1)
    Woooo, WooOO, WOOOO = imd.cc_Woooo(t1, t2, eris)

    Woooo += .5 * einsum('menf,ijef->minj', eris.ovov, tauaa)
    WOOOO += .5 * einsum('menf,ijef->minj', eris.OVOV, taubb)
    WooOO += .5 * einsum('menf,ijef->minj', eris.ovOV, tauab)

    Ht2aa += einsum('minj,mnab->ijab', Woooo, tauaa) * .5
    Ht2bb += einsum('minj,mnab->ijab', WOOOO, taubb) * .5
    Ht2ab += einsum('minj,mnab->ijab', WooOO, tauab)

    # add_vvvv block
    Wvvvv, WvvVV, WVVVV = imd.cc_Wvvvv_half(t1, t2, eris)
    tmp = einsum('acbd,ijcd->ijab', Wvvvv, tauaa) * .5
    Ht2aa += tmp
    Ht2aa -= tmp.transpose(0,1,3,2)

    tmp = einsum('acbd,ijcd->ijab', WVVVV, taubb) * .5
    Ht2bb += tmp
    Ht2bb -= tmp.transpose(0,1,3,2)
    Ht2ab += einsum('acbd,ijcd->ijab', WvvVV, tauab)
    del Wvvvv, WvvVV, WVVVV, tmp, tauaa, tauab, taubb

    Wovvo, WovVO, WOVvo, WOVVO, WoVVo, WOvvO = \
        imd.cc_Wovvo(t1, t2, eris)

    Ht2ab += einsum('imae,mebj->ijab', t2aa, WovVO)
    Ht2ab += einsum('imae,mebj->ijab', t2ab, WOVVO)
    Ht2ab -= einsum('ie,ma,emjb->ijab', t1a, t1a, eris.voOV.conj())

    Ht2ab += einsum('miea,mebj->jiba', t2ab, Wovvo)
    Ht2ab += einsum('miea,mebj->jiba', t2bb, WOVvo)

    Ht2ab -= einsum('ie,ma,bjme->jiba', t1b, t1b, eris.voOV)
    Ht2ab += einsum('imea,mebj->ijba', t2ab, WOvvO)
    Ht2ab -= einsum('ie,ma,mjbe->ijba', t1a, t1b, eris.OOvv)
    Ht2ab += einsum('miae,mebj->jiab', t2ab, WoVVo)
    Ht2ab -= einsum('ie,ma,mjbe->jiab', t1b, t1a, eris.ooVV)

    u2aa = einsum('imae,mebj->ijab', t2aa, Wovvo)
    u2aa += einsum('imae,mebj->ijab', t2ab, WOVvo)
    u2aa += einsum('ie,ma,mjbe->ijab',t1a,t1a,eris.oovv)
    u2aa -= einsum('ie,ma,bjme->ijab',t1a,t1a,eris.voov)

    u2aa += einsum('ie,bjae->ijab', t1a, eris.vovv)
    u2aa -= einsum('ma,imjb->ijab', t1a, eris.ooov.conj())

    u2aa = u2aa - u2aa.transpose(1,0,2,3)
    u2aa = u2aa - u2aa.transpose(0,1,3,2)
    Ht2aa += u2aa
    del u2aa, WOvvO, WoVVo, Wovvo, WOVvo

    u2bb = einsum('imae,mebj->ijab', t2bb, WOVVO)
    u2bb += einsum('miea,mebj->ijab', t2ab,WovVO)
    u2bb += einsum('ie,ma,mjbe->ijab',t1b, t1b, eris.OOVV)
    u2bb -= einsum('ie,ma,bjme->ijab',t1b, t1b, eris.VOOV)
    u2bb += einsum('ie,bjae->ijab', t1b, eris.VOVV)
    u2bb -= einsum('ma,imjb->ijab', t1b, eris.OOOV.conj())

    u2bb = u2bb - u2bb.transpose(1,0,2,3)
    u2bb = u2bb - u2bb.transpose(0,1,3,2)
    Ht2bb += u2bb
    del u2bb, WOVVO, WovVO

    Ht2ab += einsum('ie,bjae->ijab', t1a, eris.VOvv)
    Ht2ab += einsum('je,aibe->ijab', t1b, eris.voVV)
    Ht2ab -= einsum('ma,imjb->ijab', t1a, eris.ooOV.conj())
    Ht2ab -= einsum('mb,jmia->ijab', t1b, eris.OOov.conj())

    Ht1a /= eris.eia
    Ht1b /= eris.eIA

    Ht2aa /= eris.eijab
    Ht2ab /= eris.eiJaB
    Ht2bb /= eris.eIJAB

    time0 = log.timer_debug1('update t1 t2', *time0)
    return (Ht1a, Ht1b), (Ht2aa, Ht2ab, Ht2bb)


def amplitudes_to_vector(t1, t2):
    vector = ctf.hstack((t1[0].array.ravel(), t1[1].array.ravel(),\
                         t2[0].array.ravel(), t2[1].array.ravel(), t2[2].array.ravel()))
    return vector

def vector_to_amplitudes(vector, nmo, nocc):
    nocca, noccb = nocc
    nmoa, nmob = nmo
    nvira, nvirb = nmoa-nocca, nmob-noccb
    nova, novb = nocca*nvira, noccb*nvirb
    sizea = nova**2
    sizeab = nova * novb
    sizeb = novb**2
    t1a = tensor(vector[:nova].reshape(nocca,nvira))
    t1b = tensor(vector[nova:nova+novb].reshape(noccb,nvirb))
    t2aa = tensor(vector[nova+novb:nova+novb+sizea].reshape(nocca,nocca,nvira,nvira))
    t2ab = tensor(vector[nova+novb+sizea:nova+novb+sizea+sizeab].reshape(nocca,noccb,nvira,nvirb))
    t2bb = tensor(vector[nova+novb+sizea+sizeab:].reshape(noccb,noccb,nvirb,nvirb))
    return (t1a,t1b), (t2aa,t2ab,t2bb)

class UCCSD(rccsd.RCCSD):

    init_amps = init_amps
    energy = energy
    update_amps = update_amps

    get_nocc = ump2.get_nocc
    get_nmo = ump2.get_nmo
    get_frozen_mask = ump2.get_frozen_mask

    def amplitudes_to_vector(self, t1, t2):
        return amplitudes_to_vector(t1, t2)

    def vector_to_amplitudes(self, vector, nmo=None, nocc=None):
        if nocc is None: nocc = self.nocc
        if nmo is None: nmo = self.nmo
        return vector_to_amplitudes(vector, nmo, nocc)

    def ao2mo(self, mo_coeff=None):
        return _ChemistsERIs(self, mo_coeff)

    def ipccsd(self, nroots=1, koopmans=False, guess=None, left=False,
               eris=None, imds=None, **kwargs):
        from pyscf.ctfcc import eom_uccsd
        return eom_uccsd.EOMIP(self).kernel(nroots, koopmans, guess, left, eris,
                                            imds, **kwargs)

    def eaccsd(self, nroots=1, koopmans=False, guess=None, left=False,
               eris=None, imds=None, **kwargs):
        from pyscf.ctfcc import eom_uccsd
        return eom_uccsd.EOMEA(self).kernel(nroots, koopmans, guess, left, eris,
                                            imds, **kwargs)

class _ChemistsERIs:
    def __init__(self, mycc, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = mycc.mo_coeff
        nocca, noccb = mycc.nocc
        mo_idx = mycc.get_frozen_mask()
        self.mo_coeff = mo_coeff = \
                (mo_coeff[0][:,mo_idx[0]], mo_coeff[1][:,mo_idx[1]])
        if rank==0:
            dm = mycc._scf.make_rdm1(mycc.mo_coeff, mycc.mo_occ)
            vhf = mycc._scf.get_veff(mycc.mol, dm)
            fockao = mycc._scf.get_fock(vhf=vhf, dm=dm)
            focka = reduce(numpy.dot, (mo_coeff[0].conj().T, fockao[0], mo_coeff[0]))
            fockb = reduce(numpy.dot, (mo_coeff[1].conj().T, fockao[1], mo_coeff[1]))
            e_hf = mycc._scf.energy_tot(dm=dm, vhf=vhf)
        else:
            focka = fockb = e_hf = None
        focka = comm.bcast(focka, root=0)
        fockb = comm.bcast(fockb, root=0)
        nmoa, nmob = focka.shape[0], fockb.shape[0]
        nvira, nvirb = nmoa - nocca, nmob - noccb
        self.e_hf = comm.bcast(e_hf, root=0)
        mo_ea = focka.diagonal().real
        mo_eb = fockb.diagonal().real
        eia = mo_ea[:nocca][:,None] - mo_ea[nocca:][None,:]
        eIA = mo_eb[:noccb][:,None] - mo_eb[noccb:][None,:]
        self.eia = ctf.astensor(eia)
        self.eIA = ctf.astensor(eIA)
        focka = ctf.astensor(focka)
        fockb = ctf.astensor(fockb)

        self.foo = tensor(focka[:nocca,:nocca])
        self.fov = tensor(focka[:nocca,nocca:])
        self.fvv = tensor(focka[nocca:,nocca:])

        self.fOO = tensor(fockb[:noccb,:noccb])
        self.fOV = tensor(fockb[:noccb,noccb:])
        self.fVV = tensor(fockb[noccb:,noccb:])

        self._foo = self.foo.diagonal(preserve_shape=True)
        self._fOO = self.fOO.diagonal(preserve_shape=True)
        self._fvv = self.fvv.diagonal(preserve_shape=True)
        self._fVV = self.fVV.diagonal(preserve_shape=True)

        self.eijab = self.eia.reshape(nocca,1,nvira,1) + self.eia.reshape(1,nocca,1,nvira)
        self.eiJaB = self.eia.reshape(nocca,1,nvira,1) + self.eIA.reshape(1,noccb,1,nvirb)
        self.eIJAB = self.eIA.reshape(noccb,1,nvirb,1) + self.eIA.reshape(1,noccb,1,nvirb)

        mol = self.mol = mycc.mol

        gap_a = abs(eia)
        gap_b = abs(eIA)
        if gap_a.size > 0:
            gap_a = gap_a.min()
        else:
            gap_a = 1e9
        if gap_b.size > 0:
            gap_b = gap_b.min()
        else:
            gap_b = 1e9
        if gap_a < 1e-5 or gap_b < 1e-5:
            logger.warn(mycc, 'HOMO-LUMO gap (%s,%s) too small for UCCSD',
                        gap_a, gap_b)
        cput1 = (time.clock(), time.time())
        dtype = numpy.result_type(*mo_coeff)
        ppoo, ppov, ppvv = _make_ao_ints(mol, mo_coeff[0], nocca, dtype)
        cput1 = logger.timer(mycc, 'making ao integrals for alpha', *cput1)
        ppOO, ppOV, ppVV = _make_ao_ints(mol, mo_coeff[1], noccb, dtype)
        cput1 = logger.timer(mycc, 'making ao integrals for beta', *cput1)

        moa = ctf.astensor(mo_coeff[0])
        orba_o, orba_v = moa[:,:nocca], moa[:,nocca:]
        mob = ctf.astensor(mo_coeff[1])
        orbb_o, orbb_v = mob[:,:noccb], mob[:,noccb:]

        tmp = ctf.einsum('uvmn,ui->ivmn', ppoo, orba_o.conj())
        oooo = ctf.einsum('ivmn,vj->ijmn', tmp, orba_o)
        ooov = ctf.einsum('ivmn,va->mnia', tmp, orba_v)

        tmp = ctf.einsum('uvma,vb->ubma', ppov, orba_v)
        ovov = ctf.einsum('ubma,ui->ibma', tmp, orba_o.conj())
        tmp = ctf.einsum('uvma,ub->mabv', ppov, orba_v.conj())
        voov = ctf.einsum('mabv,vi->bima', tmp, orba_o)

        tmp = ctf.einsum('uvab,ui->ivab', ppvv, orba_o.conj())
        oovv = ctf.einsum('ivab,vj->ijab', tmp, orba_o)

        tmp = ctf.einsum('uvab,vc->ucab', ppvv, orba_v)
        vovv = ctf.einsum('ucab,ui->ciba', tmp.conj(), orba_o)
        vvvv = ctf.einsum('ucab,ud->dcab', tmp, orba_v.conj())

        self.oooo = tensor(oooo)
        self.ooov = tensor(ooov)
        self.oovv = tensor(oovv)
        self.ovov = tensor(ovov)
        self.voov = tensor(voov)
        self.vovv = tensor(vovv)
        self.vvvv = tensor(vvvv)

        del ppoo, ppov, ppvv

        tmp = ctf.einsum('uvmn,ui->ivmn', ppOO, orbb_o.conj())
        OOOO = ctf.einsum('ivmn,vj->ijmn', tmp, orbb_o)
        OOOV = ctf.einsum('ivmn,va->mnia', tmp, orbb_v)

        tmp = ctf.einsum('uvma,vb->ubma', ppOV, orbb_v)
        OVOV = ctf.einsum('ubma,ui->ibma', tmp, orbb_o.conj())
        tmp = ctf.einsum('uvma,ub->mabv', ppOV, orbb_v.conj())
        VOOV = ctf.einsum('mabv,vi->bima', tmp, orbb_o)

        tmp = ctf.einsum('uvab,ui->ivab', ppVV, orbb_o.conj())
        OOVV = ctf.einsum('ivab,vj->ijab', tmp, orbb_o)

        tmp = ctf.einsum('uvab,vc->ucab', ppVV, orbb_v)
        VOVV = ctf.einsum('ucab,ui->ciba', tmp.conj(), orbb_o)
        VVVV = ctf.einsum('ucab,ud->dcab', tmp, orbb_v.conj())

        self.OOOO = tensor(OOOO)
        self.OOOV = tensor(OOOV)
        self.OOVV = tensor(OOVV)
        self.OVOV = tensor(OVOV)
        self.VOOV = tensor(VOOV)
        self.VOVV = tensor(VOVV)
        self.VVVV = tensor(VVVV)

        ooOO = ctf.einsum('uvmn,ui,vj->ijmn', ppOO, orba_o.conj(), orba_o)
        ooOV = ctf.einsum('uvma,ui,vj->ijma', ppOV, orba_o.conj(), orba_o)
        ovOV = ctf.einsum('uvma,ui,vb->ibma', ppOV, orba_o.conj(), orba_v)
        voOV = ctf.einsum('uvma,ub,vi->bima', ppOV, orba_v.conj(), orba_o)
        ooVV = ctf.einsum('uvab,ui,vj->ijab', ppVV, orba_o.conj(), orba_o)
        voVV = ctf.einsum('uvab,uc,vi->ciab', ppVV, orba_v.conj(), orba_o)
        vvVV = ctf.einsum('uvab,uc,vd->cdab', ppVV, orba_v.conj(), orba_v)

        OOov = ctf.einsum('uvmn,ui,va->mnia', ppOO, orba_o.conj(), orba_v)
        OOvv = ctf.einsum('uvmn,ua,vb->mnab', ppOO, orba_v.conj(), orba_v)
        OVov = ovOV.transpose(2,3,0,1)
        VOov = voOV.transpose(3,2,1,0).conj()
        VOvv = ctf.einsum('uvma,ub,vc->amcb', ppOV, orba_v.conj(), orba_v).conj()
        del ppOO, ppOV, ppVV

        self.ooOO = tensor(ooOO)
        self.ooOV = tensor(ooOV)
        self.ooVV = tensor(ooVV)
        self.ovOV = tensor(ovOV)
        self.voOV = tensor(voOV)
        self.voVV = tensor(voVV)
        self.vvVV = tensor(vvVV)

        self.OOoo = None
        self.OOov = tensor(OOov)
        self.OOvv = tensor(OOvv)
        self.OVov = tensor(OVov)
        self.VOov = tensor(VOov)
        self.VOvv = tensor(VOvv)


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
        mf.run()
    frozen = [[0,1], [0,1]]
    ucc = UCCSD(mf, frozen=frozen)
    ecc, t1, t2 = ucc.kernel()
    print(ecc - -0.3486987472235819)
