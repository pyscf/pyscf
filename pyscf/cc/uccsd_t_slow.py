#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import numpy
from pyscf import lib
from pyscf import ao2mo
from pyscf.cc import uccsd

'''
UCCSD(T)
'''

def kernel(eris, t1, t2):
    def p6(t):
        return (t + t.transpose(1,2,0,4,5,3) +
                t.transpose(2,0,1,5,3,4) + t.transpose(0,2,1,3,5,4) +
                t.transpose(2,1,0,5,4,3) + t.transpose(1,0,2,4,3,5))
    def r6(w):
        return (w + w.transpose(2,0,1,3,4,5) + w.transpose(1,2,0,3,4,5)
                - w.transpose(2,1,0,3,4,5) - w.transpose(0,2,1,3,4,5)
                - w.transpose(1,0,2,3,4,5))

    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nocca = eris.nocca
    noccb = eris.noccb
    nmoa = eris.focka.shape[0]
    nmob = eris.fockb.shape[0]
    mo_ea = eris.focka.diagonal()
    mo_eb = eris.fockb.diagonal()
    eia = mo_ea[:nocca,None] - mo_ea[nocca:]
    eIA = mo_eb[:noccb,None] - mo_eb[noccb:]

    # aaa
    d3 = lib.direct_sum('ia+jb+kc->ijkabc', eia, eia, eia)
    w = numpy.einsum('ijae,ckbe->ijkabc', t2aa, eris.vovp[:,:,:,nocca:])
    w-= numpy.einsum('imab,ckmj->ijkabc', t2aa, eris.vooo)
    r = r6(w)
    v = numpy.einsum('bjck,ia->ijkabc', eris.vovp[:,:,:,:nocca], t1a)
    wvd = p6(w + v) / d3
    et = numpy.einsum('ijkabc,ijkabc', wvd, r)

    # bbb
    d3 = lib.direct_sum('ia+jb+kc->ijkabc', eIA, eIA, eIA)
    w = numpy.einsum('ijae,ckbe->ijkabc', t2bb, eris.VOVP[:,:,:,noccb:])
    w-= numpy.einsum('imab,ckmj->ijkabc', t2bb, eris.VOOO)
    r = r6(w)
    v = numpy.einsum('bjck,ia->ijkabc', eris.VOVP[:,:,:,:noccb], t1b)
    wvd = p6(w + v) / d3
    et += numpy.einsum('ijkabc,ijkabc', wvd, r)

    # baa
    w  = numpy.einsum('jIeA,ckbe->IjkAbc', t2ab, eris.vovp[:,:,:,nocca:]) * 2
    w += numpy.einsum('jIbE,ckAE->IjkAbc', t2ab, eris.voVP[:,:,:,noccb:]) * 2
    w += numpy.einsum('jkbe,AIce->IjkAbc', t2aa, eris.VOvp[:,:,:,nocca:])
    w -= numpy.einsum('mIbA,ckmj->IjkAbc', t2ab, eris.vooo) * 2
    w -= numpy.einsum('jMbA,ckMI->IjkAbc', t2ab, eris.voOO) * 2
    w -= numpy.einsum('jmbc,AImk->IjkAbc', t2aa, eris.VOoo)
    r = w - w.transpose(0,2,1,3,4,5)
    v  = numpy.einsum('bjck,IA->IjkAbc', eris.vovp[:,:,:,:nocca], t1b)
    v += numpy.einsum('AIck,jb->IjkAbc', eris.VOvp[:,:,:,:nocca], t1a)
    v += numpy.einsum('ckAI,jb->IjkAbc', eris.voVP[:,:,:,:noccb], t1a)
    w += v
    w = w + w.transpose(0,2,1,3,5,4)
    d3 = lib.direct_sum('ia+jb+kc->ijkabc', eIA, eia, eia)
    w /= d3
    et += numpy.einsum('ijkabc,ijkabc', w, r)

    # bba
    w  = numpy.einsum('ijae,ckbe->ijkabc', t2ab, eris.VOVP[:,:,:,noccb:]) * 2
    w += numpy.einsum('ijeb,ckae->ijkabc', t2ab, eris.VOvp[:,:,:,nocca:]) * 2
    w += numpy.einsum('jkbe,aice->ijkabc', t2bb, eris.voVP[:,:,:,noccb:])
    w -= numpy.einsum('imab,ckmj->ijkabc', t2ab, eris.VOOO) * 2
    w -= numpy.einsum('mjab,ckmi->ijkabc', t2ab, eris.VOoo) * 2
    w -= numpy.einsum('jmbc,aimk->ijkabc', t2bb, eris.voOO)
    r = w - w.transpose(0,2,1,3,4,5)
    v  = numpy.einsum('bjck,ia->ijkabc', eris.VOVP[:,:,:,:noccb], t1a)
    v += numpy.einsum('aick,jb->ijkabc', eris.voVP[:,:,:,:noccb], t1b)
    v += numpy.einsum('ckai,jb->ijkabc', eris.VOvp[:,:,:,:nocca], t1b)
    w += v
    w = w + w.transpose(0,2,1,3,5,4)
    d3 = lib.direct_sum('ia+jb+kc->ijkabc', eia, eIA, eIA)
    w /= d3
    et += numpy.einsum('ijkabc,ijkabc', w, r)

    et *= .25
    return et


class _ERIS:
    def __init__(self, mycc, mo_coeff=None):
        moidx = uccsd.get_umoidx(mycc)
        if mo_coeff is None:
            mo_coeff = (mycc.mo_coeff[0][:,moidx[0]], mycc.mo_coeff[1][:,moidx[1]])
        else:
            mo_coeff = (mo_coeff[0][:,moidx[0]], mo_coeff[1][:,moidx[1]])
# Note: Always recompute the fock matrix in UCISD because the mf object may be
# converted from ROHF object in which orbital energies are eigenvalues of
# Roothaan Fock rather than the true alpha, beta orbital energies. 
        dm = mycc._scf.make_rdm1(mycc.mo_coeff, mycc.mo_occ)
        fockao = mycc._scf.get_hcore() + mycc._scf.get_veff(mycc.mol, dm)
        self.focka = reduce(numpy.dot, (mo_coeff[0].T, fockao[0], mo_coeff[0]))
        self.fockb = reduce(numpy.dot, (mo_coeff[1].T, fockao[1], mo_coeff[1]))
        self.mo_coeff = mo_coeff
        self.nocca, self.noccb = mycc.get_nocc()

        nocca = self.nocca
        noccb = self.noccb
        nmoa = self.focka.shape[0]
        nmob = self.fockb.shape[0]
        nvira = nmoa - nocca
        nvirb = nmob - noccb
        moa, mob = self.mo_coeff

        eri_aa = ao2mo.restore(1, ao2mo.full(mycc._scf._eri, moa), nmoa)
        self.vovp = eri_aa[nocca:,:nocca,nocca:].copy()
        self.vooo = eri_aa[nocca:,:nocca,:nocca,:nocca].copy()

        eri_bb = ao2mo.restore(1, ao2mo.full(mycc._scf._eri, mob), nmob)
        self.VOVP = eri_bb[noccb:,:noccb,noccb:].copy()
        self.VOOO = eri_bb[noccb:,:noccb,:noccb,:noccb].copy()

        eri_ab = ao2mo.general(mycc._scf._eri, (moa,moa,mob,mob), compact=False)
        eri_ab = eri_ab.reshape(nmoa,nmoa,nmob,nmob)
        self.voVP = eri_ab[nocca:,:nocca,noccb:].copy()
        self.voOO = eri_ab[nocca:,:nocca,:noccb,:noccb].copy()

        eri_ba = lib.transpose(eri_ab.reshape(nmoa**2,nmob**2))
        eri_ba = eri_ba.reshape(nmob,nmob,nmoa,nmoa)
        self.VOvp = eri_ba[noccb:,:noccb,nocca:].copy()
        self.VOoo = eri_ba[noccb:,:noccb,:nocca,:nocca].copy()


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf import cc

    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -.757 , .587)],
        [1 , (0. ,  .757 , .587)]]

    mol.basis = '631g'
    mol.build()
    rhf = scf.RHF(mol)
    rhf.conv_tol = 1e-14
    rhf.scf()
    mcc = cc.CCSD(rhf)
    mcc.conv_tol = 1e-14
    mcc.ccsd()
    t1a = t1b = mcc.t1
    t2ab = mcc.t2
    t2aa = t2bb = t2ab - t2ab.transpose(1,0,2,3)
    e3a = kernel(_ERIS(uccsd.UCCSD(scf.addons.convert_to_uhf(rhf))),
                 (t1a,t1b), (t2aa,t2ab,t2bb))
    print(e3a - -0.00099642337843278096)

    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -.757 , .587)],
        [1 , (0. ,  .757 , .587)]]
    mol.spin = 2
    mol.basis = '3-21g'
    mol.build()
    mf = scf.UHF(mol).run(conv_tol=1e-14)
    numpy.random.seed(10)
    mf.mo_coeff = numpy.random.random(mf.mo_coeff.shape)

    numpy.random.seed(12)
    nocca, noccb = mol.nelec
    nmo = mf.mo_occ[0].size
    nvira = nmo - nocca
    nvirb = nmo - noccb
    t1a  = .1 * numpy.random.random((nocca,nvira))
    t1b  = .1 * numpy.random.random((noccb,nvirb))
    t2aa = .1 * numpy.random.random((nocca,nocca,nvira,nvira))
    t2aa = t2aa - t2aa.transpose(0,1,3,2)
    t2aa = t2aa - t2aa.transpose(1,0,2,3)
    t2bb = .1 * numpy.random.random((noccb,noccb,nvirb,nvirb))
    t2bb = t2bb - t2bb.transpose(0,1,3,2)
    t2bb = t2bb - t2bb.transpose(1,0,2,3)
    t2ab = .1 * numpy.random.random((nocca,noccb,nvira,nvirb))
    t1 = t1a, t1b
    t2 = t2aa, t2ab, t2bb
    e3a = kernel(_ERIS(uccsd.UCCSD(scf.addons.convert_to_uhf(mf)), mf.mo_coeff),
                 [t1a,t1b], [t2aa, t2ab, t2bb])
    print(e3a - 8193.064821311109)

