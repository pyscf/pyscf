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

def kernel(mcc, eris, t1, t2):
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
    nocca, noccb = t2ab.shape[:2]
    nmoa = eris.focka.shape[0]
    nmob = eris.fockb.shape[0]
    mo_ea = eris.focka.diagonal().real
    mo_eb = eris.fockb.diagonal().real
    eia = mo_ea[:nocca,None] - mo_ea[nocca:]
    eIA = mo_eb[:noccb,None] - mo_eb[noccb:]
    fvo = eris.focka[nocca:,:nocca]
    fVO = eris.fockb[noccb:,:noccb]

    # aaa
    d3 = lib.direct_sum('ia+jb+kc->ijkabc', eia, eia, eia)
    w = numpy.einsum('ijae,ckbe->ijkabc', t2aa, eris.vovp[:,:,:,nocca:])
    w-= numpy.einsum('mkbc,aimj->ijkabc', t2aa, eris.vooo)
    r = r6(w)
    v = numpy.einsum('bjck,ia->ijkabc', eris.vovp[:,:,:,:nocca], t1a)
    v+= numpy.einsum('jkbc,ai->ijkabc', t2aa, fvo) * .5
    wvd = p6(w + v) / d3
    et = numpy.einsum('ijkabc,ijkabc', wvd.conj(), r)

    # bbb
    d3 = lib.direct_sum('ia+jb+kc->ijkabc', eIA, eIA, eIA)
    w = numpy.einsum('ijae,ckbe->ijkabc', t2bb, eris.VOVP[:,:,:,noccb:])
    w-= numpy.einsum('imab,ckmj->ijkabc', t2bb, eris.VOOO)
    r = r6(w)
    v = numpy.einsum('bjck,ia->ijkabc', eris.VOVP[:,:,:,:noccb], t1b)
    v+= numpy.einsum('jkbc,ai->ijkabc', t2bb, fVO) * .5
    wvd = p6(w + v) / d3
    et += numpy.einsum('ijkabc,ijkabc', wvd.conj(), r)

    # baa
    w  = numpy.einsum('jIeA,ckbe->IjkAbc', t2ab, eris.vovp[:,:,:,nocca:]) * 2
    w += numpy.einsum('jIbE,ckAE->IjkAbc', t2ab, eris.voVP[:,:,:,noccb:]) * 2
    w += numpy.einsum('jkbe,AIce->IjkAbc', t2aa, eris.VOvp[:,:,:,nocca:])
    w -= numpy.einsum('mIbA,ckmj->IjkAbc', t2ab, eris.vooo) * 2
    w -= numpy.einsum('jMbA,ckMI->IjkAbc', t2ab, eris.voOO) * 2
    w -= numpy.einsum('jmbc,AImk->IjkAbc', t2aa, eris.VOoo)
    r = w - w.transpose(0,2,1,3,4,5)
    r = r + r.transpose(0,2,1,3,5,4)
    v  = numpy.einsum('bjck,IA->IjkAbc', eris.vovp[:,:,:,:nocca], t1b)
    v += numpy.einsum('AIck,jb->IjkAbc', eris.VOvp[:,:,:,:nocca], t1a)
    v += numpy.einsum('ckAI,jb->IjkAbc', eris.voVP[:,:,:,:noccb], t1a)
    v += numpy.einsum('jkbc,AI->IjkAbc', t2aa, fVO) * .5
    v += numpy.einsum('kIcA,bj->IjkAbc', t2ab, fvo) * 2
    w += v
    d3 = lib.direct_sum('ia+jb+kc->ijkabc', eIA, eia, eia)
    r /= d3
    et += numpy.einsum('ijkabc,ijkabc', w.conj(), r)

    # bba
    w  = numpy.einsum('ijae,ckbe->ijkabc', t2ab, eris.VOVP[:,:,:,noccb:]) * 2
    w += numpy.einsum('ijeb,ckae->ijkabc', t2ab, eris.VOvp[:,:,:,nocca:]) * 2
    w += numpy.einsum('jkbe,aice->ijkabc', t2bb, eris.voVP[:,:,:,noccb:])
    w -= numpy.einsum('imab,ckmj->ijkabc', t2ab, eris.VOOO) * 2
    w -= numpy.einsum('mjab,ckmi->ijkabc', t2ab, eris.VOoo) * 2
    w -= numpy.einsum('jmbc,aimk->ijkabc', t2bb, eris.voOO)
    r = w - w.transpose(0,2,1,3,4,5)
    r = r + r.transpose(0,2,1,3,5,4)
    v  = numpy.einsum('bjck,ia->ijkabc', eris.VOVP[:,:,:,:noccb], t1a)
    v += numpy.einsum('aick,jb->ijkabc', eris.voVP[:,:,:,:noccb], t1b)
    v += numpy.einsum('ckai,jb->ijkabc', eris.VOvp[:,:,:,:nocca], t1b)
    v += numpy.einsum('JKBC,ai->iJKaBC', t2bb, fvo) * .5
    v += numpy.einsum('iKaC,BJ->iJKaBC', t2ab, fVO) * 2
    w += v
    d3 = lib.direct_sum('ia+jb+kc->ijkabc', eia, eIA, eIA)
    r /= d3
    et += numpy.einsum('ijkabc,ijkabc', w.conj(), r)

    et *= .25
    return et


class _ChemistsERIs:
    def __init__(self, mycc, mo_coeff=None):
        moidx = mycc.get_frozen_mask()
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
    mcc = uccsd.UCCSD(scf.addons.convert_to_uhf(rhf))
    e3a = kernel(mcc, _ChemistsERIs(mcc), (t1a,t1b), (t2aa,t2ab,t2bb))
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
    nao, nmo = mf.mo_coeff[0].shape
    numpy.random.seed(10)
    mf.mo_coeff = numpy.random.random((2,nao,nmo))

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
    mcc = uccsd.UCCSD(scf.addons.convert_to_uhf(mf))
    e3a = kernel(mcc, _ChemistsERIs(mcc, mf.mo_coeff), [t1a,t1b], [t2aa, t2ab, t2bb])
    print(e3a - 9877.2780859693339)

    mycc = cc.GCCSD(scf.addons.convert_to_ghf(mf))
    eris = mycc.ao2mo()
    t1 = mycc.spatial2spin(t1, eris.orbspin)
    t2 = mycc.spatial2spin(t2, eris.orbspin)
    from pyscf.cc import gccsd_t
    et = gccsd_t.kernel(mycc, eris, t1, t2)
    print(et - 9877.2780859693339)


    mol = gto.M()
    numpy.random.seed(12)
    nocca, noccb, nvira, nvirb = 3, 2, 4, 5
    nmo = nocca + nvira
    eris = cc.uccsd._ChemistsERIs()
    eri1 = (numpy.random.random((3,nmo,nmo,nmo,nmo)) +
            numpy.random.random((3,nmo,nmo,nmo,nmo)) * .8j - .5-.4j)
    eri1 = eri1 + eri1.transpose(0,2,1,4,3).conj()
    eri1[0] = eri1[0] + eri1[0].transpose(2,3,0,1)
    eri1[2] = eri1[2] + eri1[2].transpose(2,3,0,1)
    eri1 *= .1
    eris.vovp = eri1[0,nocca:,:nocca,nocca:,:     ]
    eris.vooo = eri1[0,nocca:,:nocca,:nocca,:nocca]
    eris.VOVP = eri1[2,noccb:,:noccb,noccb:,:     ]
    eris.VOOO = eri1[2,noccb:,:noccb,:noccb,:noccb]
    eris.voVP = eri1[1,nocca:,:nocca,noccb:,:     ]
    eris.voOO = eri1[1,nocca:,:nocca,:noccb,:noccb]
    eris.VOvp = eri1[1,nocca:,:     ,noccb:,:noccb].transpose(2,3,0,1)
    eris.VOoo = eri1[1,:nocca,:nocca,noccb:,:noccb].transpose(2,3,0,1)
    t1a  = .1 * numpy.random.random((nocca,nvira)) + numpy.random.random((nocca,nvira))*.1j
    t1b  = .1 * numpy.random.random((noccb,nvirb)) + numpy.random.random((noccb,nvirb))*.1j
    t2aa = .1 * numpy.random.random((nocca,nocca,nvira,nvira)) + numpy.random.random((nocca,nocca,nvira,nvira))*.1j
    t2aa = t2aa - t2aa.transpose(0,1,3,2)
    t2aa = t2aa - t2aa.transpose(1,0,2,3)
    t2bb = .1 * numpy.random.random((noccb,noccb,nvirb,nvirb)) + numpy.random.random((noccb,noccb,nvirb,nvirb))*.1j
    t2bb = t2bb - t2bb.transpose(0,1,3,2)
    t2bb = t2bb - t2bb.transpose(1,0,2,3)
    t2ab = .1 * numpy.random.random((nocca,noccb,nvira,nvirb)) + numpy.random.random((nocca,noccb,nvira,nvirb))*.1j
    f = (numpy.random.random((2,nmo,nmo)) * .4 +
         numpy.random.random((2,nmo,nmo)) * .4j)
    eris.focka = f[0]+f[0].T.conj() + numpy.diag(numpy.arange(nmo))
    eris.fockb = f[1]+f[1].T.conj() + numpy.diag(numpy.arange(nmo))
    t1 = t1a, t1b
    t2 = t2aa, t2ab, t2bb
    mcc = cc.UCCSD(scf.UHF(mol))
    print(kernel(mcc, eris, t1, t2) - (-0.056092415718338388-0.011390417704868244j))
