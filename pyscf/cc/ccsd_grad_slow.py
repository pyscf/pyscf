#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import time
import ctypes
import tempfile
import numpy
import h5py
import pyscf.lib as lib
from pyscf.lib import logger
import pyscf.ao2mo
import pyscf.cc.ccsd_slow as ccsd
from pyscf.cc import ccsd_rdm
from pyscf import grad

libcc = lib.load_library('libcc')

def IX_intermediates(cc, t1, t2, l1, l2, eris=None, d1=None, d2=None):
    if eris is None:
# Note eris are in Chemist's notation
        eris = ccsd._ERIS(cc)
    if d1 is None:
        doo, dov, dvo, dvv = ccsd_rdm.gamma1_intermediates(cc, t1, t2, l1, l2)
    else:
        doo, dov, dvo, dvv = d1
    if d2 is None:
# Note gamma2 are in Chemist's notation
        d2 = ccsd_rdm.gamma2_intermediates(cc, t1, t2, l1, l2)
    dovov, dvvvv, doooo, doovv, dovvo, dvvov, dovvv, dooov = d2
    dvvov = dovvv.transpose(2,3,0,1)
    nocc, nvir = t1.shape
    dvvvv = pyscf.ao2mo.restore(1, dvvvv, nvir).reshape((nvir,)*4)

# Note Ioo is not hermitian
    Ioo  =(numpy.einsum('jakb,iakb->ij', dovov, eris.ovov)
         + numpy.einsum('kbja,iakb->ij', dovov, eris.ovov))
    Ioo +=(numpy.einsum('jabk,iakb->ij', dovvo, eris.ovov)
         + numpy.einsum('kbaj,iakb->ij', dovvo, eris.ovov)
         + numpy.einsum('jkab,ikab->ij', doovv, eris.oovv)
         + numpy.einsum('kjba,ikab->ij', doovv, eris.oovv))
    Ioo +=(numpy.einsum('jmlk,imlk->ij', doooo, eris.oooo) * 2
         + numpy.einsum('mjkl,imlk->ij', doooo, eris.oooo) * 2)
    Ioo +=(numpy.einsum('jlka,ilka->ij', dooov, eris.ooov)
         + numpy.einsum('klja,klia->ij', dooov, eris.ooov))
    Ioo += numpy.einsum('abjc,icab->ij', dvvov, eris.ovvv)
    Ioo += numpy.einsum('ljka,lika->ij', dooov, eris.ooov)
    Ioo *= -1

# Note Ivv is not hermitian
    Ivv  =(numpy.einsum('ibjc,iajc->ab', dovov, eris.ovov)
         + numpy.einsum('jcib,iajc->ab', dovov, eris.ovov))
    Ivv +=(numpy.einsum('jcbi,iajc->ab', dovvo, eris.ovov)
         + numpy.einsum('ibcj,iajc->ab', dovvo, eris.ovov)
         + numpy.einsum('jibc,jiac->ab', doovv, eris.oovv)
         + numpy.einsum('ijcb,jiac->ab', doovv, eris.oovv))
    Ivv +=(numpy.einsum('bced,aced->ab', dvvvv, eris.vvvv) * 2
         + numpy.einsum('cbde,aced->ab', dvvvv, eris.vvvv) * 2)
    Ivv +=(numpy.einsum('dbic,icda->ab', dvvov, eris.ovvv)
         + numpy.einsum('dcib,iadc->ab', dvvov, eris.ovvv))
    Ivv += numpy.einsum('bcid,idac->ab', dvvov, eris.ovvv)
    Ivv += numpy.einsum('jikb,jika->ab', dooov, eris.ooov)
    Ivv *= -1

    Ivo  =(numpy.einsum('kajb,kijb->ai', dovov, eris.ooov)
         + numpy.einsum('kbja,jikb->ai', dovov, eris.ooov))
    Ivo +=(numpy.einsum('acbd,icbd->ai', dvvvv, eris.ovvv) * 2
         + numpy.einsum('cadb,icbd->ai', dvvvv, eris.ovvv) * 2)
    Ivo +=(numpy.einsum('jbak,jbik->ai', dovvo, eris.ovoo)
         + numpy.einsum('kabj,jbik->ai', dovvo, eris.ovoo)
         + numpy.einsum('jkab,jkib->ai', doovv, eris.ooov)
         + numpy.einsum('kjba,jkib->ai', doovv, eris.ooov))
    Ivo +=(numpy.einsum('dajc,idjc->ai', dvvov, eris.ovov)
         + numpy.einsum('dcja,jidc->ai', dvvov, eris.oovv))
    Ivo += numpy.einsum('abjc,ibjc->ai', dvvov, eris.ovov)
    Ivo += numpy.einsum('jlka,jlki->ai', dooov, eris.oooo)
    Ivo *= -1

    Xvo  =(numpy.einsum('kj,kjia->ai', doo, eris.ooov) * 2
         + numpy.einsum('kj,kjia->ai', doo, eris.ooov) * 2
         - numpy.einsum('kj,kija->ai', doo, eris.ooov)
         - numpy.einsum('kj,ijka->ai', doo, eris.ooov))
    Xvo +=(numpy.einsum('cb,iacb->ai', dvv, eris.ovvv) * 2
         + numpy.einsum('cb,iacb->ai', dvv, eris.ovvv) * 2
         - numpy.einsum('cb,icab->ai', dvv, eris.ovvv)
         - numpy.einsum('cb,ibca->ai', dvv, eris.ovvv))
    Xvo +=(numpy.einsum('icjb,jbac->ai', dovov, eris.ovvv)
         + numpy.einsum('jcib,jcab->ai', dovov, eris.ovvv))
    Xvo +=(numpy.einsum('iklj,ljka->ai', doooo, eris.ooov) * 2
         + numpy.einsum('kijl,ljka->ai', doooo, eris.ooov) * 2)
    Xvo +=(numpy.einsum('ibcj,jcab->ai', dovvo, eris.ovvv)
         + numpy.einsum('jcbi,jcab->ai', dovvo, eris.ovvv)
         + numpy.einsum('ijcb,jacb->ai', doovv, eris.ovvv)
         + numpy.einsum('jibc,jacb->ai', doovv, eris.ovvv))
    Xvo +=(numpy.einsum('ijkb,jakb->ai', dooov, eris.ovov)
         + numpy.einsum('kjib,kjab->ai', dooov, eris.oovv))
    Xvo += numpy.einsum('dbic,dbac->ai', dvvov, eris.vvvv)
    Xvo += numpy.einsum('jikb,jakb->ai', dooov, eris.ovov)
    Xvo += Ivo
    return Ioo, Ivv, Ivo, Xvo


def response_dm1(cc, t1, t2, l1, l2, eris=None, IX=None):
    from pyscf.scf import cphf
    if eris is None:
# Note eris are in Chemist's notation
        eris = ccsd._ERIS(cc)
    if IX is None:
        Ioo, Ivv, Ivo, Xvo = IX_intermediates(cc, t1, t2, l1, l2, eris)
    else:
        Ioo, Ivv, Ivo, Xvo = IX
    nocc, nvir = t1.shape
    nmo = nocc + nvir
    def fvind(x):
        x = x.reshape(Xvo.shape)
        if eris is None:
            mo_coeff = cc.mo_coeff
            dm = reduce(numpy.dot, (mo_coeff[:,nocc:], x, mo_coeff[:,:nocc].T))
            v = reduce(numpy.dot, (mo_coeff[:,nocc:].T, cc._scf.get_veff(mol, dm),
                                   mo_coeff[:,:nocc]))
        else:
            v  = numpy.einsum('iajb,bj->ai', eris.ovov, x) * 4
            v -= numpy.einsum('jiab,bj->ai', eris.oovv, x)
            v -= numpy.einsum('ibja,bj->ai', eris.ovov, x)
        return v
    mo_energy = eris.fock.diagonal()
    mo_occ = numpy.zeros_like(mo_energy)
    mo_occ[:nocc] = 2
    dvo = cphf.solve(fvind, mo_energy, mo_occ, Xvo, max_cycle=30)[0]
    dm1 = numpy.zeros((nmo,nmo))
    dm1[nocc:,:nocc] = dvo
    dm1[:nocc,nocc:] = dvo.T
    return dm1


#
# Note: only works with canonical orbitals
# Non-canonical formula refers to JCP, 95, 2639
#
def kernel(cc, t1=None, t2=None, l1=None, l2=None, eris=None, atmlst=None):
    if t1 is None: t1 = cc.t1
    if t2 is None: t2 = cc.t2
    if l1 is None: l1 = cc.l1
    if l2 is None: l2 = cc.l2
    if eris is None: eris = ccsd._ERIS(cc)
    mol = cc.mol
    mo_coeff = cc.mo_coeff  #FIXME: ensure cc.mo_coeff is canonical orbital
    mo_energy = cc.mo_energy
    nocc, nvir = t1.shape
    nao, nmo = mo_coeff.shape
    d1 = ccsd_rdm.gamma1_intermediates(cc, t1, t2, l1, l2)
    d2 = ccsd_rdm.gamma2_intermediates(cc, t1, t2, l1, l2)
    Ioo, Ivv, Ivo, Xvo = IX_intermediates(cc, t1, t2, l1, l2, eris, d1, d2)
    doo, dov, dvo, dvv = d1
    dovov, dvvvv, doooo, doovv, dovvo, dvvov, dovvv, dooov = d2
    dvvov = dovvv.transpose(2,3,0,1)
    dvvvv = pyscf.ao2mo.restore(1, dvvvv, nvir).reshape((nvir,)*4)

    dm1mo = response_dm1(cc, t1, t2, l1, l2, eris, (Ioo, Ivv, Ivo, Xvo))
    dm1mo[:nocc,:nocc] = doo * 2
    dm1mo[nocc:,nocc:] = dvv * 2
    dm1ao = reduce(numpy.dot, (mo_coeff, dm1mo, mo_coeff.T))
    dm0 = numpy.dot(mo_coeff[:,:nocc], mo_coeff[:,:nocc].T)*2
    im1 = numpy.zeros_like(dm1ao)
    im1[:nocc,:nocc] = Ioo
    im1[nocc:,nocc:] = Ivv
    im1[nocc:,:nocc] = Ivo
    im1[:nocc,nocc:] = Ivo.T
    im1 = reduce(numpy.dot, (mo_coeff, im1, mo_coeff.T))
    dme0 = grad.rhf.make_rdm1e(cc._scf.mo_energy, mo_coeff, cc._scf.mo_occ)

    h1 =-(mol.intor('int1e_ipkin', comp=3)
         +mol.intor('int1e_ipnuc', comp=3))
    s1 =-mol.intor('int1e_ipovlp', comp=3)
    zeta = lib.direct_sum('i+j->ij', mo_energy, mo_energy) * .5
    zeta[nocc:,:nocc] = mo_energy[:nocc]
    zeta[:nocc,nocc:] = mo_energy[:nocc].reshape(-1,1)
    zeta = reduce(numpy.dot, (mo_coeff, zeta*dm1mo, mo_coeff.T))
    eri0 = mol.intor('int2e_ip1', 3).reshape(3,nao,nao,nao,nao)
    dm2 = numpy.zeros((nmo,)*4)
    dm2[:nocc,nocc:,:nocc,nocc:] = dovov
    dm2[nocc:,nocc:,nocc:,nocc:] = dvvvv
    dm2[:nocc,:nocc,:nocc,:nocc] = doooo
    dm2[:nocc,nocc:,nocc:,:nocc] = dovvo
    dm2[:nocc,:nocc,nocc:,nocc:] = doovv
    dm2[nocc:,nocc:,:nocc,nocc:] = dvvov
    dm2[:nocc,:nocc,:nocc,nocc:] = dooov
    for i in range(nocc):
        dm2[i,i,:,:] += dm1mo
        dm2[:,i,i,:] -= dm1mo * .5
    for i in range(nocc):  # for HF gradeint
        for j in range(nocc):
            dm2[i,i,j,j] += 1
            dm2[i,j,j,i] -= .5
    dm2 = numpy.einsum('pjkl,ip->ijkl', dm2, mo_coeff)
    dm2 = numpy.einsum('ipkl,jp->ijkl', dm2, mo_coeff)
    dm2 = numpy.einsum('ijpl,kp->ijkl', dm2, mo_coeff)
    dm2 = numpy.einsum('ijkp,lp->ijkl', dm2, mo_coeff)

    if atmlst is None:
        atmlst = range(mol.natm)
        offsetdic = mol.offset_nr_by_atom()
    de = numpy.empty((mol.natm,3))
    for k,ia in enumerate(atmlst):
        p0, p1 = offsetdic[ia]
# s[1] dot I, note matrix im1 is not hermitian
        de[k] =(numpy.einsum('xij,ij->x', s1[:,p0:p1], im1[p0:p1])
              + numpy.einsum('xji,ij->x', s1[:,p0:p1], im1[:,p0:p1]))
# h[1] \dot DM, *2 for +c.c.,  contribute to f1
        mol.set_rinv_origin(mol.atom_coord(k))
        vrinv = -mol.atom_charge(k) * mol.intor('int1e_iprinv', comp=3)
        de[k] +=(numpy.einsum('xij,ij->x', h1[:,p0:p1], dm1ao[p0:p1])
               + numpy.einsum('xji,ij->x', h1[:,p0:p1], dm1ao[:,p0:p1]))
        de[k] +=(numpy.einsum('xij,ij->x', vrinv, dm1ao)
               + numpy.einsum('xji,ij->x', vrinv, dm1ao))
# -s[1]*e \dot DM,  contribute to f1
        de[k] -=(numpy.einsum('xij,ij->x', s1[:,p0:p1], zeta[p0:p1])
               + numpy.einsum('xji,ij->x', s1[:,p0:p1], zeta[:,p0:p1]))
        s1ij = []
        for i in range(3):
            mocc = mo_coeff[:,:nocc]
            s1mo = reduce(numpy.dot, (mocc[p0:p1].T, s1[i,p0:p1], mocc))
            s1mo = s1mo + s1mo.T
            s1ij.append(reduce(numpy.dot, (mocc, s1mo, mocc.T)))
# -vhf[s_ij[1]],  contribute to f1, *2 because get_veff returns J-.5K
        de[k] -= numpy.einsum('xij,ij->x', cc._scf.get_veff(mol, s1ij), dm1ao)*2

# 2e AO integrals dot 2pdm
        de[k] -= numpy.einsum('xijkl,ijkl->x', eri0[:,p0:p1], dm2[p0:p1]) * 2
        de[k] -= numpy.einsum('xijkl,jikl->x', eri0[:,p0:p1], dm2[:,p0:p1]) * 2
        de[k] -= numpy.einsum('xijkl,klij->x', eri0[:,p0:p1], dm2[:,:,p0:p1]) * 2
        de[k] -= numpy.einsum('xijkl,klji->x', eri0[:,p0:p1], dm2[:,:,:,p0:p1]) * 2

# HF gradients, J1*2-K1 was merged into previous contraction to dm2
        de[k] +=(numpy.einsum('xij,ij->x', h1[:,p0:p1], dm0[p0:p1])
               + numpy.einsum('xji,ij->x', h1[:,p0:p1], dm0[:,p0:p1]))
        de[k] +=(numpy.einsum('xij,ij->x', vrinv, dm0)
               + numpy.einsum('xji,ij->x', vrinv, dm0))
        de[k] -=(numpy.einsum('xij,ij->x', s1[:,p0:p1], dme0[p0:p1])
               + numpy.einsum('xji,ij->x', s1[:,p0:p1], dme0[:,p0:p1]))
    de += grad.rhf.grad_nuc(mol)
    return de


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    import pyscf.cc.ccsd
    from pyscf import ao2mo
    from pyscf import grad

    mol = gto.M()
    mf = scf.RHF(mol)

    mycc = pyscf.cc.ccsd.CCSD(mf)

    numpy.random.seed(2)
    nocc = 5
    nmo = 12
    nvir = nmo - nocc
    eri0 = numpy.random.random((nmo,nmo,nmo,nmo))
    eri0 = ao2mo.restore(1, ao2mo.restore(8, eri0, nmo), nmo)
    fock0 = numpy.random.random((nmo,nmo))
    fock0 = fock0 + fock0.T + numpy.diag(range(nmo))*20
    t1 = numpy.random.random((nocc,nvir))
    t2 = numpy.random.random((nocc,nocc,nvir,nvir))
    t2 = t2 + t2.transpose(1,0,3,2)
    l1 = numpy.random.random((nocc,nvir))
    l2 = numpy.random.random((nocc,nocc,nvir,nvir))
    l2 = l2 + l2.transpose(1,0,3,2)

    h1 = fock0 - (numpy.einsum('kkpq->pq', eri0[:nocc,:nocc])*2
                - numpy.einsum('pkkq->pq', eri0[:,:nocc,:nocc]))
    eris = lambda:None
    eris.oooo = eri0[:nocc,:nocc,:nocc,:nocc].copy()
    eris.ooov = eri0[:nocc,:nocc,:nocc,nocc:].copy()
    eris.ovoo = eri0[:nocc,nocc:,:nocc,:nocc].copy()
    eris.oovo = eri0[:nocc,:nocc,nocc:,:nocc].copy()
    eris.oovv = eri0[:nocc,:nocc,nocc:,nocc:].copy()
    eris.ovov = eri0[:nocc,nocc:,:nocc,nocc:].copy()
    eris.ovvo = eri0[:nocc,nocc:,nocc:,:nocc].copy()
    eris.ovvv = eri0[:nocc,nocc:,nocc:,nocc:].copy()
    eris.vvvv = eri0[nocc:,nocc:,nocc:,nocc:].copy()
    eris.vvvo = eri0[nocc:,nocc:,nocc:,:nocc].copy()
    eris.vovv = eri0[nocc:,:nocc,nocc:,nocc:].copy()
    eris.vvov = eri0[nocc:,nocc:,:nocc,nocc:].copy()
    eris.vvoo = eri0[nocc:,nocc:,:nocc,:nocc].copy()
    eris.voov = eri0[nocc:,:nocc,:nocc,nocc:].copy()
    eris.vooo = eri0[nocc:,:nocc,:nocc,:nocc].copy()
    eris.fock = fock0

    print('-----------------------------------')
    Ioo, Ivv, Ivo, Xvo = IX_intermediates(mycc, t1, t2, l1, l2, eris)
    numpy.random.seed(1)
    h1 = numpy.random.random((nmo,nmo))
    h1 = h1 + h1.T
    print(numpy.einsum('ij,ij', h1[:nocc,:nocc], Ioo) - 2613213.0346526774)
    print(numpy.einsum('ab,ab', h1[nocc:,nocc:], Ivv) - 6873038.9907923322)
    print(numpy.einsum('ai,ai', h1[nocc:,:nocc], Ivo) - 4353360.4241635408)
    print(numpy.einsum('ai,ai', h1[nocc:,:nocc], Xvo) - 203575.42337558540)
    dm1 = response_dm1(mycc, t1, t2, l1, l2, eris)
    print(numpy.einsum('pq,pq', h1[nocc:,:nocc], dm1[nocc:,:nocc])--486.638981725713393)

    print('-----------------------------------')
    mol = gto.M(
        verbose = 0,
        atom = [
            ["O" , (0. , 0.     , 0.)],
            [1   , (0. , -0.757 , 0.587)],
            [1   , (0. , 0.757  , 0.587)]],
        basis = '631g'
    )
    mf = scf.RHF(mol)
    ehf = mf.scf()

    mycc = pyscf.cc.ccsd.CCSD(mf)
    mycc.conv_tol = 1e-10
    mycc.conv_tol_normt = 1e-10
    ecc, t1, t2 = mycc.kernel()
    l1, l2 = mycc.solve_lambda()
    g1 = kernel(mycc, t1, t2, l1, l2)
    print(g1)
#[[ 0   0                1.00950925e-02]
# [ 0   2.28063426e-02  -5.04754623e-03]
# [ 0  -2.28063426e-02  -5.04754623e-03]]

    lib.parameters.BOHR = 1
    r = 1.76#.748
    mol = gto.M(
        verbose = 0,
        atom = '''H 0 0 0; H 0 0 %f''' % r,
        basis = '631g')
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-14
    ehf0 = mf.scf()
    ghf = grad.RHF(mf).grad()
    mycc = pyscf.cc.ccsd.CCSD(mf)
    mycc.conv_tol = 1e-10
    mycc.conv_tol_normt = 1e-10
    ecc, t1, t2 = mycc.kernel()
    l1, l2 = mycc.solve_lambda()
    g1 = kernel(mycc, t1, t2, l1, l2)
    print(g1)
# [[ 0.          0.         -0.07080036]
#  [ 0.          0.          0.07080036]]

