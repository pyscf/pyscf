#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import time
import ctypes
import tempfile
import numpy
import h5py
from pyscf import lib
from pyscf.lib import logger
import pyscf.ao2mo
from pyscf.cc import ccsd
from pyscf import gto
from pyscf.cc import ccsd_rdm
from pyscf.cc import ccsd_grad_incore as ccsd_grad
from pyscf.cc import _ccsd
from pyscf.cc import ccsd_t_rdm_slow as ccsd_t_rdm
import pyscf.grad

libcc = lib.load_library('libcc')

def IX_intermediates(mycc, t1, t2, l1, l2, d1=None, d2=None, eris=None):
    if d1 is None:
        doo, dov, dvo, dvv = ccsd_t_rdm.gamma1_intermediates(mycc, t1, t2, l1, l2, eris)
    else:
        doo, dov, dvo, dvv = d1
    if d2 is None:
# Note gamma2 are in Chemist's notation
        d2 = ccsd_t_rdm.gamma2_intermediates(mycc, t1, t2, l1, l2, eris)
    else:
        dovov, dvvvv, doooo, doovv, dovvo, dvvov, dovvv, dooov = d2
    if dovvv is None:
        dovvv = dvvov.transpose(2,3,0,1)
    elif dvvov is None:
        dvvov = dovvv.transpose(2,3,0,1)

    nocc, nvir = t1.shape
    eris_ovvv = lib.unpack_tril(_cp(eris.ovvv).reshape(nocc*nvir,-1))
    eris_ovvv = eris_ovvv.reshape(nocc,nvir,nvir,nvir)
    eris_vvvv = pyscf.ao2mo.restore(1, _cp(eris.vvvv), nvir)
    dvvvv = pyscf.ao2mo.restore(1, _cp(dvvvv), nvir)

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
    Ioo += numpy.einsum('abjc,icab->ij', dvvov, eris_ovvv)
    Ioo += numpy.einsum('ljka,lika->ij', dooov, eris.ooov)
    Ioo *= -1

# Note Ivv is not hermitian
    Ivv  =(numpy.einsum('ibjc,iajc->ab', dovov, eris.ovov)
         + numpy.einsum('jcib,iajc->ab', dovov, eris.ovov))
    Ivv +=(numpy.einsum('jcbi,iajc->ab', dovvo, eris.ovov)
         + numpy.einsum('ibcj,iajc->ab', dovvo, eris.ovov)
         + numpy.einsum('jibc,jiac->ab', doovv, eris.oovv)
         + numpy.einsum('ijcb,jiac->ab', doovv, eris.oovv))
    Ivv +=(numpy.einsum('bced,aced->ab', dvvvv, eris_vvvv) * 2
         + numpy.einsum('cbde,aced->ab', dvvvv, eris_vvvv) * 2)
    Ivv +=(numpy.einsum('dbic,icda->ab', dvvov, eris_ovvv)
         + numpy.einsum('dcib,iadc->ab', dvvov, eris_ovvv))
    Ivv += numpy.einsum('bcid,idac->ab', dvvov, eris_ovvv)
    Ivv += numpy.einsum('jikb,jika->ab', dooov, eris.ooov)
    Ivv *= -1

    Ivo  =(numpy.einsum('kajb,kijb->ai', dovov, eris.ooov)
         + numpy.einsum('kbja,jikb->ai', dovov, eris.ooov))
    Ivo +=(numpy.einsum('acbd,icbd->ai', dvvvv, eris_ovvv) * 2
         + numpy.einsum('cadb,icbd->ai', dvvvv, eris_ovvv) * 2)
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
    Xvo +=(numpy.einsum('cb,iacb->ai', dvv, eris_ovvv) * 2
         + numpy.einsum('cb,iacb->ai', dvv, eris_ovvv) * 2
         - numpy.einsum('cb,icab->ai', dvv, eris_ovvv)
         - numpy.einsum('cb,ibca->ai', dvv, eris_ovvv))
    Xvo +=(numpy.einsum('icjb,jbac->ai', dovov, eris_ovvv)
         + numpy.einsum('jcib,jcab->ai', dovov, eris_ovvv))
    Xvo +=(numpy.einsum('iklj,ljka->ai', doooo, eris.ooov) * 2
         + numpy.einsum('kijl,ljka->ai', doooo, eris.ooov) * 2)
    Xvo +=(numpy.einsum('ibcj,jcab->ai', dovvo, eris_ovvv)
         + numpy.einsum('jcbi,jcab->ai', dovvo, eris_ovvv)
         + numpy.einsum('ijcb,jacb->ai', doovv, eris_ovvv)
         + numpy.einsum('jibc,jacb->ai', doovv, eris_ovvv))
    Xvo +=(numpy.einsum('ijkb,jakb->ai', dooov, eris.ovov)
         + numpy.einsum('kjib,kjab->ai', dooov, eris.oovv))
    Xvo += numpy.einsum('dbic,dbac->ai', dvvov, eris_vvvv)
    Xvo += numpy.einsum('jikb,jakb->ai', dooov, eris.ovov)
    Xvo += Ivo
    return Ioo, Ivv, Ivo, Xvo


# Only works with canonical orbitals
def kernel(mycc, t1=None, t2=None, l1=None, l2=None, eris=None, atmlst=None,
           mf_grad=None, verbose=logger.INFO):
    if t1 is None: t1 = mycc.t1
    if t2 is None: t2 = mycc.t2
    if l1 is None: l1 = mycc.l1
    if l2 is None: l2 = mycc.l2
    if eris is None: eris = ccsd._ERIS(mycc)
    if mf_grad is None:
        mf_grad = pyscf.grad.RHF(mycc._scf)

    log = logger.Logger(mycc.stdout, mycc.verbose)
    time0 = time.clock(), time.time()
    mol = mycc.mol
    if mycc.frozen is not 0:
        raise NotImplementedError('frozen orbital ccsd_grad')
    moidx = ccsd.get_moidx(mycc)
    mo_coeff = mycc.mo_coeff[:,moidx]  #FIXME: ensure mycc.mo_coeff is canonical orbital
    mo_energy = eris.fock.diagonal()
    nocc, nvir = t1.shape
    nao, nmo = mo_coeff.shape
    nao_pair = nao * (nao+1) // 2

    log.debug('Build ccsd rdm1 intermediates')
    d1 = ccsd_t_rdm.gamma1_intermediates(mycc, t1, t2, l1, l2, eris)
    doo, dov, dvo, dvv = d1
    time1 = log.timer('rdm1 intermediates', *time0)

    log.debug('Build ccsd rdm2 intermediates')
    d2 = ccsd_t_rdm.gamma2_intermediates(mycc, t1, t2, l1, l2, eris)
    time1 = log.timer('rdm2 intermediates', *time1)
    log.debug('Build ccsd response_rdm1')
    Ioo, Ivv, Ivo, Xvo = IX_intermediates(mycc, t1, t2, l1, l2, d1, d2, eris)
    time1 = log.timer('response_rdm1 intermediates', *time1)

    dm1mo = ccsd_grad.response_dm1(mycc, t1, t2, l1, l2, eris, (Ioo, Ivv, Ivo, Xvo))
    dm1mo[:nocc,:nocc] = doo * 2
    dm1mo[nocc:,nocc:] = dvv * 2
    dm1ao = reduce(numpy.dot, (mo_coeff, dm1mo, mo_coeff.T))
    im1 = numpy.zeros_like(dm1mo)
    im1[:nocc,:nocc] = Ioo
    im1[nocc:,nocc:] = Ivv
    im1[nocc:,:nocc] = Ivo
    im1[:nocc,nocc:] = Ivo.T
    im1 = reduce(numpy.dot, (mo_coeff, im1, mo_coeff.T))
    time1 = log.timer('response_rdm1', *time1)

    log.debug('symmetrized rdm2 and MO->AO transformation')
    dm2ao = ccsd_grad._rdm2_mo2ao(mycc, d2, dm1mo, mo_coeff)
    time1 = log.timer('MO->AO transformation', *time1)

#TODO: pass hf_grad object to compute h1 and s1
    log.debug('h1 and JK1')
    h1 = mf_grad.get_hcore(mol)
    s1 = mf_grad.get_ovlp(mol)
    zeta = numpy.empty((nmo,nmo))
    zeta[:nocc,:nocc] = (mo_energy[:nocc].reshape(-1,1) + mo_energy[:nocc]) * .5
    zeta[nocc:,nocc:] = (mo_energy[nocc:].reshape(-1,1) + mo_energy[nocc:]) * .5
    zeta[nocc:,:nocc] = mo_energy[:nocc]
    zeta[:nocc,nocc:] = mo_energy[:nocc].reshape(-1,1)
    zeta = reduce(numpy.dot, (mo_coeff, zeta*dm1mo, mo_coeff.T))
    p1 = numpy.dot(mo_coeff[:,:nocc], mo_coeff[:,:nocc].T)
    vhf4sij = reduce(numpy.dot, (p1, mycc._scf.get_veff(mol, dm1ao+dm1ao.T), p1))
    time1 = log.timer('h1 and JK1', *time1)

    # Hartree-Fock part contribution
    hf_dm1 = mycc._scf.make_rdm1(mycc.mo_coeff, mycc.mo_occ)
    dm1ao += hf_dm1
    zeta += mf_grad.make_rdm1e(mo_energy, mycc.mo_coeff, mycc.mo_occ)

    if atmlst is None:
        atmlst = range(mol.natm)
    offsetdic = mol.offset_nr_by_atom()
    de = numpy.zeros((len(atmlst),3))
    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = offsetdic[ia]
# s[1] dot I, note matrix im1 is not hermitian
        de[k] =(numpy.einsum('xij,ij->x', s1[:,p0:p1], im1[p0:p1])
              + numpy.einsum('xji,ij->x', s1[:,p0:p1], im1[:,p0:p1]))
# h[1] \dot DM, *2 for +c.c.,  contribute to f1
        vrinv = mf_grad._grad_rinv(mol, ia)
        de[k] +=(numpy.einsum('xij,ij->x', h1[:,p0:p1], dm1ao[p0:p1]  )
               + numpy.einsum('xji,ij->x', h1[:,p0:p1], dm1ao[:,p0:p1]))
        de[k] +=(numpy.einsum('xij,ij->x', vrinv, dm1ao)
               + numpy.einsum('xji,ij->x', vrinv, dm1ao))
# -s[1]*e \dot DM,  contribute to f1
        de[k] -=(numpy.einsum('xij,ij->x', s1[:,p0:p1], zeta[p0:p1]  )
               + numpy.einsum('xji,ij->x', s1[:,p0:p1], zeta[:,p0:p1]))
# -vhf[s_ij[1]],  contribute to f1, *2 for s1+s1.T
        de[k] -= numpy.einsum('xij,ij->x', s1[:,p0:p1], vhf4sij[p0:p1]) * 2

# 2e AO integrals dot 2pdm
        eri1 = mol.intor('int2e_ip1', comp=3, aosym='s2kl',
                         shls_slice=(shl0,shl1,0,mol.nbas,0,mol.nbas,0,mol.nbas))
        eri1 = eri1.reshape(3,p1-p0,nao,-1)
        dm2buf = ccsd_grad._load_block_tril(dm2ao, p0, p1)
        de[k] -= numpy.einsum('xijk,ijk->x', eri1, dm2buf) * 2

        for i in range(3):
            #:tmp = lib.unpack_tril(eri1[i].reshape(-1,nao_pair))
            #:vj = numpy.einsum('ijkl,kl->ij', tmp, hf_dm1)
            #:vk = numpy.einsum('ijkl,jk->il', tmp, hf_dm1)
            vj, vk = ccsd_grad.hf_get_jk_incore(eri1[i], hf_dm1)
            de[k,i] -=(numpy.einsum('ij,ij->', vj, hf_dm1[p0:p1])
                     - numpy.einsum('ij,ij->', vk, hf_dm1[p0:p1])*.5) * 2
        eri1 = dm2buf = None
        log.debug('grad of atom %d %s = %s', ia, mol.atom_symbol(ia), de[k])
        time1 = log.timer('grad of atom %d'%ia, *time1)

    log.note('CCSD gradinets')
    log.note('==============')
    log.note('           x                y                z')
    for k, ia in enumerate(atmlst):
        log.note('%d %s  %15.9f  %15.9f  %15.9f', ia, mol.atom_symbol(ia),
                 de[k,0], de[k,1], de[k,2])
    log.timer('CCSD gradients', *time0)
    return de


def _cp(a):
    return numpy.array(a, copy=False, order='C')


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf.cc import ccsd_t_slow as ccsd_t
    from pyscf import ao2mo
    from pyscf import grad
    from pyscf.cc import ccsd_t_lambda_slow as ccsd_t_lambda

    mol = gto.M(
        verbose = 0,
        atom = [
            ["O" , (0. , 0.     , 0.    )],
            [1   , (0. ,-0.757  ,-0.587)],
            [1   , (0. , 0.757  ,-0.587)]],
        basis = '631g'
    )
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-14
    ehf = mf.scf()

    mycc = ccsd.CCSD(mf)
    mycc.conv_tol = 1e-10
    mycc.conv_tol_normt = 1e-10
    ecc, t1, t2 = mycc.kernel()
    eris = mycc.ao2mo()
    e3ref = ccsd_t.kernel(mycc, eris, t1, t2)
    print(ehf+ecc+e3ref)
    eris = mycc.ao2mo(mf.mo_coeff)
    conv, l1, l2 = ccsd_t_lambda.kernel(mycc, eris, t1, t2)
    g1 = kernel(mycc, t1, t2, l1, l2, eris=eris, mf_grad=grad.RHF(mf))
    print(g1 + grad.grad_nuc(mol))
#O      0.0000000000            0.0000000000           -0.0112045345
#H      0.0000000000            0.0234464201            0.0056022672
#H      0.0000000000           -0.0234464201            0.0056022672


    mol = gto.M(
        verbose = 0,
        atom = '''
H         -1.90779510     0.92319522     0.08700656
H         -1.08388168    -1.61405643    -0.07315086
H          2.02822318    -0.61402169     0.09396693
H          0.96345360     1.30488291    -0.10782263
               ''',
        unit='bohr',
        basis = '631g')
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-14
    ehf0 = mf.scf()

    mycc = ccsd.CCSD(mf)
    mycc.conv_tol = 1e-10
    mycc.conv_tol_normt = 1e-10
    ecc, t1, t2 = mycc.kernel()
    eris = mycc.ao2mo()
    e3ref = ccsd_t.kernel(mycc, eris, t1, t2)
    print(ehf0+ecc+e3ref)
    eris = mycc.ao2mo(mf.mo_coeff)
    conv, l1, l2 = ccsd_t_lambda.kernel(mycc, eris, t1, t2)
    g1 = kernel(mycc, t1, t2, l1, l2, eris=eris, mf_grad=grad.RHF(mf))
    print(g1 + grad.grad_nuc(mol))
#CCSD
#H   0.0113620114            0.0664344363            0.0029855587
#H   0.0528858926           -0.0483942979           -0.0033960631
#H   0.0109676543           -0.0827248466            0.0074005299
#H  -0.0752155583            0.0646847082           -0.0069900255
#
#CCSD(T) gradient:
#
#H   0.0112264011            0.0658917731            0.0029936671
#H   0.0525667206           -0.0481602008           -0.0033751503
#H   0.0107197424           -0.0823677005            0.0073804163
#H  -0.0745128642            0.0646361282           -0.0069989330 
