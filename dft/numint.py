#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import ctypes
import _ctypes
import time
import numpy
import scipy.linalg
import pyscf.lib
import pyscf.dft.vxc

libdft = pyscf.lib.load_library('libdft')
OCCDROP = 1e-12
ALWAYSMALL = 3
BLKSIZE = 224

def eval_ao(mol, coords, isgga=False, relativity=0, bastart=0, bascount=None,
            non0tab=None, verbose=None):
    assert(coords.flags.c_contiguous)
    natm = ctypes.c_int(mol._atm.shape[0])
    nbas = ctypes.c_int(mol.nbas)
    nao = mol.nao_nr()
    ngrids = len(coords)
    if bascount is None:
        bascount = mol.nbas - bastart
    if isgga:
        ao = numpy.empty((4, ngrids,nao)) # plain, dx, dy, dz
        feval = _ctypes.dlsym(libdft._handle, 'VXCeval_nr_gto_grad')
    else:
        ao = numpy.empty((ngrids,nao))
        feval = _ctypes.dlsym(libdft._handle, 'VXCeval_nr_gto')

    if non0tab is None:
        non0tab = numpy.ones(((ngrids+BLKSIZE-1)//BLKSIZE,mol.nbas),
                             dtype=numpy.int8)

    libdft.VXCeval_ao_drv(ctypes.c_void_p(feval),
                          ctypes.c_int(nao), ctypes.c_int(ngrids),
                          ctypes.c_int(bastart), ctypes.c_int(bascount),
                          ctypes.c_int(BLKSIZE),
                          ao.ctypes.data_as(ctypes.c_void_p),
                          coords.ctypes.data_as(ctypes.c_void_p),
                          non0tab.ctypes.data_as(ctypes.c_void_p),
                          mol._atm.ctypes.data_as(ctypes.c_void_p), natm,
                          mol._bas.ctypes.data_as(ctypes.c_void_p), nbas,
                          mol._env.ctypes.data_as(ctypes.c_void_p))
    return ao

def make_mask(mol, coords, relativity=0, bastart=0, bascount=None,
              verbose=None):
    assert(coords.flags.c_contiguous)
    natm = ctypes.c_int(mol._atm.shape[0])
    nbas = ctypes.c_int(mol.nbas)
    ngrids = len(coords)
    if bascount is None:
        bascount = mol.nbas - bastart

    non0tab = numpy.empty(((ngrids+BLKSIZE-1)//BLKSIZE,mol.nbas),
                          dtype=numpy.int8)
    libdft.VXCnr_ao_screen(non0tab.ctypes.data_as(ctypes.c_void_p),
                           coords.ctypes.data_as(ctypes.c_void_p),
                           ctypes.c_int(ngrids), ctypes.c_int(BLKSIZE),
                           mol._atm.ctypes.data_as(ctypes.c_void_p), natm,
                           mol._bas.ctypes.data_as(ctypes.c_void_p), nbas,
                           mol._env.ctypes.data_as(ctypes.c_void_p))
    return non0tab

def eval_rho(mol, ao, dm, non0tab=None, isgga=False, verbose=None):
    if isgga:
        ngrids, nao = ao[0].shape
    else:
        ngrids, nao = ao.shape

    if non0tab is None:
        non0tab = numpy.ones(((ngrids+BLKSIZE-1)//BLKSIZE,mol.nbas),
                             dtype=numpy.int8)
    if isgga:
        rho = numpy.empty((4,ngrids))
        c0 = _dot_ao_dm(mol, ao[0], dm, nao, ngrids, non0tab)
        rho[0] = numpy.einsum('pi,pi->p', ao[0], c0)
        for i in range(1, 4):
            c1 = _dot_ao_dm(mol, ao[i], dm, nao, ngrids, non0tab)
            rho[i] = numpy.einsum('pi,pi->p', ao[0], c1) * 2 # *2 for +c.c.
    else:
        c0 = _dot_ao_dm(mol, ao, dm, nao, ngrids, non0tab)
        rho = numpy.einsum('pi,pi->p', ao, c0)
    return rho

def eval_rho2(mol, ao, mo_coeff, mo_occ, non0tab=None, isgga=False,
              verbose=None):
    if isgga:
        ngrids, nao = ao[0].shape
    else:
        ngrids, nao = ao.shape

    if non0tab is None:
        non0tab = numpy.ones(((ngrids+BLKSIZE-1)//BLKSIZE,mol.nbas),
                             dtype=numpy.int8)
    pos = mo_occ > OCCDROP
    cpos = numpy.einsum('ij,j->ij', mo_coeff[:,pos], numpy.sqrt(mo_occ[pos]))
    if isgga:
        rho = numpy.empty((4,ngrids))
        c0 = _dot_ao_dm(mol, ao[0], cpos, nao, ngrids, non0tab)
        rho[0] = numpy.einsum('pi,pi->p', c0, c0)
        for i in range(1, 4):
            c1 = _dot_ao_dm(mol, ao[i], cpos, nao, ngrids, non0tab)
            rho[i] = numpy.einsum('pi,pi->p', c0, c1) * 2 # *2 for +c.c.
    else:
        c0 = _dot_ao_dm(mol, ao, cpos, nao, ngrids, non0tab)
        rho = numpy.einsum('pi,pi->p', c0, c0)

    neg = mo_occ < -OCCDROP
    if neg.sum() > 0:
        cneg = numpy.einsum('ij,j->ij', mo_coeff[:,neg], numpy.sqrt(-mo_occ[neg]))
        if isgga:
            c0 = _dot_ao_dm(mol, ao[0], cneg, nao, ngrids, non0tab)
            rho[0] -= numpy.einsum('pi,pi->p', c0, c0)
            for i in range(1, 4):
                c1 = _dot_ao_dm(mol, ao[i], cneg, nao, ngrids, non0tab)
                rho[i] -= numpy.einsum('pi,pi->p', c0, c1) * 2 # *2 for +c.c.
        else:
            c0 = _dot_ao_dm(mol, ao, cneg, nao, ngrids, non0tab)
            rho -= numpy.einsum('pi,pi->p', c0, c0)
    return rho

def eval_mat(mol, ao, weight, rho, vrho, vsigma=None, non0tab=None,
             isgga=False, verbose=None):
    ''' RKS matrix
    '''
    if isgga:
        ngrids, nao = ao[0].shape
    else:
        ngrids, nao = ao.shape

    if non0tab is None:
        non0tab = numpy.ones(((ngrids+BLKSIZE-1)//BLKSIZE,mol.nbas),
                             dtype=numpy.int8)
    if isgga:
        assert(vsigma is not None and rho.ndim==2)
        #wv = weight * vsigma * 2
        #aow  = numpy.einsum('pi,p->pi', ao[1], rho[1]*wv)
        #aow += numpy.einsum('pi,p->pi', ao[2], rho[2]*wv)
        #aow += numpy.einsum('pi,p->pi', ao[3], rho[3]*wv)
        #aow += numpy.einsum('pi,p->pi', ao[0], .5*weight*vrho)
        wv = numpy.empty_like(rho)
        wv[0]  = weight * vrho * .5
        wv[1:] = rho[1:] * (weight * vsigma * 2)
        aow = numpy.einsum('npi,np->pi', ao, wv)
        #mat = pyscf.lib.dot(ao[0].T, aow)
        mat = _dot_ao_ao(mol, ao[0], aow, nao, ngrids, non0tab)
    else:
        # *.5 because return mat + mat.T
        #:aow = numpy.einsum('pi,p->pi', ao, .5*weight*vrho)
        aow = ao * (.5*weight*vrho).reshape(-1,1)
        #mat = pyscf.lib.dot(ao.T, aow)
        mat = _dot_ao_ao(mol, ao, aow, nao, ngrids, non0tab)
    return mat + mat.T

def eval_x(x_id, rho, sigma, spin=0, relativity=0, verbose=None):
    '''
    For LDA and GGA functional, ec, vcrho, vcsigma are not zero
    For hybrid functional, ec, vcrho, vcsigma are zero
    '''
    rho = numpy.asarray(rho, order='C')
    sigma = numpy.asarray(sigma, order='C')
    ngrids = len(rho)
    if spin == 0:
        exc = numpy.empty(ngrids)
        vrho = numpy.empty(ngrids)
        vsigma = numpy.empty(ngrids)
        nspin = 1
    else:
        exc = numpy.empty(ngrids)
        vrho = numpy.empty((ngrids,2))
        vsigma = numpy.empty((ngrids,3))
        nspin = 2
    libdft.VXCnr_eval_x(ctypes.c_int(x_id),
                        ctypes.c_int(nspin), ctypes.c_int(relativity),
                        ctypes.c_int(ngrids),
                        rho.ctypes.data_as(ctypes.c_void_p),
                        sigma.ctypes.data_as(ctypes.c_void_p),
                        exc.ctypes.data_as(ctypes.c_void_p),
                        vrho.ctypes.data_as(ctypes.c_void_p),
                        vsigma.ctypes.data_as(ctypes.c_void_p))
    return exc, vrho, vsigma

def eval_c(c_id, rho, sigma, spin=0, relativity=0, verbose=None):
    rho = numpy.asarray(rho, order='C')
    sigma = numpy.asarray(sigma, order='C')
    ngrids = len(rho)
    if spin == 0:
        exc = numpy.empty(ngrids)
        vrho = numpy.empty(ngrids)
        vsigma = numpy.empty(ngrids)
        nspin = 1
    else:
        exc = numpy.empty(ngrids)
        vrho = numpy.empty((ngrids,2))
        vsigma = numpy.empty((ngrids,3))
        nspin = 2
    libdft.VXCnr_eval_c(ctypes.c_int(c_id),
                        ctypes.c_int(nspin), ctypes.c_int(relativity),
                        ctypes.c_int(ngrids),
                        rho.ctypes.data_as(ctypes.c_void_p),
                        sigma.ctypes.data_as(ctypes.c_void_p),
                        exc.ctypes.data_as(ctypes.c_void_p),
                        vrho.ctypes.data_as(ctypes.c_void_p),
                        vsigma.ctypes.data_as(ctypes.c_void_p))
    return exc, vrho, vsigma

def eval_xc(x_id, c_id, rho, sigma, spin=0, relativity=0, verbose=None):
    '''
    For LDA and GGA functional, ec, vcrho, vcsigma are not zero
    For hybrid functional, ec, vcrho, vcsigma are zero

    Args
        rho[n] if spin = 0
        rho[n,2] if spin = 1
    '''
    rho = numpy.asarray(rho, order='C')
    sigma = numpy.asarray(sigma, order='C')
    ngrids = len(rho)
    if spin == 0:
        exc = numpy.empty(ngrids)
        vrho = numpy.empty(ngrids)
        vsigma = numpy.empty(ngrids)
        nspin = 1
    else:
        exc = numpy.empty(ngrids)
        vrho = numpy.empty((ngrids,2))
        vsigma = numpy.empty((ngrids,3))
        nspin = 2
    libdft.VXCnr_eval_xc(ctypes.c_int(x_id), ctypes.c_int(c_id),
                         ctypes.c_int(nspin), ctypes.c_int(relativity),
                         ctypes.c_int(ngrids),
                         rho.ctypes.data_as(ctypes.c_void_p),
                         sigma.ctypes.data_as(ctypes.c_void_p),
                         exc.ctypes.data_as(ctypes.c_void_p),
                         vrho.ctypes.data_as(ctypes.c_void_p),
                         vsigma.ctypes.data_as(ctypes.c_void_p))
    return exc, vrho, vsigma


def _dot_ao_ao(mol, ao1, ao2, nao, ngrids, non0tab):
    return pyscf.lib.dot(ao1.T, ao2)
    natm = ctypes.c_int(mol._atm.shape[0])
    nbas = ctypes.c_int(mol.nbas)
    vv = numpy.empty((nao,nao))
    libdft.VXCdot_ao_ao(vv.ctypes.data_as(ctypes.c_void_p),
                        ao1.ctypes.data_as(ctypes.c_void_p),
                        ao2.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(nao), ctypes.c_int(ngrids),
                        ctypes.c_int(BLKSIZE),
                        non0tab.ctypes.data_as(ctypes.c_void_p),
                        mol._atm.ctypes.data_as(ctypes.c_void_p), natm,
                        mol._bas.ctypes.data_as(ctypes.c_void_p), nbas,
                        mol._env.ctypes.data_as(ctypes.c_void_p))
    return vv

def _dot_ao_dm(mol, ao, dm, nao, ngrids, non0tab):
    #return pyscf.lib.dot(ao, dm)
    natm = ctypes.c_int(mol._atm.shape[0])
    nbas = ctypes.c_int(mol.nbas)
    vm = numpy.empty((ngrids,dm.shape[1]))
    libdft.VXCdot_ao_dm(vm.ctypes.data_as(ctypes.c_void_p),
                        ao.ctypes.data_as(ctypes.c_void_p),
                        dm.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(nao), ctypes.c_int(dm.shape[1]),
                        ctypes.c_int(ngrids), ctypes.c_int(BLKSIZE),
                        non0tab.ctypes.data_as(ctypes.c_void_p),
                        mol._atm.ctypes.data_as(ctypes.c_void_p), natm,
                        mol._bas.ctypes.data_as(ctypes.c_void_p), nbas,
                        mol._env.ctypes.data_as(ctypes.c_void_p))
    return vm

def nr_vxc(mol, grids, x_id, c_id, dm, spin=0, relativity=0, hermi=1,
           max_memory=2000, verbose=None):
    '''RKS Vxc matrix
    '''
    nao = dm.shape[0]
    ngrids = len(grids.weights)
    blksize = min(int(max_memory/6*1e6/8/nao), ngrids)
    nelec = 0
    excsum = 0
    vmat = numpy.zeros_like(dm)
    for ip0 in range(0, ngrids, blksize):
        ip1 = min(ngrids, ip0+blksize)
        coords = grids.coords[ip0:ip1]
        weight = grids.weights[ip0:ip1]
        if pyscf.dft.vxc._is_lda(x_id) and pyscf.dft.vxc._is_lda(c_id):
            isgga = False
            ao = eval_ao(mol, coords, isgga=isgga)
            rho = eval_rho(mol, ao, dm, isgga=isgga)
            exc, vrho, vsigma = eval_xc(x_id, c_id, rho, rho,
                                        spin, relativity, verbose)
            den = rho*weight
            nelec += den.sum()
            excsum += (den*exc).sum()
        else:
            isgga = True
            ao = eval_ao(mol, coords, isgga=isgga)
            rho = eval_rho(mol, ao, dm, isgga=isgga)
            sigma = numpy.einsum('ip,ip->p', rho[1:], rho[1:])
            exc, vrho, vsigma = eval_xc(x_id, c_id, rho[0], sigma,
                                        spin, relativity, verbose)
            den = rho[0]*weight
            nelec += den.sum()
            excsum += (den*exc).sum()
        vmat += eval_mat(mol, ao, weight, rho, vrho, vsigma, isgga=isgga,
                         verbose=verbose)
    return nelec, excsum, vmat


class _NumInt(object):
    def __init__(self):
        self.non0tab = None

    def nr_vxc(self, mol, grids, x_id, c_id, dm, spin=0, relativity=0, hermi=1,
               max_memory=2000, verbose=None):
        if spin == 0:
            return self.nr_rks(mol, grids, x_id, c_id, dm, hermi=hermi,
                               max_memory=max_memory, verbose=verbose)
        else:
            return self.nr_uks(mol, grids, x_id, c_id, dm, hermi=hermi,
                               max_memory=max_memory, verbose=verbose)

    def nr_rks(self, mol, grids, x_id, c_id, dms, hermi=1,
               max_memory=2000, verbose=None):
        if self.non0tab is None:
            self.non0tab = make_mask(mol, grids.coords)
        nao = mol.nao_nr()
        ngrids = len(grids.weights)
# NOTE to index self.non0tab, the blksize needs to be the integer multiplier of BLKSIZE
        blksize = min(int(max_memory/6*1e6/8/nao/BLKSIZE)*BLKSIZE, ngrids)
        if pyscf.dft.vxc._is_lda(x_id) and pyscf.dft.vxc._is_lda(c_id):
            isgga = False
        else:
            isgga = True

        natocc = []
        natorb = []
        if isinstance(dms, numpy.ndarray) and dms.ndim == 2:
            e, c = scipy.linalg.eigh(dms)
            natocc.append(e)
            natorb.append(c)
        else:
            for dm in dms:
                e, c = scipy.linalg.eigh(dm)
                natocc.append(e)
                natorb.append(c)
        nset = len(natocc)
        nelec = numpy.zeros(nset)
        excsum = numpy.zeros(nset)
        vmat = numpy.zeros((nset,nao,nao))
        for ip0, ip1 in prange(0, ngrids, blksize):
            coords = grids.coords[ip0:ip1]
            weight = grids.weights[ip0:ip1]
            non0tab = self.non0tab[ip0//BLKSIZE:]
            ao = eval_ao(mol, coords, isgga=isgga, non0tab=non0tab)
            for idm in range(nset):
                rho = eval_rho2(mol, ao, natorb[idm], natocc[idm],
                                non0tab=non0tab, isgga=isgga)
                if isgga:
                    sigma = numpy.einsum('ip,ip->p', rho[1:], rho[1:])
                    exc, vrho, vsigma = eval_xc(x_id, c_id, rho[0], sigma,
                                                spin=0, verbose=verbose)
                    den = rho[0]*weight
                    nelec[idm] += den.sum()
                    excsum[idm] += (den*exc).sum()
# ref eval_mat function
                    wv = numpy.empty_like(rho)
                    wv[0]  = weight * vrho * .5
                    wv[1:] = rho[1:] * (weight * vsigma * 2)
                    aow = numpy.einsum('npi,np->pi', ao, wv)
                    vmat[idm] += _dot_ao_ao(mol, ao[0], aow, nao, ip1-ip0,
                                            non0tab)
                else:
                    exc, vrho, vsigma = eval_xc(x_id, c_id, rho, rho,
                                                spin=0, verbose=verbose)
                    den = rho*weight
                    nelec[idm] += den.sum()
                    excsum[idm] += (den*exc).sum()
                    aow = ao * (.5*weight*vrho).reshape(-1,1)
                    vmat[idm] += _dot_ao_ao(mol, ao, aow, nao, ip1-ip0,
                                            non0tab)
            wv = aow = None
        for i in range(nset):
            vmat[i] = vmat[i] + vmat[i].T
        if nset == 1:
            nelec = nelec[0]
            excsum = excsum[0]
            vmat = vmat.reshape(nao,nao)
        return nelec, excsum, vmat

    def nr_uks(self, mol, grids, x_id, c_id, dms, hermi=1,
               max_memory=2000, verbose=None):
        if self.non0tab is None:
            self.non0tab = make_mask(mol, grids.coords)
        nao = mol.nao_nr()
        ngrids = len(grids.weights)
# NOTE to index self.non0tab, the blksize needs to be the integer multiplier of BLKSIZE
        blksize = min(int(max_memory/6*1e6/8/nao/BLKSIZE)*BLKSIZE, ngrids)
        if pyscf.dft.vxc._is_lda(x_id) and pyscf.dft.vxc._is_lda(c_id):
            isgga = False
        else:
            isgga = True

        natocc = []
        natorb = []
        if isinstance(dms, numpy.ndarray) and dms.ndim == 2:
            e, c = scipy.linalg.eigh(dms)
            natocc.append((e*.5,e*.5))
            natorb.append((c,c))
            nset = 1
        else:
            nset = len(dms) // 2
            for idm in range(nset):
                e_a, c_a = scipy.linalg.eigh(dms[idm])
                e_b, c_b = scipy.linalg.eigh(dms[nset+idm])
                natocc.append((e_a,e_b))
                natorb.append((c_a,c_b))
        nelec = numpy.zeros((2,nset))
        excsum = numpy.zeros(nset)
        vmat = numpy.zeros((2,nset,nao,nao))
        for ip0, ip1 in prange(0, ngrids, blksize):
            coords = grids.coords[ip0:ip1]
            weight = grids.weights[ip0:ip1]
            non0tab = self.non0tab[ip0//BLKSIZE:]
            ao = eval_ao(mol, coords, isgga=isgga, non0tab=non0tab)
            for idm in range(nset):
                c_a, c_b = natorb[idm]
                e_a, e_b = natocc[idm]
                rho_a = eval_rho2(mol, ao, c_a, e_a, non0tab=non0tab, isgga=isgga)
                rho_b = eval_rho2(mol, ao, c_b, e_b, non0tab=non0tab, isgga=isgga)
                if isgga:
                    rho = numpy.hstack((rho_a[0].reshape(-1,1),
                                        rho_b[0].reshape(-1,1)))
                    sigma = numpy.empty((ip1-ip0,3))
                    sigma[:,0] = numpy.einsum('ip,ip->p', rho_a[1:], rho_a[1:])
                    sigma[:,1] = numpy.einsum('ip,ip->p', rho_a[1:], rho_b[1:])
                    sigma[:,2] = numpy.einsum('ip,ip->p', rho_b[1:], rho_b[1:])
                    exc, vrho, vsigma = eval_xc(x_id, c_id, rho, sigma,
                                                spin=1, verbose=verbose)
                    den = rho[:,0]*weight
                    nelec[0,idm] += den.sum()
                    excsum[idm] += (den*exc).sum()
                    den = rho[:,1]*weight
                    nelec[1,idm] += den.sum()
                    excsum[idm] += (den*exc).sum()

                    wv = numpy.empty_like(rho_a)
                    wv[0]  = weight * vrho[:,0] * .5
                    wv[1:] = rho_a[1:] * (weight * vsigma[:,0] * 2)  # sigma_uu
                    wv[1:]+= rho_b[1:] * (weight * vsigma[:,1])      # sigma_ud
                    aow = numpy.einsum('npi,np->pi', ao, wv)
                    vmat[0,idm] += _dot_ao_ao(mol, ao[0], aow, nao, ip1-ip0,
                                              non0tab)
                    wv[0]  = weight * vrho[:,1] * .5
                    wv[1:] = rho_b[1:] * (weight * vsigma[:,2] * 2)  # sigma_dd
                    wv[1:]+= rho_a[1:] * (weight * vsigma[:,1])      # sigma_ud
                    aow = numpy.einsum('npi,np->pi', ao, wv)
                    vmat[1,idm] += _dot_ao_ao(mol, ao[0], aow, nao, ip1-ip0,
                                              non0tab)

                else:
                    rho = numpy.hstack((rho_a[:,None],rho_b[:,None]))
                    exc, vrho, vsigma = eval_xc(x_id, c_id, rho, rho,
                                                spin=1, verbose=verbose)
                    den = rho[:,0]*weight
                    nelec[0,idm] += den.sum()
                    excsum[idm] += (den*exc).sum()
                    den = rho[:,1]*weight
                    nelec[1,idm] += den.sum()
                    excsum[idm] += (den*exc).sum()

                    aow = ao * (.5*weight*vrho[:,0]).reshape(-1,1)
                    vmat[0,idm] += _dot_ao_ao(mol, ao, aow, nao, ip1-ip0,
                                              non0tab)
                    aow = ao * (.5*weight*vrho[:,1]).reshape(-1,1)
                    vmat[1,idm] += _dot_ao_ao(mol, ao, aow, nao, ip1-ip0,
                                              non0tab)
            wv = aow = None
        for i in range(nset):
            vmat[0,i] = vmat[0,i] + vmat[0,i].T
            vmat[1,i] = vmat[1,i] + vmat[1,i].T
        if nset == 1:
            nelec = nelec.reshape(2)
            excsum = excsum[0]
            vmat = vmat.reshape(2,nao,nao)
        return nelec, excsum, vmat

def prange(start, end, step):
    for i in range(start, end, step):
        yield i, min(i+step, end)


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import dft

    mol = gto.M(
        atom = [
        ["O" , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)] ],
        grids = {"H": (100, 194),
                 "O": (100, 194),},
        basis = '6311g*',)
    mf = dft.RKS(mol)
    mf.grids.setup_grids()
    dm = mf.get_init_guess(key='minao')

    x_code, c_code = pyscf.dft.vxc.parse_xc_name(mf.xc)
#res = vxc.nr_vxc(mol, mf.grids, x_code, c_code, dm, spin=1, relativity=0)
    print(time.clock())
    res = nr_vxc(mol, mf.grids, x_code, c_code, dm, spin=mol.spin, relativity=0)
    print(time.clock())

