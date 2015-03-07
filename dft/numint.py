#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import os
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
    c_atm = numpy.array(mol._atm, dtype=numpy.int32)
    c_bas = numpy.array(mol._bas, dtype=numpy.int32)
    c_env = numpy.array(mol._env)
    natm = ctypes.c_int(c_atm.shape[0])
    nbas = ctypes.c_int(c_bas.shape[0])
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
        non0tab = numpy.empty(((ngrids+BLKSIZE-1)//BLKSIZE,nbas.value),
                              dtype=numpy.int8)
    libdft.VXCnr_ao_screen(non0tab.ctypes.data_as(ctypes.c_void_p),
                           coords.ctypes.data_as(ctypes.c_void_p),
                           ctypes.c_int(ngrids), ctypes.c_int(BLKSIZE),
                           c_atm.ctypes.data_as(ctypes.c_void_p), natm,
                           c_bas.ctypes.data_as(ctypes.c_void_p), nbas,
                           c_env.ctypes.data_as(ctypes.c_void_p))

    libdft.VXCeval_ao_drv(ctypes.c_void_p(feval),
                          ctypes.c_int(nao), ctypes.c_int(ngrids),
                          ctypes.c_int(bastart), ctypes.c_int(bascount),
                          ctypes.c_int(BLKSIZE),
                          ao.ctypes.data_as(ctypes.c_void_p),
                          coords.ctypes.data_as(ctypes.c_void_p),
                          non0tab.ctypes.data_as(ctypes.c_void_p),
                          c_atm.ctypes.data_as(ctypes.c_void_p), natm,
                          c_bas.ctypes.data_as(ctypes.c_void_p), nbas,
                          c_env.ctypes.data_as(ctypes.c_void_p))
    return ao, non0tab

def eval_rho(mol, ao, dm, non0tab=None, isgga=False, verbose=None):
    c_atm = numpy.array(mol._atm, dtype=numpy.int32)
    c_bas = numpy.array(mol._bas, dtype=numpy.int32)
    c_env = numpy.array(mol._env)
    natm = ctypes.c_int(c_atm.shape[0])
    nbas = ctypes.c_int(c_bas.shape[0])

    if isgga:
        ngrids, nao = ao[0].shape
    else:
        ngrids, nao = ao.shape

    if non0tab is None:
        non0tab = numpy.ones(((ngrids+BLKSIZE-1)/BLKSIZE,nbas.value),
                             dtype=numpy.int8)
    def adot(ao, dm):
        #return pyscf.lib.dot(ao, dm)
        vm = numpy.empty((ngrids,dm.shape[1]))
        libdft.VXCdot_ao_dm(vm.ctypes.data_as(ctypes.c_void_p),
                            ao.ctypes.data_as(ctypes.c_void_p),
                            dm.ctypes.data_as(ctypes.c_void_p),
                            ctypes.c_int(nao), ctypes.c_int(dm.shape[1]),
                            ctypes.c_int(ngrids), ctypes.c_int(BLKSIZE),
                            non0tab.ctypes.data_as(ctypes.c_void_p),
                            c_atm.ctypes.data_as(ctypes.c_void_p), natm,
                            c_bas.ctypes.data_as(ctypes.c_void_p), nbas,
                            c_env.ctypes.data_as(ctypes.c_void_p))
        return vm

    def frho(dm):
        e, c = scipy.linalg.eigh(dm)
        pos = e > OCCDROP
        cpos = numpy.einsum('ij,j->ij', c[:,pos], numpy.sqrt(e[pos]))
        if isgga:
            rho = numpy.empty((4,ngrids))
            c0 = adot(ao[0], cpos)
            rho[0] = numpy.einsum('pi,pi->p', c0, c0)
            for i in range(1, 4):
                c1 = adot(ao[i], cpos)
                rho[i] = numpy.einsum('pi,pi->p', c0, c1) * 2 # *2 for +c.c.
        else:
            c0 = adot(ao, cpos)
            rho = numpy.einsum('pi,pi->p', c0, c0)

        neg = e < -OCCDROP
        if neg.sum() > 0:
            cneg = numpy.einsum('ij,j->ij', c[:,neg], numpy.sqrt(-e[neg]))
            if isgga:
                c0 = adot(ao[0], cneg)
                rho[0] -= numpy.einsum('pi,pi->p', c0, c0)
                for i in range(1, 4):
                    c1 = adot(ao[i], cpos)
                    rho[i] -= numpy.einsum('pi,pi->p', c0, c1) * 2 # *2 for +c.c.
            else:
                c0 = adot(ao, cneg)
                rho -= numpy.einsum('pi,pi->p', c0, c0)
        return rho

    if isinstance(dm, numpy.ndarray) and dm.ndim == 2:
        return frho(dm)
    else:
        return numpy.array([frho(x) for x in dm])

def eval_mat(mol, ao, weight, rho, vrho, vsigma=None, non0tab=None,
             isgga=False, verbose=None):
    c_atm = numpy.array(mol._atm, dtype=numpy.int32)
    c_bas = numpy.array(mol._bas, dtype=numpy.int32)
    c_env = numpy.array(mol._env)
    natm = ctypes.c_int(c_atm.shape[0])
    nbas = ctypes.c_int(c_bas.shape[0])

    if isgga:
        ngrids, nao = ao[0].shape
    else:
        ngrids, nao = ao.shape

    if non0tab is None:
        non0tab = numpy.ones(((ngrids+BLKSIZE-1)/BLKSIZE,nbas.value),
                             dtype=numpy.int8)

    def adot(ao1, ao2):
        #return pyscf.lib.dot(ao, dm)
        vv = numpy.empty((nao,nao))
        libdft.VXCdot_ao_ao(vv.ctypes.data_as(ctypes.c_void_p),
                            ao1.ctypes.data_as(ctypes.c_void_p),
                            ao2.ctypes.data_as(ctypes.c_void_p),
                            ctypes.c_int(nao), ctypes.c_int(ngrids),
                            ctypes.c_int(BLKSIZE),
                            non0tab.ctypes.data_as(ctypes.c_void_p),
                            c_atm.ctypes.data_as(ctypes.c_void_p), natm,
                            c_bas.ctypes.data_as(ctypes.c_void_p), nbas,
                            c_env.ctypes.data_as(ctypes.c_void_p))
        return vv

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
        mat = adot(ao[0], aow)
    else:
        # *.5 because return mat + mat.T
        #:aow = numpy.einsum('pi,p->pi', ao, .5*weight*vrho)
        aow = ao * (.5*weight*vrho).reshape(-1,1)
        #mat = pyscf.lib.dot(ao.T, aow)
        mat = adot(ao, aow)
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
        vrho = numpy.empty((2,ngrids))
        vsigma = numpy.empty((3,ngrids))
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
        vrho = numpy.empty((2,ngrids))
        vsigma = numpy.empty((3,ngrids))
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
        vrho = numpy.empty((2,ngrids))
        vsigma = numpy.empty((3,ngrids))
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

def nr_vxc(mol, grids, x_id, c_id, dm, spin=0, relativity=0, hermi=1,
           max_memory=2000, verbose=None):
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
            ao, non0tab = eval_ao(mol, coords, isgga=isgga)
            rho = eval_rho(mol, ao, dm, non0tab, isgga=isgga)
            exc, vrho, vsigma = eval_xc(x_id, c_id, rho, rho,
                                        spin, relativity, verbose)
            den = rho*weight
        else:
            isgga = True
            ao, non0tab = eval_ao(mol, coords, isgga=isgga)
            rho = eval_rho(mol, ao, dm, non0tab, isgga=isgga)
            sigma = numpy.einsum('ip,ip->p', rho[1:], rho[1:])
            exc, vrho, vsigma = eval_xc(x_id, c_id, rho[0], sigma,
                                        spin, relativity, verbose)
            den = rho[0]*weight

        nelec += den.sum()
        excsum += (den*exc).sum()
        vmat += eval_mat(mol, ao, weight, rho, vrho, vsigma, isgga,
                         non0tab=non0tab, verbose=verbose)
    return nelec, excsum, vmat


class _NumInt:
    def __init__(self):
        self.non0tab = None

    def nr_vxc(self, mol, grids, x_id, c_id, dm, spin=0, relativity=0, hermi=1,
               max_memory=2000, verbose=None):
        nao = dm.shape[0]
        ngrids = len(grids.weights)
        blksize = min(int(max_memory/6*1e6/8/nao), ngrids)
        nelec = 0
        excsum = 0
        vmat = numpy.zeros_like(dm)
        for ip0, ip1 in prange(0, ngrids, blksize):
            coords = grids.coords[ip0:ip1]
            weight = grids.weights[ip0:ip1]
            if pyscf.dft.vxc._is_lda(x_id) and pyscf.dft.vxc._is_lda(c_id):
                isgga = False
                ao, self.non0tab = eval_ao(mol, coords, isgga=isgga,
                                           non0tab=self.non0tab)
                rho = eval_rho(mol, ao, dm, non0tab=self.non0tab, isgga=isgga)
                exc, vrho, vsigma = eval_xc(x_id, c_id, rho, rho,
                                            spin, relativity, verbose)
                den = rho*weight
            else:
                isgga = True
                ao, self.non0tab = eval_ao(mol, coords, isgga=isgga,
                                           non0tab=self.non0tab)
                rho = eval_rho(mol, ao, dm, non0tab=self.non0tab, isgga=isgga)
                sigma = numpy.einsum('ip,ip->p', rho[1:], rho[1:])
                exc, vrho, vsigma = eval_xc(x_id, c_id, rho[0], sigma,
                                            spin, relativity, verbose)
                den = rho[0]*weight

            nelec += den.sum()
            excsum += (den*exc).sum()
            vmat += eval_mat(mol, ao, weight, rho, vrho, vsigma, isgga=isgga,
                             non0tab=self.non0tab, verbose=verbose)
        return nelec, excsum, vmat

def prange(start, end, step):
    for i in range(start, end, step):
        yield i, min(i+step, end)


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import lib
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

