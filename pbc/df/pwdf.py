#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''Density expansion on plane waves'''

import time
import numpy
import copy
from pyscf import lib
from pyscf import gto
from pyscf.lib import logger
from pyscf.pbc import tools
from pyscf.pbc.gto import pseudo
from pyscf.pbc.df import ft_ao
from pyscf.pbc.df import pwdf_jk
from pyscf.pbc.df import pwdf_ao2mo

KPT_DIFF_TOL = 1e-6


def get_nuc(mydf, kpts=None):
    cell = mydf.cell
    log = logger.Logger(mydf.stdout, mydf.verbose)
    t1 = t0 = (time.clock(), time.time())
    nkpts = len(kpts)

    nao = cell.nao_nr()
    Gv, Gvbase, kws = cell.get_Gv_weights(mydf.gs)
    vpplocG = pseudo.pp_int.get_gth_vlocG_part1(cell, Gv)
    vpplocG = -numpy.einsum('ij,ij->j', cell.get_SI(Gv), vpplocG)
    vpplocG *= kws
    kpt_allow = numpy.zeros(3)
    real = gamma_point(kpts)

    if real:
        vne = numpy.zeros((nkpts,nao**2))
    else:
        vne = numpy.zeros((nkpts,nao**2), dtype=numpy.complex128)
    max_memory = mydf.max_memory - lib.current_memory()[0]
    for k, pqkR, pqkI, p0, p1 \
            in mydf.ft_loop(mydf.gs, kpt_allow, kpts, max_memory=max_memory):
        vG = vpplocG[p0:p1]
        if not real:
            vne[k] += numpy.einsum('k,xk->x', vG.real, pqkI) * 1j
            vne[k] += numpy.einsum('k,xk->x', vG.imag, pqkR) *-1j
        vne[k] += numpy.einsum('k,xk->x', vG.real, pqkR)
        vne[k] += numpy.einsum('k,xk->x', vG.imag, pqkI)
        pqkR = pqkI = None
    t1 = log.timer_debug1('contracting Vnuc', *t1)

    vne = vne.reshape(nkpts,nao,nao)
    if kpts is None or numpy.shape(kpts) == (3,):
        vne = vne[0]
    return vne
get_pp_loc_part1 = get_nuc


def get_pp(mydf, kpts=None):
    '''Get the periodic pseudotential nuc-el AO matrix, with G=0 removed.
    '''
    cell = mydf.cell
    if kpts is None:
        kpts_lst = numpy.zeros((1,3))
    else:
        kpts_lst = numpy.reshape(kpts, (-1,3))
    nkpts = len(kpts_lst)

    vloc1 = mydf.get_nuc(kpts_lst)
    vloc2 = pseudo.pp_int.get_pp_loc_part2(cell, kpts_lst)
    vpp = pseudo.pp_int.get_pp_nl(cell, kpts_lst)
    for k in range(nkpts):
        vpp[k] += vloc1[k] + vloc2[k]

    if kpts is None or numpy.shape(kpts) == (3,):
        vpp = vpp[0]
    return vpp


class PWDF(lib.StreamObject):
    '''Density expansion on plane waves
    '''
    def __init__(self, cell, kpts=numpy.zeros((1,3))):
        self.cell = cell
        self.stdout = cell.stdout
        self.verbose = cell.verbose
        self.max_memory = cell.max_memory

        self.kpts = kpts
        self.gs = cell.gs

# Not input options
        self.exxdiv = None  # to mimic KRHF/KUHF object in function get_coulG
        self._keys = set(self.__dict__.keys())

    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)
        logger.info(self, '\n')
        logger.info(self, '******** %s flags ********', self.__class__)
        logger.info(self, 'gs = %s', self.gs)
        logger.info(self, 'len(kpts) = %d', len(self.kpts))
        logger.debug1(self, '    kpts = %s', self.kpts)

    def pw_loop(self, gs=None, kpti_kptj=None, shls_slice=None,
                max_memory=2000, aosym='s1'):
        '''Plane wave part'''
        cell = self.cell
        if gs is None:
            gs = self.gs
        if kpti_kptj is None:
            kpti = kptj = numpy.zeros(3)
        else:
            kpti, kptj = kpti_kptj

        ao_loc = cell.ao_loc_nr()
        Gv, Gvbase, kws = cell.get_Gv_weights(gs)
        b = cell.reciprocal_vectors()
        gxyz = lib.cartesian_prod([numpy.arange(len(x)) for x in Gvbase])
        ngs = gxyz.shape[0]

        if shls_slice is None:
            shls_slice = (0, cell.nbas, 0, cell.nbas)
        if aosym == 's2':
            assert(shls_slice[2] == 0)
            i0 = ao_loc[shls_slice[0]]
            i1 = ao_loc[shls_slice[1]]
            nij = i1*(i1+1)//2 - i0*(i0+1)//2
        else:
            ni = ao_loc[shls_slice[1]] - ao_loc[shls_slice[0]]
            nj = ao_loc[shls_slice[3]] - ao_loc[shls_slice[2]]
            nij = ni*nj
        blksize = min(max(16, int(max_memory*1e6*.75/16/nij)), 16384)
        sublk = max(16, int(blksize//4))
        buf = [numpy.zeros(nij*blksize, dtype=numpy.complex128)]
        pqkRbuf = numpy.empty(nij*sublk)
        pqkIbuf = numpy.empty(nij*sublk)

        if aosym == 's2':
            for p0, p1 in self.prange(0, ngs, blksize):
                aoao = ft_ao._ft_aopair_kpts(cell, Gv[p0:p1], shls_slice, aosym,
                                             b, gxyz[p0:p1], Gvbase, kptj-kpti,
                                             kptj.reshape(1,3), out=buf)[0]
                for i0, i1 in lib.prange(0, p1-p0, sublk):
                    nG = i1 - i0
                    pqkR = numpy.ndarray((nij,nG), buffer=pqkRbuf)
                    pqkI = numpy.ndarray((nij,nG), buffer=pqkIbuf)
                    pqkR[:] = aoao[i0:i1].T
                    pqkI[:] = aoao[i0:i1].T
                    yield (pqkR, pqkI, p0+i0, p0+i1)
                aoao[:] = 0
        else:
            for p0, p1 in self.prange(0, ngs, blksize):
                #aoao = ft_ao.ft_aopair(cell, Gv[p0:p1], shls_slice, aosym,
                #                       b, Gvbase, gxyz[p0:p1], gs, (kpti, kptj))
                aoao = ft_ao._ft_aopair_kpts(cell, Gv[p0:p1], shls_slice, aosym,
                                             b, gxyz[p0:p1], Gvbase, kptj-kpti,
                                             kptj.reshape(1,3), out=buf)[0]
                for i0, i1 in lib.prange(0, p1-p0, sublk):
                    nG = i1 - i0
                    pqkR = numpy.ndarray((ni,nj,nG), buffer=pqkRbuf)
                    pqkI = numpy.ndarray((ni,nj,nG), buffer=pqkIbuf)
                    pqkR[:] = aoao[i0:i1].real.transpose(1,2,0)
                    pqkI[:] = aoao[i0:i1].imag.transpose(1,2,0)
                    yield (pqkR.reshape(-1,nG), pqkI.reshape(-1,nG), p0+i0, p0+i1)
                aoao[:] = 0

    def ft_loop(self, gs=None, kpt=numpy.zeros(3), kpts=None, shls_slice=None,
                max_memory=4000, aosym='s1'):
        '''
        Fourier transform iterator for all kpti which satisfy  kpt = kpts - kpti
        '''
        cell = self.cell
        if gs is None:
            gs = self.gs
        if kpts is None:
            assert(gamma_point(kpt))
            kpts = self.kpts
        kpts = numpy.asarray(kpts)
        nkpts = len(kpts)

        ao_loc = cell.ao_loc_nr()
        b = cell.reciprocal_vectors()
        Gv, Gvbase, kws = cell.get_Gv_weights(gs)
        gxyz = lib.cartesian_prod([numpy.arange(len(x)) for x in Gvbase])
        ngs = gxyz.shape[0]

        if shls_slice is None:
            shls_slice = (0, cell.nbas, 0, cell.nbas)
        if aosym == 's2':
            assert(shls_slice[2] == 0)
            i0 = ao_loc[shls_slice[0]]
            i1 = ao_loc[shls_slice[1]]
            nij = i1*(i1+1)//2 - i0*(i0+1)//2
        else:
            ni = ao_loc[shls_slice[1]] - ao_loc[shls_slice[0]]
            nj = ao_loc[shls_slice[3]] - ao_loc[shls_slice[2]]
            nij = ni*nj
        blksize = max(16, int(max_memory*.9e6/(nij*(nkpts+1)*16)))
        blksize = min(blksize, ngs, 16384)
        buf = [numpy.zeros(nij*blksize, dtype='D') for k in range(nkpts)]
        pqkRbuf = numpy.empty(nij*blksize)
        pqkIbuf = numpy.empty(nij*blksize)

        if aosym == 's2':
            for p0, p1 in self.prange(0, ngs, blksize):
                ft_ao._ft_aopair_kpts(cell, Gv[p0:p1], shls_slice, aosym,
                                      b, gxyz[p0:p1], Gvbase, kpt, kpts, out=buf)
                nG = p1 - p0
                for k in range(nkpts):
                    aoao = numpy.ndarray((nG,nij), dtype=numpy.complex128,
                                         order='F', buffer=buf[k])
                    pqkR = numpy.ndarray((nij,nG), buffer=pqkRbuf)
                    pqkI = numpy.ndarray((nij,nG), buffer=pqkIbuf)
                    pqkR[:] = aoao.real.T
                    pqkI[:] = aoao.imag.T
                    yield (k, pqkR, pqkI, p0, p1)
                    aoao[:] = 0  # == buf[k][:] = 0
        else:
            for p0, p1 in self.prange(0, ngs, blksize):
                ft_ao._ft_aopair_kpts(cell, Gv[p0:p1], shls_slice, aosym,
                                      b, gxyz[p0:p1], Gvbase, kpt, kpts, out=buf)
                nG = p1 - p0
                for k in range(nkpts):
                    aoao = numpy.ndarray((nG,ni,nj), dtype=numpy.complex128,
                                         order='F', buffer=buf[k])
                    pqkR = numpy.ndarray((ni,nj,nG), buffer=pqkRbuf)
                    pqkI = numpy.ndarray((ni,nj,nG), buffer=pqkIbuf)
                    pqkR[:] = aoao.real.transpose(1,2,0)
                    pqkI[:] = aoao.imag.transpose(1,2,0)
                    yield (k, pqkR.reshape(-1,nG), pqkI.reshape(-1,nG), p0, p1)
                    aoao[:] = 0  # == buf[k][:] = 0

    def prange(self, start, stop, step):
        return lib.prange(start, stop, step)

    def weighted_coulG(self, kpt=numpy.zeros(3), exx=False, gs=None):
        cell = self.cell
        if gs is None:
            gs = self.gs
        Gv, Gvbase, kws = cell.get_Gv_weights(gs)
        coulG = tools.get_coulG(cell, kpt, exx, self, gs, Gv)
        coulG *= kws
        return coulG

    get_pp = get_pp
    get_nuc = get_nuc

    def get_jk(self, dm, hermi=1, kpts=None, kpt_band=None,
               with_j=True, with_k=True, exxdiv='ewald'):
        if kpts is None:
            if numpy.all(self.kpts == 0):
                # Gamma-point calculation by default
                kpts = numpy.zeros(3)
            else:
                kpts = self.kpts

        if kpts.shape == (3,):
            return pwdf_jk.get_jk(self, dm, hermi, kpts, kpt_band, with_j,
                                  with_k, exxdiv)

        vj = vk = None
        if with_k:
            vk = pwdf_jk.get_k_kpts(self, dm, hermi, kpts, kpt_band, exxdiv)
        if with_j:
            vj = pwdf_jk.get_j_kpts(self, dm, hermi, kpts, kpt_band)
        return vj, vk

    get_eri = get_ao_eri = pwdf_ao2mo.get_eri
    ao2mo = get_mo_eri = pwdf_ao2mo.general

    def update_mf(self, mf):
        mf = copy.copy(mf)
        mf.with_df = self
        return mf

def gamma_point(kpt):
    return abs(kpt).sum() < KPT_DIFF_TOL


if __name__ == '__main__':
    from pyscf.pbc import gto as pbcgto
    import pyscf.pbc.scf.hf as phf
    cell = pbcgto.Cell()
    cell.verbose = 0
    cell.atom = 'C 0 0 0; C 1 1 1; C 0 2 2; C 2 0 2'
    cell.a = numpy.diag([4, 4, 4])
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.gs = [10, 10, 10]
    cell.build()
    k = numpy.ones(3)*.25
    df = PWDF(cell)
    v1 = get_pp(df, k)

