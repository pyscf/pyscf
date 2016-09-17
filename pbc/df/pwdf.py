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
from pyscf import dft
from pyscf.lib import logger
from pyscf.pbc import tools
from pyscf.pbc.gto import pseudo
from pyscf.pbc.df import ft_ao
from pyscf.pbc.df import pwdf_jk
from pyscf.pbc.df import pwdf_ao2mo

KPT_DIFF_TOL = 1e-6


def get_nuc(mydf, kpts=None):
    log = logger.Logger(mydf.stdout, mydf.verbose)
    t1 = t0 = (time.clock(), time.time())
    cell = mydf.cell
    if kpts is None:
        kpts_lst = numpy.zeros((1,3))
    else:
        kpts_lst = numpy.reshape(kpts, (-1,3))
    nkpts = len(kpts_lst)

    nao = cell.nao_nr()
    charge = -cell.atom_charges()
    Gv = cell.get_Gv(mydf.gs)
    SI = cell.get_SI(Gv)
    rhoG = numpy.dot(charge, SI)
    kpt_allow = numpy.zeros(3)
    real = gamma_point(kpts_lst)
    coulG = tools.get_coulG(cell, kpt_allow, gs=mydf.gs, Gv=Gv) / cell.vol
    vneG = rhoG * coulG

    if real:
        vne = numpy.zeros((nkpts,nao**2))
    else:
        vne = numpy.zeros((nkpts,nao**2), dtype=numpy.complex128)
    max_memory = mydf.max_memory - lib.current_memory()[0]
    for k, pqkR, pqkI, p0, p1 \
            in mydf.ft_loop(cell, mydf.gs, kpt_allow, kpts_lst,
                            max_memory=max_memory):
# rho_ij(G) nuc(-G) / G^2
# = [Re(rho_ij(G)) + Im(rho_ij(G))*1j] [Re(nuc(G)) - Im(nuc(G))*1j] / G^2
        vG = vneG[p0:p1]
        if not real:
            vne[k] += numpy.einsum('k,xk->x', vG.real, pqkI) * 1j
            vne[k] += numpy.einsum('k,xk->x', vG.imag, pqkR) *-1j
        vne[k] += numpy.einsum('k,xk->x', vG.real, pqkR)
        vne[k] += numpy.einsum('k,xk->x', vG.imag, pqkI)
    vne = vne.reshape(-1,nao,nao)
    t1 = log.timer_debug1('contracting Vnuc', *t1)

    if kpts is None or numpy.shape(kpts) == (3,):
        vne = vne[0]
    return vne

def get_pp(mydf, kpts=None):
    '''Get the periodic pseudotential nuc-el AO matrix, with G=0 removed.
    '''
    cell = mydf.cell
    if kpts is None:
        kpts_lst = numpy.zeros((1,3))
    else:
        kpts_lst = numpy.reshape(kpts, (-1,3))
    nkpts = len(kpts_lst)

    vloc1 = get_pp_loc_part1(mydf, cell, kpts_lst)
    vloc2 = pseudo.pp_int.get_pp_loc_part2(cell, kpts_lst)
    vpp = pseudo.pp_int.get_pp_nl(cell, kpts_lst)
    for k in range(nkpts):
        vpp[k] += vloc1[k] + vloc2[k]

    if kpts is None or numpy.shape(kpts) == (3,):
        vpp = vpp[0]
    return vpp

def get_pp_loc_part1(mydf, cell, kpts):
    log = logger.Logger(mydf.stdout, mydf.verbose)
    t1 = t0 = (time.clock(), time.time())
    nkpts = len(kpts)

    gs = mydf.gs
    nao = cell.nao_nr()
    Gv = cell.get_Gv(gs)
    SI = cell.get_SI(Gv)
    vpplocG = pseudo.pp_int.get_gth_vlocG_part1(cell, Gv)
    vpplocG = -1./cell.vol * numpy.einsum('ij,ij->j', SI, vpplocG)
    kpt_allow = numpy.zeros(3)
    real = gamma_point(kpts)

    if real:
        vloc = numpy.zeros((nkpts,nao**2))
    else:
        vloc = numpy.zeros((nkpts,nao**2), dtype=numpy.complex128)
    max_memory = mydf.max_memory - lib.current_memory()[0]
    for k, pqkR, pqkI, p0, p1 \
            in mydf.ft_loop(cell, mydf.gs, kpt_allow, kpts, max_memory=max_memory):
        vG = vpplocG[p0:p1]
        if not real:
            vloc[k] += numpy.einsum('k,xk->x', vG.real, pqkI) * 1j
            vloc[k] += numpy.einsum('k,xk->x', vG.imag, pqkR) *-1j
        vloc[k] += numpy.einsum('k,xk->x', vG.real, pqkR)
        vloc[k] += numpy.einsum('k,xk->x', vG.imag, pqkI)
    t1 = log.timer_debug1('contracting vloc part1', *t1)
    return vloc.reshape(-1,nao,nao)


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

    def pw_loop(self, cell, gs=None, kpti_kptj=None, shls_slice=None,
                max_memory=2000):
        '''Plane wave part'''
        if gs is None:
            gs = self.gs
        if kpti_kptj is None:
            kpti = kptj = numpy.zeros(3)
        else:
            kpti, kptj = kpti_kptj

        nao = cell.nao_nr()
        gxyz = lib.cartesian_prod((numpy.append(range(gs[0]+1), range(-gs[0],0)),
                                   numpy.append(range(gs[1]+1), range(-gs[1],0)),
                                   numpy.append(range(gs[2]+1), range(-gs[2],0))))
        invh = numpy.linalg.inv(cell._h)
        Gv = 2*numpy.pi * numpy.dot(gxyz, invh)
        ngs = gxyz.shape[0]

# Theoretically, hermitian symmetry can be also found for kpti == kptj:
#       f_ji(G) = \int f_ji exp(-iGr) = \int f_ij^* exp(-iGr) = [f_ij(-G)]^*
# The hermi operation needs reordering the axis-0.  It is inefficient
        if gamma_point(kpti) and gamma_point(kptj):
            aosym = 's1hermi'
        else:
            aosym = 's1'

        blksize = min(max(16, int(max_memory*1e6*.75/16/nao**2)), 16384)
        sublk = max(16, int(blksize//4))
        buf = [numpy.zeros(nao*nao*blksize, dtype=numpy.complex128)]
        pqkRbuf = numpy.empty(nao*nao*sublk)
        pqkIbuf = numpy.empty(nao*nao*sublk)

        for p0, p1 in self.prange(0, ngs, blksize):
            #aoao = ft_ao.ft_aopair(cell, Gv[p0:p1], shls_slice, aosym, invh,
            #                       gxyz[p0:p1], gs, (kpti, kptj))
            aoao = ft_ao._ft_aopair_kpts(cell, Gv[p0:p1], shls_slice, aosym,
                                         invh, gxyz[p0:p1], gs,
                                         kptj-kpti, kptj.reshape(1,3), out=buf)[0]
            for i0, i1 in lib.prange(0, p1-p0, sublk):
                nG = i1 - i0
                pqkR = numpy.ndarray((nao,nao,nG), buffer=pqkRbuf)
                pqkI = numpy.ndarray((nao,nao,nG), buffer=pqkIbuf)
                pqkR[:] = aoao[i0:i1].real.transpose(1,2,0)
                pqkI[:] = aoao[i0:i1].imag.transpose(1,2,0)
                yield (pqkR.reshape(-1,nG),
                       pqkI.reshape(-1,nG), p0+i0, p0+i1)
            aoao[:] = 0

    def ft_loop(self, cell, gs=None, kpt=numpy.zeros(3),
                kpts=None, shls_slice=None, max_memory=4000):
        '''
        Fourier transform iterator for all kpti which satisfy  kpt = kpts - kpti
        '''
        if gs is None: gs = self.gs
        if kpts is None:
            assert(gamma_point(kpt))
            kpts = self.kpts
        kpts = numpy.asarray(kpts)
        nkpts = len(kpts)

        nao = cell.nao_nr()
        gxyz = lib.cartesian_prod((numpy.append(range(gs[0]+1), range(-gs[0],0)),
                                   numpy.append(range(gs[1]+1), range(-gs[1],0)),
                                   numpy.append(range(gs[2]+1), range(-gs[2],0))))
        invh = numpy.linalg.inv(cell._h)
        Gv = 2*numpy.pi * numpy.dot(gxyz, invh)
        ngs = gxyz.shape[0]

# Theoretically, hermitian symmetry can be also found for kpti == kptj:
#       f_ji(G) = \int f_ji exp(-iGr) = \int f_ij^* exp(-iGr) = [f_ij(-G)]^*
# The hermi operation needs reordering the axis-0.  It is inefficient
        if gamma_point(kpt) and gamma_point(kpts):
            aosym = 's1hermi'
        else:
            aosym = 's1'

        blksize = min(max(16, int(max_memory*.9e6/(nao**2*(nkpts+1)*16))), 16384)
        buf = [numpy.zeros(nao*nao*blksize, dtype=numpy.complex128)
               for k in range(nkpts)]
        pqkRbuf = numpy.empty(nao*nao*blksize)
        pqkIbuf = numpy.empty(nao*nao*blksize)

        for p0, p1 in self.prange(0, ngs, blksize):
            ft_ao._ft_aopair_kpts(cell, Gv[p0:p1], shls_slice, aosym, invh,
                                  gxyz[p0:p1], gs, kpt, kpts, out=buf)
            nG = p1 - p0
            for k in range(nkpts):
                aoao = numpy.ndarray((nG,nao,nao), dtype=numpy.complex128,
                                     order='F', buffer=buf[k])
                pqkR = numpy.ndarray((nao,nao,nG), buffer=pqkRbuf)
                pqkI = numpy.ndarray((nao,nao,nG), buffer=pqkIbuf)
                pqkR[:] = aoao.real.transpose(1,2,0)
                pqkI[:] = aoao.imag.transpose(1,2,0)
                yield (k, pqkR.reshape(-1,nG), pqkI.reshape(-1,nG), p0, p1)
                aoao[:] = 0  # == buf[k][:] = 0

    def prange(self, start, stop, step):
        return lib.prange(start, stop, step)

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
    cell.h = numpy.diag([4, 4, 4])
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.gs = [10, 10, 10]
    cell.build()
    k = numpy.ones(3)*.25
    df = PWDF(cell)
    v1 = get_pp(df, k)

