#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''Density expansion on plane waves'''

import time
import numpy
from pyscf import lib
from pyscf import gto
from pyscf import dft
from pyscf.lib import logger
from pyscf.pbc import tools
from pyscf.pbc.df import ft_ao
from pyscf.pbc import scf as pscf
from pyscf.pbc import dft as pdft
from pyscf.pbc.gto import pseudo
from pyscf.pbc.df import pwdf_jk
from pyscf.pbc.df import pwdf_ao2mo
from pyscf.pbc.df import pwfft_ao2mo


def get_nuc(pwdf, cell=None, kpts=None):
    log = logger.Logger(pwdf.stdout, pwdf.verbose)
    t1 = t0 = (time.clock(), time.time())
    if cell is not None:
        assert(id(cell) == id(pwdf.cell))
    cell = pwdf.cell
    if kpts is None:
        kpts_lst = numpy.zeros((1,3))
    else:
        kpts_lst = numpy.reshape(kpts, (-1,3))

    nao = cell.nao_nr()
    charge = -cell.atom_charges()
    Gv = cell.get_Gv(pwdf.gs)
    SI = cell.get_SI(Gv)
    rhoG = numpy.dot(charge, SI)
    max_memory = pwdf.max_memory - lib.current_memory()[0]

    vne = [0] * len(kpts_lst)
    for k, kpt in enumerate(kpts_lst):
        p0 = 0
        for pqkR, pqkI, coulG \
                in pwdf.pw_loop(cell, pwdf.gs, (kpt,kpt), max_memory):
# rho_ij(G) nuc(-G) / G^2
# = [Re(rho_ij(G)) + Im(rho_ij(G))*1j] [Re(nuc(G)) - Im(nuc(G))*1j] / G^2
            nG = len(coulG)
            vG = rhoG[p0:p0+nG] * coulG
            if abs(kpt).sum() > 1e-9:  # if not gamma point
                vne[k] += numpy.einsum('k,xk->x', vG.real, pqkI) * 1j
                vne[k] += numpy.einsum('k,xk->x', vG.imag, pqkR) *-1j
            vne[k] += numpy.einsum('k,xk->x', vG.real, pqkR)
            vne[k] += numpy.einsum('k,xk->x', vG.imag, pqkI)
            p0 += nG
    vne = [v.reshape(nao,nao) for v in vne]
    t1 = log.timer('contracting Vnuc', *t1)

    if kpts is None or numpy.shape(kpts) == (3,):
        vne = vne[0]
    return vne

def get_pp(pwdf, cell=None, kpts=None):
    '''Get the periodic pseudotential nuc-el AO matrix, with G=0 removed.
    '''
    log = logger.Logger(pwdf.stdout, pwdf.verbose)
    t1 = t0 = (time.clock(), time.time())
    if cell is not None:
        assert(id(cell) == id(pwdf.cell))
    cell = pwdf.cell
    if kpts is None:
        kpts_lst = numpy.zeros((1,3))
    else:
        kpts_lst = numpy.reshape(kpts, (-1,3))

    gs = cell.gs  # use xdf.gs?

    SI = cell.get_SI()
    vpplocG = -numpy.einsum('ij,ij->j', SI, pseudo.get_vlocG(cell)) / cell.vol
    ngs = len(vpplocG)
    nao = cell.nao_nr()
    max_memory = pwdf.max_memory - lib.current_memory()[0]

    vloc = [0] * len(kpts_lst)
    for k, kpt in enumerate(kpts_lst):
        p0 = 0
        for pqkR, pqkI, coulG \
                in pwdf.pw_loop(cell, gs, (kpt,kpt), max_memory):
# rho_ij(G) nuc(-G) / G^2
# = [Re(rho_ij(G)) + Im(rho_ij(G))*1j] [Re(nuc(G)) - Im(nuc(G))*1j] / G^2
            nG = len(coulG)
            vG = vpplocG[p0:p0+nG]
            if abs(kpt).sum() > 1e-9:  # if not gamma point
                vloc[k] += numpy.einsum('k,xk->x', vG.real, pqkI) * 1j
                vloc[k] += numpy.einsum('k,xk->x', vG.imag, pqkR) *-1j
            vloc[k] += numpy.einsum('k,xk->x', vG.real, pqkR)
            vloc[k] += numpy.einsum('k,xk->x', vG.imag, pqkI)
            p0 += nG

    # vppnonloc evaluated in reciprocal space
    fakemol = gto.Mole()
    fakemol._atm = numpy.zeros((1,gto.ATM_SLOTS), dtype=numpy.int32)
    fakemol._bas = numpy.zeros((1,gto.BAS_SLOTS), dtype=numpy.int32)
    ptr = gto.PTR_ENV_START
    fakemol._env = numpy.zeros(ptr+10)
    fakemol._bas[0,gto.NPRIM_OF ] = 1
    fakemol._bas[0,gto.NCTR_OF  ] = 1
    fakemol._bas[0,gto.PTR_EXP  ] = ptr+3
    fakemol._bas[0,gto.PTR_COEFF] = ptr+4

    def vppnl_by_k(k):
        Gv = cell.get_Gv(gs)
        Gk = Gv + kpts_lst[k]
        G_rad = lib.norm(Gk, axis=1)
        aokG = ft_ao.ft_ao(cell, Gv, kpt=kpts_lst[k]) * (ngs/cell.vol)
        vppnl = 0
        for ia in range(cell.natm):
            symb = cell.atom_symbol(ia)
            if symb not in cell._pseudo:
                continue
            pp = cell._pseudo[symb]
            for l, proj in enumerate(pp[5:]):
                rl, nl, hl = proj
                if nl > 0:
                    hl = numpy.asarray(hl)
                    fakemol._bas[0,gto.ANG_OF] = l
                    fakemol._env[ptr+3] = .5*rl**2
                    fakemol._env[ptr+4] = rl**(l+1.5)*numpy.pi**1.25
                    pYlm_part = dft.numint.eval_ao(fakemol, Gk, deriv=0)

                    pYlm = numpy.empty((nl,l*2+1,ngs))
                    for k in range(nl):
                        qkl = pseudo.pp._qli(G_rad*rl, l, k)
                        pYlm[k] = pYlm_part.T * qkl
                    # pYlm is real
                    SPG_lmi = numpy.einsum('g,nmg->nmg', SI[ia].conj(), pYlm)
                    for k, kpt in enumerate(kpts_lst):
                        SPG_lm_aoG = numpy.einsum('nmg,gp->nmp', SPG_lmi, aokG)
                        tmp = numpy.einsum('ij,jmp->imp', hl, SPG_lm_aoG)
                        vppnl += numpy.einsum('imp,imq->pq', SPG_lm_aoG.conj(), tmp)
        return vppnl * (1./ngs**2)

    vpp = []
    for k, kpt in enumerate(kpts_lst):
        vppnl = vppnl_by_k(k)
        if abs(kpt).sum() < 1e-9:  # gamma_point
            vpp.append(vloc[k].reshape(nao,nao).real + vppnl.real)
        else:
            vpp.append(vloc[k].reshape(nao,nao) + vppnl)

    if kpts is None or numpy.shape(kpts) == (3,):
        vpp = vpp[0]
    return vpp


class PWDF(lib.StreamObject):
    '''Density expansion on plane waves
    '''
    def __init__(self, cell):
        self.cell = cell
        self.stdout = cell.stdout
        self.verbose = cell.verbose
        self.max_memory = cell.max_memory

        self.kpts = numpy.zeros((1,3))  # default is gamma point
        self.gs = cell.gs
        self.analytic_ft = False
        self.exxdiv = 'ewald'

# Not input options
        self._ni = None
        self._keys = set(self.__dict__.keys())

    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)
        logger.info(self, '\n')
        logger.info(self, '******** %s flags ********', self.__class__)
        logger.info(self, 'gs = %s', self.gs)
        logger.info(self, 'analytic Fourier transformation = %s', self.analytic_ft)
        logger.info(self, 'exxdiv = %s', self.exxdiv)
        logger.info(self, 'len(kpts) = %d', len(self.kpts))
        logger.debug1(self, '    kpts = %s', self.kpts)

    def pw_loop(self, cell, gs=None, kpti_kptj=None, max_memory=2000):
        '''Plane wave part'''
        if gs is None:
            gs = self.gs
        if kpti_kptj is None:
            kpti = kptj = numpy.zeros(3)
        else:
            kpti, kptj = kpti_kptj

        nao = cell.nao_nr()
        gxrange = numpy.append(range(gs[0]+1), range(-gs[0],0))
        gyrange = numpy.append(range(gs[1]+1), range(-gs[1],0))
        gzrange = numpy.append(range(gs[2]+1), range(-gs[2],0))
        gxyz = lib.cartesian_prod((gxrange, gyrange, gzrange))
        invh = numpy.linalg.inv(cell._h)
        Gv = 2*numpy.pi * numpy.dot(gxyz, invh)
        ngs = gxyz.shape[0]
        coulG = tools.get_coulG(cell, kptj-kpti, gs=gs, Gv=Gv) / cell.vol

# Theoretically, hermitian symmetry can be also found for kpti == kptj:
#       f_ji(G) = \int f_ji exp(-iGr) = \int f_ij^* exp(-iGr) = [f_ij(-G)]^*
# The hermi operation needs reordering the axis-0.  It is inefficient
        hermi = abs(kpti).sum() < 1e-9 and abs(kptj).sum() < 1e-9  # gamma point

        blksize = min(max(16, int(max_memory*1e6*.7/16/nao**2)), 16384)
        sublk = max(16, int(blksize//8))
        pqkRbuf = numpy.empty(nao*nao*sublk)
        pqkIbuf = numpy.empty(nao*nao*sublk)

        for p0, p1 in lib.prange(0, ngs, blksize):
            aoao = ft_ao.ft_aopair(cell, Gv[p0:p1], None, hermi, invh, gxyz[p0:p1],
                                   gs, (kpti, kptj)).reshape(-1,nao**2)
            for i0, i1 in lib.prange(0, p1-p0, sublk):
                nG = i1 - i0
                pqkR = numpy.ndarray((nao*nao,nG), buffer=pqkRbuf)
                pqkI = numpy.ndarray((nao*nao,nG), buffer=pqkIbuf)
                pqkR[:] = aoao[i0:i1].real.T
                pqkI[:] = aoao[i0:i1].imag.T
                yield pqkR, pqkI, coulG[p0+i0:p0+i1]

    def aoR_loop(self, cell, gs=None, kpts=None, max_memory=2000):
        if gs is None: gs = self.gs
        if kpts is None: kpts = self.kpts
        if self._ni is None:
            self._ni = pdft.numint._KNumInt(self.kpts)
        coords = pdft.gen_grid.gen_uniform_grids(cell, gs)
        ngs = len(coords)
        nao = cell.nao_nr()
        blksize = min(max(16, int(max_memory*1e6*.7/16/nao**2)), 16384)
        buf = numpy.empty(nao*nao*blksize, dtype=numpy.complex)
        for p0, p1 in lib.prange(0, ngs, blksize):
            aoR_ks = self._ni.eval_ao(cell, coords[p0:p1], kpts, out=buf)
            yield aoR_ks

    def get_pp(self, cell=None, kpts=None):
        if self.analytic_ft:
            return get_pp(self, cell, kpts)
        else:
            if kpts is None:
                kpts_lst = numpy.zeros((1,3))
            else:
                kpts_lst = numpy.reshape(kpts, (-1,3))
            vne = [pscf.hf.get_pp(cell, k) for k in kpts_lst]
            if kpts is None or numpy.shape(kpts) == (3,):
                vne = vne[0]
            return vne

    def get_nuc(self, cell=None, kpts=None):
        if self.analytic_ft:
            return get_nuc(self, cell, kpts)
        else:
            if cell is not None:
                assert(id(cell) == id(self.cell))
            cell = self.cell
            if kpts is None:
                kpts_lst = numpy.zeros((1,3))
            else:
                kpts_lst = numpy.reshape(kpts, (-1,3))

            charge = -cell.atom_charges()
            Gv = cell.get_Gv(self.gs)
            SI = cell.get_SI(Gv)
            rhoG = numpy.dot(charge, SI)

            coulG = tools.get_coulG(cell, gs=self.gs, Gv=Gv)
            vneG = rhoG * coulG
            vneR = tools.ifft(vneG, self.gs).real

            max_memory = self.max_memory - lib.current_memory()[0]
            vne = [0] * len(kpts_lst)
            p0 = 0
            for aoR_ks in self.aoR_loop(cell, self.gs, kpts_lst, max_memory):
                nG = aoR_ks[0].shape[0]
                for k, aoR in enumerate(aoR_ks):
                    vne[k] += lib.dot(aoR.T.conj()*vneR[p0:p0+nG], aoR)
                p0 += nG

            nao = cell.nao_nr()
            vne = [v.reshape(nao,nao) for v in vne]
            if kpts is None or numpy.shape(kpts) == (3,):
                vne = vne[0]
            return vne

    def get_jk(self, cell, dm, hermi=1, kpts=None, kpt_band=None,
               with_j=True, with_k=True, mf=None):
        if kpts is None:
            if numpy.all(self.kpts == 0):
                # Gamma-point calculation by default
                kpts = numpy.zeros(3)
            else:
                kpts = self.kpts

        if kpts.shape == (3,):
            if self.analytic_ft:
                return pwdf_jk.get_jk(self, cell, dm, hermi, mf, kpts,
                                      kpt_band, with_j, with_k)
            else:
                return pscf.hf.get_jk(mf, cell, dm, hermi, None, kpts,
                                      kpt_band)

        if self.analytic_ft:
            vj, vk = None, None
            if with_k:
                vk = pwdf_jk.get_k_kpts(self, cell, dm, hermi, mf, kpts, kpt_band)
            if with_j:
                vj = pwdf_jk.get_j_kpts(self, cell, dm, hermi, mf, kpts, kpt_band)
        else:
            if with_k:
                vj, vk = pscf.khf.get_jk(mf, cell, dm, kpts, kpt_band)
            elif with_j:
                vj = pscf.khf.get_j(mf, cell, dm, kpts, kpt_band)
        return vj, vk

    def get_eri(self, kpts=None, compact=False):
        if self.analytic_ft:
            return pwdf_ao2mo.get_eri(self, kpts, compact)
        else:
            return pwfft_ao2mo.get_eri(self, kpts, compact)
    get_ao_eri = get_eri

    def ao2mo(self, mo_coeffs, kpts=None, compact=False):
        if self.analytic_ft:
            return pwdf_ao2mo.general(self, mo_coeffs, kpts, compact)
        else:
            return pwfft_ao2mo.general(self, mo_coeffs, kpts, compact)
    get_mo_eri = ao2mo

    get_ao_pairs_G = get_ao_pairs = pwfft_ao2mo.get_ao_pairs_G
    get_mo_pairs_G = get_mo_pairs = pwfft_ao2mo.get_mo_pairs_G


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
    v0 = phf.get_pp(cell, k)
    v1 = get_pp(df, cell, k)
    print(numpy.linalg.norm(v1-v0))

