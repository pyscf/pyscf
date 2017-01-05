#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''Density expansion on plane waves'''

import time
import ctypes
import numpy
import copy
from pyscf import lib
from pyscf import gto
from pyscf.lib import logger
from pyscf.pbc import tools
from pyscf.pbc.gto import pseudo
from pyscf.pbc.df import ft_ao
from pyscf.pbc.df import incore
from pyscf.pbc.df import pwdf_jk
from pyscf.pbc.df import pwdf_ao2mo

KPT_DIFF_TOL = 1e-6

def estimate_eta(cell, cutoff=1e-12):
    '''The exponent of the smooth gaussian model density, requiring that at
    boundary, density ~ 4pi rmax^2 exp(-eta*rmax^2) ~ 1e-12
    '''
    eta = max(numpy.log(4*numpy.pi*cell.rcut**4/cutoff)/cell.rcut**2, .1)
    return eta

def get_nuc(mydf, kpts=None):
    cell = mydf.cell
    if kpts is None:
        kpts_lst = numpy.zeros((1,3))
    else:
        kpts_lst = numpy.reshape(kpts, (-1,3))

    log = logger.Logger(mydf.stdout, mydf.verbose)
    t1 = t0 = (time.clock(), time.time())

    nkpts = len(kpts_lst)
    nao = cell.nao_nr()
    nao_pair = nao * (nao+1) // 2

    Gv, Gvbase, kws = cell.get_Gv_weights(mydf.gs)
    kpt_allow = numpy.zeros(3)
    if mydf.eta == 0:
        vpplocG = pseudo.pp_int.get_gth_vlocG_part1(cell, Gv)
        vpplocG = -numpy.einsum('ij,ij->j', cell.get_SI(Gv), vpplocG)
        vpplocG *= kws
        vGR = vpplocG.real
        vGI = vpplocG.imag
        vjR = numpy.zeros((nkpts,nao_pair))
        vjI = numpy.zeros((nkpts,nao_pair))
    else:
        nuccell = copy.copy(cell)
        half_sph_norm = .5/numpy.sqrt(numpy.pi)
        norm = half_sph_norm/gto.mole._gaussian_int(2, mydf.eta)
        chg_env = [mydf.eta, norm]
        ptr_eta = cell._env.size
        ptr_norm = ptr_eta + 1
        chg_bas = [[ia, 0, 1, 1, 0, ptr_eta, ptr_norm, 0] for ia in range(cell.natm)]
        nuccell._atm = cell._atm
        nuccell._bas = numpy.asarray(chg_bas, dtype=numpy.int32)
        nuccell._env = numpy.hstack((cell._env, chg_env))

        # PP-loc part1 is handled by fakenuc in _int_nuc_vloc
        vj = lib.asarray(mydf._int_nuc_vloc(nuccell, kpts_lst))
        vjR = vj.real
        vjI = vj.imag
        t1 = log.timer_debug1('vnuc pass1: analytic int', *t1)

        charge = -cell.atom_charges()
        coulG = tools.get_coulG(cell, kpt_allow, gs=mydf.gs, Gv=Gv)
        coulG *= kws
        aoaux = ft_ao.ft_ao(nuccell, Gv)
        vGR = numpy.einsum('i,xi->x', charge, aoaux.real) * coulG
        vGI = numpy.einsum('i,xi->x', charge, aoaux.imag) * coulG

    max_memory = max(2000, mydf.max_memory-lib.current_memory()[0])
    for k, pqkR, pqkI, p0, p1 \
            in mydf.ft_loop(mydf.gs, kpt_allow, kpts_lst,
                            max_memory=max_memory, aosym='s2'):
# rho_ij(G) nuc(-G) / G^2
# = [Re(rho_ij(G)) + Im(rho_ij(G))*1j] [Re(nuc(G)) - Im(nuc(G))*1j] / G^2
        if not gamma_point(kpts_lst[k]):
            vjI[k] += numpy.einsum('k,xk->x', vGR[p0:p1], pqkI)
            vjI[k] -= numpy.einsum('k,xk->x', vGI[p0:p1], pqkR)
        vjR[k] += numpy.einsum('k,xk->x', vGR[p0:p1], pqkR)
        vjR[k] += numpy.einsum('k,xk->x', vGI[p0:p1], pqkI)
    t1 = log.timer_debug1('contracting Vnuc', *t1)

    if mydf.eta != 0 and cell.dimension == 3:
        nucbar = sum([z/nuccell.bas_exp(i)[0] for i,z in enumerate(charge)])
        nucbar *= numpy.pi/cell.vol
        ovlp = cell.pbc_intor('cint1e_ovlp_sph', 1, lib.HERMITIAN, kpts_lst)
        for k in range(nkpts):
            s = lib.pack_tril(ovlp[k])
            vjR[k] -= nucbar * s.real
            vjI[k] -= nucbar * s.imag

    vj = []
    for k, kpt in enumerate(kpts_lst):
        if gamma_point(kpt):
            vj.append(lib.unpack_tril(vjR[k]))
        else:
            vj.append(lib.unpack_tril(vjR[k]+vjI[k]*1j))

    if kpts is None or numpy.shape(kpts) == (3,):
        vj = vj[0]
    return vj

def _int_nuc_vloc(mydf, nuccell, kpts):
    '''Vnuc - Vloc'''
    cell = mydf.cell
    rcut = max(cell.rcut, nuccell.rcut)
    Ls = cell.get_lattice_Ls(rcut=rcut)
    expLk = numpy.asarray(numpy.exp(1j*numpy.dot(Ls, kpts.T)), order='C')
    nkpts = len(kpts)

# Use the 3c2e code with steep s gaussians to mimic nuclear density
    fakenuc = _fake_nuc(cell)
    fakenuc._atm, fakenuc._bas, fakenuc._env = \
            gto.conc_env(nuccell._atm, nuccell._bas, nuccell._env,
                         fakenuc._atm, fakenuc._bas, fakenuc._env)

    nao = cell.nao_nr()
    buf = [numpy.zeros((nao,nao,fakenuc.natm), order='F', dtype=numpy.complex128)
           for k in range(nkpts)]
    ints = incore._wrap_int3c(cell, fakenuc, 'cint3c2e_sph', 1, Ls, buf)
    atm, bas, env = ints._envs[:3]
    c_shls_slice = (ctypes.c_int*6)(0, cell.nbas, cell.nbas, cell.nbas*2,
                                    cell.nbas*2, cell.nbas*2+fakenuc.natm)

    xyz = numpy.asarray(cell.atom_coords(), order='C')
    ptr_coordL = atm[:cell.natm,gto.PTR_COORD]
    ptr_coordL = numpy.vstack((ptr_coordL,ptr_coordL+1,ptr_coordL+2)).T.copy('C')
    for l, L1 in enumerate(Ls):
        env[ptr_coordL] = xyz + L1
        exp_Lk = numpy.einsum('k,ik->ik', expLk[l].conj(), expLk[:l+1])
        exp_Lk = numpy.asarray(exp_Lk, order='C')
        exp_Lk[l] = .5
        ints(exp_Lk, c_shls_slice)

    charge = cell.atom_charges()
    charge = numpy.append(charge, -charge)  # (charge-of-nuccell, charge-of-fakenuc)
    for k, kpt in enumerate(kpts):
        v = numpy.einsum('ijz,z->ij', buf[k], charge)
        buf[k] = lib.pack_tril(v + v.T.conj())
    return buf

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
# For nuclear attraction integrals using Ewald-like technique.
# Set to 0 to use the regular reciprocal space Poisson-equation method.
        self.eta = estimate_eta(cell)

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
                    pqkR[:] = aoao[i0:i1].real.T
                    pqkI[:] = aoao[i0:i1].imag.T
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

    _int_nuc_vloc = _int_nuc_vloc
    get_nuc = get_nuc
    get_pp = get_pp

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

# Since the real-space lattice-sum for nuclear attraction is not implemented,
# use the 3c2e code with steep gaussians to mimic nuclear density
def _fake_nuc(cell):
    fakenuc = gto.Mole()
    fakenuc._atm = cell._atm.copy()
    fakenuc._atm[:,gto.PTR_COORD] = numpy.arange(gto.PTR_ENV_START,
                                                 gto.PTR_ENV_START+cell.natm*3,3)
    _bas = []
    _env = [0]*gto.PTR_ENV_START + [cell.atom_coords().ravel()]
    ptr = gto.PTR_ENV_START + cell.natm * 3
    half_sph_norm = .5/numpy.sqrt(numpy.pi)
    for ia in range(cell.natm):
        symb = cell.atom_symbol(ia)
        if symb in cell._pseudo:
            pp = cell._pseudo[symb]
            rloc, nexp, cexp = pp[1:3+1]
            eta = .5 / rloc**2
        else:
            eta = 1e16
        norm = half_sph_norm/gto.mole._gaussian_int(2, eta)
        _env.extend([eta, norm])
        _bas.append([ia, 0, 1, 1, 0, ptr, ptr+1, 0])
        ptr += 2
    fakenuc._bas = numpy.asarray(_bas, dtype=numpy.int32)
    fakenuc._env = numpy.asarray(numpy.hstack(_env), dtype=numpy.double)
    fakenuc.rcut = cell.rcut
    return fakenuc


if __name__ == '__main__':
    from pyscf.pbc import gto as pbcgto
    import pyscf.pbc.scf.hf as phf
    from pyscf.pbc import df
    cell = pbcgto.Cell()
    cell.verbose = 0
    cell.atom = 'C 0 0 0; C 1 1 1'
    cell.a = numpy.diag([4, 4, 4])
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.gs = [10, 10, 10]
    cell.build()
    k = numpy.ones(3)*.25
    v1 = PWDF(cell).get_pp(k)
    print(abs(v1).sum(), 21.7504294462)
