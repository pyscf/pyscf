#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Gaussian and planewaves mixed density fitting
Ref:
'''

import time
import copy
import tempfile
import ctypes
import numpy
import h5py
import scipy.linalg
from pyscf import lib
from pyscf import gto
from pyscf.lib import logger
import pyscf.df
import pyscf.df.mdf
from pyscf.pbc import gto as pgto
from pyscf.pbc.df import incore
from pyscf.pbc.df import outcore
from pyscf.pbc import tools
from pyscf.pbc.df import ft_ao
from pyscf.df.mdf import _uncontract_basis
from pyscf.pbc.df import mdf_jk
from pyscf.pbc.df.mdf_jk import zdotNN, zdotCN, zdotNC
from pyscf.pbc.df import mdf_ao2mo
from pyscf.pbc.df import pwdf

KPT_DIFF_TOL = 1e-6

#
# Split the Coulomb potential to two parts.  Computing short range part in
# real space, long range part in reciprocal space.
#

#
# Be very careful with the smooth function exponents.
# The smooth function can be considered as an effective nuclear (distribution)
# potential.  This potential leads to attraction effects somewhere in the
# space.  The attraction potential can interact with diffuse AO basis, which
# may artificially produce a dipole
#
# A relatively steep function is local in space thus maybe have smaller region
# of attraction potential.  An atom-specific eta based on atomic radius might
# be needed
#
def estimate_eta(cell, cutoff=1e-12):
    '''The exponent of the smooth gaussian model density, requiring that at
    boundary, density ~ 4pi rmax^2 exp(-eta*rmax^2) ~ 1e-12
    '''
    rmax = max(lib.norm(cell.lattice_vectors(), axis=0))
    eta = max(-numpy.log(cutoff/(4*numpy.pi*rmax**2))/rmax**2, .1)
    return eta

def make_modrho_basis(cell, auxbasis=None, drop_eta=1.):
    auxcell = copy.copy(cell)
    if auxbasis is None:
        _basis = _uncontract_basis(cell, auxbasis)
    elif isinstance(auxbasis, str):
        uniq_atoms = set([a[0] for a in cell._atom])
        _basis = auxcell.format_basis(dict([(a, auxbasis) for a in uniq_atoms]))
    else:
        _basis = auxcell.format_basis(auxbasis)
    auxcell._basis = _basis
    auxcell._atm, auxcell._bas, auxcell._env = \
            auxcell.make_env(cell._atom, auxcell._basis, cell._env[:gto.PTR_ENV_START])

# Note libcint library will multiply the norm of the integration over spheric
# part sqrt(4pi) to the basis.
    half_sph_norm = numpy.sqrt(.25/numpy.pi)
    steep_shls = []
    ndrop = 0
    for ib in range(len(auxcell._bas)):
        l = auxcell.bas_angular(ib)
        np = auxcell.bas_nprim(ib)
        nc = auxcell.bas_nctr(ib)
        es = auxcell.bas_exp(ib)
        ptr = auxcell._bas[ib,gto.PTR_COEFF]
        cs = auxcell._env[ptr:ptr+np*nc].reshape(nc,np).T

#        if l > 0:
#            continue
        if numpy.any(es < drop_eta):
            es = es[es>=drop_eta]
            cs = cs[es>=drop_eta]
            np, ndrop = len(es), ndrop+np-len(es)
            pe = auxcell._bas[ib,gto.PTR_EXP]
            auxcell._bas[ib,gto.NPRIM_OF] = np
            auxcell._env[pe:pe+np] = es

        if np > 0:
# int1 is the multipole value. l*2+2 is due to the radial part integral
# \int (r^l e^{-ar^2} * Y_{lm}) (r^l Y_{lm}) r^2 dr d\Omega
            int1 = gto.mole._gaussian_int(l*2+2, es)
            s = numpy.einsum('pi,p->i', cs, int1)
# The auxiliary basis normalization factor is not a must for density expansion.
# half_sph_norm here to normalize the monopole (charge).  This convention can
# simplify the formulism of \int \bar{\rho}, see function auxbar.
            cs = numpy.einsum('pi,i->pi', cs, half_sph_norm/s)
            auxcell._env[ptr:ptr+np*nc] = cs.T.reshape(-1)
            steep_shls.append(ib)

    auxcell._bas = numpy.asarray(auxcell._bas[steep_shls], order='C')
    auxcell.nimgs = cell.nimgs
    auxcell._built = True
    logger.debug(cell, 'Drop %d primitive fitting functions', ndrop)
    logger.debug(cell, 'aux basis, num shells = %d, num cGTO = %d',
                 auxcell.nbas, auxcell.nao_nr())
    return auxcell

def make_modchg_basis(auxcell, smooth_eta, l_max=3):
# * chgcell defines smooth gaussian functions for each angular momentum for
#   auxcell. The smooth functions may be used to carry the charge
    chgcell = copy.copy(auxcell)  # smooth model density for coulomb integral to carry charge
    half_sph_norm = .5/numpy.sqrt(numpy.pi)
    chg_bas = []
    chg_env = [smooth_eta]
    ptr_eta = auxcell._env.size
    ptr = ptr_eta + 1
    for ia in range(auxcell.natm):
        for l in set(auxcell._bas[auxcell._bas[:,gto.ATOM_OF]==ia, gto.ANG_OF]):
            if l <= l_max:
                norm = half_sph_norm/gto.mole._gaussian_int(l*2+2, smooth_eta)
                chg_bas.append([ia, l, 1, 1, 0, ptr_eta, ptr, 0])
                chg_env.append(norm)
                ptr += 1

    chgcell._atm = auxcell._atm
    chgcell._bas = numpy.asarray(chg_bas, dtype=numpy.int32).reshape(-1,gto.BAS_SLOTS)
    chgcell._env = numpy.hstack((auxcell._env, chg_env))
    chgcell.nimgs = auxcell.nimgs
    chgcell._built = True
    logger.debug(auxcell, 'smooth basis, num shells = %d, num cGTO = %d',
                 chgcell.nbas, chgcell.nao_nr())
    return chgcell

def get_nuc_less_accurate(mydf, kpts=None):
    log = logger.Logger(mydf.stdout, mydf.verbose)
    t1 = t0 = (time.clock(), time.time())
    if kpts is None:
        kpts_lst = numpy.zeros((1,3))
    else:
        kpts_lst = numpy.reshape(kpts, (-1,3))
    nkpts = len(kpts_lst)
    if mydf._cderi is None:
        mydf.build()
    cell = mydf.cell
    fused_cell, fuse = fuse_auxcell_(mydf, mydf.auxcell)

    nao = cell.nao_nr()
    charge = -cell.atom_charges()
    j2c = pgto.intor_cross('cint2c2e_sph', fused_cell, _fake_nuc(cell))
    jaux = j2c.dot(charge)
    jaux -= charge.sum() * mydf.auxbar(fused_cell)
    Gv = cell.get_Gv(mydf.gs)
    SI = cell.get_SI(Gv)
# The normal nuclues have been considered in function get_gth_vlocG_part1
# The result vG is the potential in G-space for erf part of the pp nuclues and
# "numpy.dot(charge, SI) * coulG" for normal nuclues.
    vpplocG = pgto.pseudo.pp_int.get_gth_vlocG_part1(cell, Gv)
    vG = -1./cell.vol * numpy.einsum('ij,ij->j', SI, vpplocG)
    kpt_allow = numpy.zeros(3)

    if is_zero(kpts_lst):
        vj = numpy.zeros((nkpts,nao**2))
    else:
        vj = numpy.zeros((nkpts,nao**2), dtype=numpy.complex128)
    max_memory = max(2000, mydf.max_memory-lib.current_memory()[0])
    for k, pqkR, pqkI, p0, p1 \
            in mydf.ft_loop(cell, mydf.gs, kpt_allow, kpts_lst, max_memory=max_memory):
        if not gamma_point(kpts_lst[k]):
            vj[k] += numpy.einsum('k,xk->x', vG.real, pqkI) * 1j
            vj[k] += numpy.einsum('k,xk->x', vG.imag, pqkR) *-1j
        vj[k] += numpy.einsum('k,xk->x', vG.real, pqkR)
        vj[k] += numpy.einsum('k,xk->x', vG.imag, pqkI)
        pqkR = pqkI = None

    Gv = cell.get_Gv(mydf.gs)
    aoaux = ft_ao.ft_ao(fused_cell, Gv)
    jaux -= numpy.einsum('x,xj->j', vG.real, aoaux.real)
    jaux -= numpy.einsum('x,xj->j', vG.imag, aoaux.imag)
    jaux = fuse(jaux)

    vj = vj.reshape(-1,nao,nao)
    for k, kpt in enumerate(kpts_lst):
        with mydf.load_Lpq((kpt,kpt)) as Lpq:
            v = 0
            for p0, p1 in lib.prange(0, jaux.size, mydf.blockdim):
                v += numpy.dot(jaux[p0:p1], numpy.asarray(Lpq[p0:p1]))
            if gamma_point(kpt):
                vj[k] += lib.unpack_tril(numpy.asarray(v.real,order='C'))
            else:
                vj[k] += lib.unpack_tril(v)

    if kpts is None or numpy.shape(kpts) == (3,):
        vj = vj[0]
    return vj


def get_nuc(mydf, kpts=None):
    cell = mydf.cell
    if kpts is None:
        kpts_lst = numpy.zeros((1,3))
    else:
        kpts_lst = numpy.reshape(kpts, (-1,3))
    if mydf._cderi is None:
        mydf.build()

    log = logger.Logger(mydf.stdout, mydf.verbose)
    t1 = t0 = (time.clock(), time.time())
    fused_cell, fuse = fuse_auxcell_(mydf, mydf.auxcell)
    nuccell = make_modchg_basis(cell, mydf.eta, 0)
    nuccell._bas = numpy.asarray(nuccell._bas[nuccell._bas[:,gto.ANG_OF]==0],
                                 dtype=numpy.int32, order='C')

    charge = -cell.atom_charges()
    nucbar = sum([z/nuccell.bas_exp(i)[0] for i,z in enumerate(charge)])
    nucbar *= numpy.pi/cell.vol

    vj = [v.ravel() for v in _int_nuc_vloc(cell, nuccell, kpts_lst)]
    t1 = log.timer_debug1('vnuc pass1: analytic int', *t1)
    j2c = pgto.intor_cross('cint2c2e_sph', fused_cell, nuccell)
    jaux = j2c.dot(charge)

    kpt_allow = numpy.zeros(3)
    coulG = tools.get_coulG(cell, kpt_allow, gs=mydf.gs) / cell.vol
    Gv = cell.get_Gv(mydf.gs)
    aoaux = ft_ao.ft_ao(nuccell, Gv)
    vGR = numpy.einsum('i,xi->x', charge, aoaux.real) * coulG
    vGI = numpy.einsum('i,xi->x', charge, aoaux.imag) * coulG

    max_memory = mydf.max_memory - lib.current_memory()[0]
    for k, pqkR, pqkI, p0, p1 \
            in mydf.ft_loop(cell, mydf.gs, kpt_allow, kpts_lst, max_memory=max_memory):
# rho_ij(G) nuc(-G) / G^2
# = [Re(rho_ij(G)) + Im(rho_ij(G))*1j] [Re(nuc(G)) - Im(nuc(G))*1j] / G^2
        if not gamma_point(kpts_lst[k]):
            vj[k] += numpy.einsum('k,xk->x', vGR[p0:p1], pqkI) * 1j
            vj[k] += numpy.einsum('k,xk->x', vGI[p0:p1], pqkR) *-1j
        vj[k] += numpy.einsum('k,xk->x', vGR[p0:p1], pqkR)
        vj[k] += numpy.einsum('k,xk->x', vGI[p0:p1], pqkI)
        pqkR = pqkI = None
    t1 = log.timer_debug1('contracting Vnuc', *t1)

    vG = numpy.einsum('i,xi,x->x', charge, ft_ao.ft_ao(nuccell, Gv), coulG)
    aoaux = ft_ao.ft_ao(fused_cell, Gv)
    jaux -= numpy.einsum('x,xj->j', vG.real, aoaux.real)
    jaux -= numpy.einsum('x,xj->j', vG.imag, aoaux.imag)
    jaux -= charge.sum() * mydf.auxbar(fused_cell)
    jaux = fuse(jaux)
    aoaux = None

    ovlp = cell.pbc_intor('cint1e_ovlp_sph', 1, lib.HERMITIAN, kpts_lst)
    nao = cell.nao_nr()
    nao_pair = nao * (nao+1) // 2
    max_memory = max(2000, mydf.max_memory-lib.current_memory()[0])
    blksize = max(16, min(int(max_memory*1e6/16/nao_pair), mydf.blockdim))
    for k, kpt in enumerate(kpts_lst):
        with mydf.load_Lpq((kpt,kpt)) as Lpq:
            v = 0
            for p0, p1 in lib.prange(0, jaux.size, blksize):
                v += numpy.dot(jaux[p0:p1], numpy.asarray(Lpq[p0:p1]))
        vj[k] = vj[k].reshape(nao,nao) - nucbar * ovlp[k]
        if gamma_point(kpt):
            vj[k] += lib.unpack_tril(numpy.asarray(v.real,order='C'))
        else:
            vj[k] += lib.unpack_tril(v)

    if kpts is None or numpy.shape(kpts) == (3,):
        vj = vj[0]
    return vj


def get_pp(mydf, kpts=None):
    cell = mydf.cell
    if kpts is None:
        kpts_lst = numpy.zeros((1,3))
    else:
        kpts_lst = numpy.reshape(kpts, (-1,3))
    nkpts = len(kpts_lst)

    #vloc1 = get_pp_loc_part1_less_accurate(mydf, kpts_lst)
    vloc1 = get_pp_loc_part1(mydf, kpts_lst)
    vloc2 = pgto.pseudo.pp_int.get_pp_loc_part2(cell, kpts_lst)
    vpp = pgto.pseudo.pp_int.get_pp_nl(cell, kpts_lst)
    for k in range(nkpts):
        vpp[k] += vloc1[k] + vloc2[k]

    if kpts is None or numpy.shape(kpts) == (3,):
        vpp = vpp[0]
    return vpp

get_pp_loc_part1_less_accurate = get_nuc_less_accurate
get_pp_loc_part1 = get_nuc


class MDF(pwdf.PWDF):
    '''Gaussian and planewaves mixed density fitting
    '''
    def __init__(self, cell, kpts=numpy.zeros((1,3))):
        self.cell = cell
        self.stdout = cell.stdout
        self.verbose = cell.verbose
        self.max_memory = cell.max_memory

        self.kpts = kpts  # default is gamma point
        self.gs = cell.gs
        self.metric = 'T'  # or 'S'
        # approximate short range fitting level
        # 0: no approximation;  1: (FIXME) gamma point for k-points;
        # 2: non-pbc approximation;  3: atomic approximation
        self.approx_sr_level = 0
        self.auxbasis = None
        self.eta = None

# Not input options
        self.exxdiv = None  # to mimic KRHF/KUHF object in function get_coulG
        self.auxcell = None
        self.charge_constraint = False
        self.blockdim = 256
        self._j_only = False
        self._cderi_file = tempfile.NamedTemporaryFile()
        self._cderi = None
        self._keys = set(self.__dict__.keys())

    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)
        logger.info(self, '\n')
        logger.info(self, '******** %s flags ********', self.__class__)
        logger.info(self, 'gs = %s', self.gs)
        logger.info(self, 'metric = %s', self.metric)
        logger.info(self, 'approx_sr_level = %s', self.approx_sr_level)
        logger.info(self, 'auxbasis = %s', self.auxbasis)
        logger.info(self, 'eta = %s', self.eta)
        if isinstance(self._cderi, str):
            logger.info(self, '_cderi = %s', self._cderi)
        else:
            logger.info(self, '_cderi = %s', self._cderi_file.name)
        logger.info(self, 'len(kpts) = %d', len(self.kpts))
        logger.debug1(self, '    kpts = %s', self.kpts)

    def build(self, j_only=False, with_j3c=True):
        log = logger.Logger(self.stdout, self.verbose)
        t1 = (time.clock(), time.time())
        cell = self.cell
        if self.eta is None:
            self.eta = estimate_eta(cell)
            log.debug('Set smooth gaussian eta to %.9g', self.eta)
        self.dump_flags()

        self.auxcell = make_modrho_basis(cell, self.auxbasis, self.eta)

        self._j_only = j_only
        if j_only:
            kptij_lst = numpy.hstack((self.kpts,self.kpts)).reshape(-1,2,3)
        else:
            kptij_lst = [(ki, self.kpts[j])
                         for i, ki in enumerate(self.kpts) for j in range(i+1)]
            kptij_lst = numpy.asarray(kptij_lst)

        if not isinstance(self._cderi, str):
            if isinstance(self._cderi_file, str):
                self._cderi = self._cderi_file
            else:
                self._cderi = self._cderi_file.name

        if with_j3c:
            if self.approx_sr_level == 0:
                build_Lpq_pbc(self, self.auxcell, kptij_lst)
            elif self.approx_sr_level == 1:
                build_Lpq_pbc(self, self.auxcell, numpy.zeros((1,2,3)))
            elif self.approx_sr_level == 2:
                build_Lpq_nonpbc(self, self.auxcell)
            elif self.approx_sr_level == 3:
                build_Lpq_1c_approx(self, self.auxcell)
            t1 = log.timer_debug1('Lpq', *t1)

            _make_j3c(self, cell, self.auxcell, kptij_lst)
            t1 = log.timer_debug1('j3c', *t1)
        return self

    def auxbar(self, fused_cell=None):
        '''
        Potential average = \sum_L V_L*Lpq

        The coulomb energy is computed with chargeless density
        \int (rho-C) V,  C = (\int rho) / vol = Tr(gamma,S)/vol
        It is equivalent to removing the averaged potential from the short range V
        vs = vs - (\int V)/vol * S
        '''
        if fused_cell is None:
            fused_cell, fuse = fuse_auxcell_(self, self.auxcell)
        aux_loc = fused_cell.ao_loc_nr()
        half_sph_norm = .5/numpy.sqrt(numpy.pi)
        vbar = numpy.zeros(aux_loc[-1])
        for i in range(fused_cell.nbas):
            l = fused_cell.bas_angular(i)
            if l == 0:
                es = fused_cell.bas_exp(i)
                if es.size == 1:
                    vbar[aux_loc[i]] = -1/es[0]
                else:
# Remove the normalization to get the primitive contraction coeffcients
                    norms = half_sph_norm/gto.mole._gaussian_int(2, es)
                    cs = numpy.einsum('i,ij->ij', 1/norms, fused_cell._libcint_ctr_coeff(i))
                    vbar[aux_loc[i]:aux_loc[i+1]] = numpy.einsum('in,i->n', cs, -1/es)
        vbar *= numpy.pi/fused_cell.vol
        return vbar

    def load_Lpq(self, kpti_kptj=numpy.zeros((2,3))):
        with h5py.File(self._cderi, 'r') as f:
            if self.approx_sr_level == 0:
                return _load3c(self._cderi, 'Lpq', kpti_kptj)
            else:
                kpti, kptj = kpti_kptj
                if is_zero(kpti-kptj):
                    return pyscf.df.addons.load(self._cderi, 'Lpq/0')
                else:
                    # See _fake_Lpq_kpts
                    return pyscf.df.addons.load(self._cderi, 'Lpq/1')

    def load_j3c(self, kpti_kptj=numpy.zeros((2,3))):
        return _load3c(self._cderi, 'j3c', kpti_kptj)

    def sr_loop(self, kpti_kptj=numpy.zeros((2,3)), max_memory=2000,
                compact=True):
        '''Short range part'''
        kpti, kptj = kpti_kptj
        unpack = is_zero(kpti-kptj) and not compact
        nao = self.cell.nao_nr()
        if is_zero(kpti_kptj) and compact:
            nao_pair = nao * (nao+1) // 2
        else:
            nao_pair = nao ** 2
        if is_zero(kpti_kptj):
            blksize = max_memory*1e6/8/(nao_pair*5+nao*(nao+1)//2)
        else:
            blksize = max_memory*1e6/16/(nao_pair*5+nao*(nao+1)//2)
        blksize = max(16, min(int(blksize), self.blockdim))
        logger.debug2(self, 'max_memory %d MB, blksize %d', max_memory, blksize)

        if unpack:
            buf = numpy.empty((blksize,nao*(nao+1)//2))
        def load(Lpq, b0, b1, bufR, bufI):
            Lpq = numpy.asarray(Lpq[b0:b1])
            shape = Lpq.shape
            if unpack:
                tmp = numpy.ndarray(shape, buffer=buf)
                tmp[:] = Lpq.real
                LpqR = lib.unpack_tril(tmp, out=bufR).reshape(-1,nao**2)
                tmp[:] = Lpq.imag
                LpqI = lib.unpack_tril(tmp, lib.ANTIHERMI, out=bufI).reshape(-1,nao**2)
            else:
                LpqR = numpy.ndarray(shape, buffer=bufR)
                LpqR[:] = Lpq.real
                LpqI = numpy.ndarray(shape, buffer=bufI)
                LpqI[:] = Lpq.imag
            return LpqR, LpqI

        LpqR = LpqI = j3cR = j3cI = None
        with self.load_Lpq(kpti_kptj) as Lpq:
            naux = Lpq.shape[0]
            with self.load_j3c(kpti_kptj) as j3c:
                for b0, b1 in lib.prange(0, naux, blksize):
                    LpqR, LpqI = load(Lpq, b0, b1, LpqR, LpqI)
                    j3cR, j3cI = load(j3c, b0, b1, j3cR, j3cI)
                    yield LpqR, LpqI, j3cR, j3cI


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
        kpts = numpy.asarray(kpts)

        if kpts.shape == (3,):
            return mdf_jk.get_jk(self, dm, hermi, kpts, kpt_band, with_j,
                                 with_k, exxdiv)

        vj = vk = None
        if with_k:
            vk = mdf_jk.get_k_kpts(self, dm, hermi, kpts, kpt_band, exxdiv)
        if with_j:
            vj = mdf_jk.get_j_kpts(self, dm, hermi, kpts, kpt_band)
        return vj, vk

    get_eri = get_ao_eri = mdf_ao2mo.get_eri
    ao2mo = get_mo_eri = mdf_ao2mo.general

    def update_mp(self):
        pass

    def update_cc(self):
        pass

    def update(self):
        pass

def unique(kpts):
    kpts = numpy.asarray(kpts)
    nkpts = len(kpts)
    uniq_kpts = []
    uniq_index = []
    uniq_inverse = numpy.zeros(nkpts, dtype=int)
    seen = numpy.zeros(nkpts, dtype=bool)
    n = 0
    for i, kpt in enumerate(kpts):
        if not seen[i]:
            uniq_kpts.append(kpt)
            uniq_index.append(i)
            idx = abs(kpt-kpts).sum(axis=1) < 1e-6
            uniq_inverse[idx] = n
            seen[idx] = True
            n += 1
    return numpy.asarray(uniq_kpts), numpy.asarray(uniq_index), uniq_inverse


def build_Lpq_pbc(mydf, auxcell, kptij_lst):
    '''Fitting coefficients for auxiliary functions'''
    kpts_ji = kptij_lst[:,1] - kptij_lst[:,0]
    uniq_kpts, uniq_index, uniq_inverse = unique(kpts_ji)
    max_memory = max(2000, (mydf.max_memory - lib.current_memory()[0]))
    if mydf.metric.upper() == 'S':
        outcore.aux_e2(mydf.cell, auxcell, mydf._cderi, 'cint3c1e_sph',
                       kptij_lst=kptij_lst, dataname='Lpq',
                       max_memory=max_memory)
        s_aux = auxcell.pbc_intor('cint1e_ovlp_sph', hermi=1, kpts=uniq_kpts)
    else:  # mydf.metric.upper() == 'T'
        outcore.aux_e2(mydf.cell, auxcell, mydf._cderi, 'cint3c1e_p2_sph',
                       kptij_lst=kptij_lst, dataname='Lpq',
                       max_memory=max_memory)
        s_aux = [x*2 for x in auxcell.pbc_intor('cint1e_kin_sph', hermi=1, kpts=uniq_kpts)]

    s_aux = [scipy.linalg.cho_factor(x) for x in s_aux]

    max_memory = mydf.max_memory - lib.current_memory()[0]
    naux = auxcell.nao_nr()
    blksize = max(int(max_memory*.5*1e6/16/naux/mydf.blockdim), 1) * mydf.blockdim
    with h5py.File(mydf._cderi) as feri:
        for k, where in enumerate(uniq_inverse):
            s_k = s_aux[where]
            key = 'Lpq/%d' % k
            Lpq = feri[key]
            nao_pair = Lpq.shape[1]
            for p0, p1 in lib.prange(0, nao_pair, blksize):
                Lpq[:,p0:p1] = scipy.linalg.cho_solve(s_k, Lpq[:,p0:p1])

# TODO: replace incore with outcore
def build_Lpq_nonpbc(mydf, auxcell):
    if mydf.metric.upper() == 'S':
        j3c = pyscf.df.incore.aux_e2(mydf.cell, auxcell, 'cint3c1e_sph',
                                     aosym='s2ij')
        j2c = auxcell.intor_symmetric('cint1e_ovlp_sph')
    else:  # mydf.metric.upper() == 'T'
        j3c = pyscf.df.incore.aux_e2(mydf.cell, auxcell, 'cint3c1e_p2_sph',
                                     aosym='s2ij')
        j2c = auxcell.intor_symmetric('cint1e_kin_sph') * 2

    naux = auxcell.nao_nr()
    nao = mydf.cell.nao_nr()
    nao_pair = nao * (nao+1) // 2
    with h5py.File(mydf._cderi) as feri:
        if 'Lpq' in feri:
            del(feri['Lpq'])
        chunks = (min(mydf.blockdim,naux), min(mydf.blockdim,nao_pair)) # 512K
        Lpq = feri.create_dataset('Lpq/0', (naux,nao_pair), 'f8', chunks=chunks)
        Lpq[:] = lib.cho_solve(j2c, j3c.T)

def build_Lpq_1c_approx(mydf, auxcell):
    get_Lpq = pyscf.df.mdf._make_Lpq_atomic_approx(mydf, mydf.cell, auxcell)

    max_memory = mydf.max_memory - lib.current_memory()[0]
    naux = auxcell.nao_nr()
    nao = mydf.cell.nao_nr()
    nao_pair = nao * (nao+1) // 2
    blksize = max(int(max_memory*.5*1e6/8/naux/mydf.blockdim), 1) * mydf.blockdim
    with h5py.File(mydf._cderi) as feri:
        if 'Lpq' in feri:
            del(feri['Lpq'])
        chunks = (min(mydf.blockdim,naux), min(mydf.blockdim,nao_pair)) # 512K
        Lpq = feri.create_dataset('Lpq/0', (naux,nao_pair), 'f8', chunks=chunks)
        for p0, p1 in lib.prange(0, nao_pair, blksize):
            v = get_Lpq(None, p0, p1)
            Lpq[:,p0:p1] = get_Lpq(None, p0, p1)

def fuse_auxcell_(mydf, auxcell):
    chgcell = make_modchg_basis(auxcell, mydf.eta)
    fused_cell = copy.copy(auxcell)
    fused_cell._atm, fused_cell._bas, fused_cell._env = \
            gto.conc_env(auxcell._atm, auxcell._bas, auxcell._env,
                         chgcell._atm, chgcell._bas, chgcell._env)

    aux_loc = auxcell.ao_loc_nr()
    naux = aux_loc[-1]
    modchg_offset = -numpy.ones((chgcell.natm,8), dtype=int)
    smooth_loc = chgcell.ao_loc_nr()
    for i in range(chgcell.nbas):
        ia = chgcell.bas_atom(i)
        l  = chgcell.bas_angular(i)
        modchg_offset[ia,l] = smooth_loc[i]

    def fuse(Lpq):
        Lpq, chgLpq = Lpq[:naux], Lpq[naux:]
        for i in range(auxcell.nbas):
            l  = auxcell.bas_angular(i)
            ia = auxcell.bas_atom(i)
            p0 = modchg_offset[ia,l]
            if p0 >= 0:
                nd = (aux_loc[i+1] - aux_loc[i]) // auxcell.bas_nctr(i)
                for i0, i1 in lib.prange(aux_loc[i], aux_loc[i+1], nd):
                    Lpq[i0:i1] -= chgLpq[p0:p0+nd]
        return Lpq
    return fused_cell, fuse

def _int_nuc_vloc(cell, nuccell, kpts):
    '''Vnuc - Vloc'''
    nimgs = numpy.max((cell.nimgs, nuccell.nimgs), axis=0)
    Ls = numpy.asarray(cell.get_lattice_Ls(nimgs), order='C')
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
        if gamma_point(kpt):
            buf[k] = v.real + v.real.T
        else:
            buf[k] = v + v.T.conj()
    return buf

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
    fakenuc.nimgs = cell.nimgs
    return fakenuc


# kpti == kptj: s2 symmetry
# kpti == kptj == 0 (gamma point): real
def _make_j3c(mydf, cell, auxcell, kptij_lst):
    t1 = (time.clock(), time.time())
    log = logger.Logger(mydf.stdout, mydf.verbose)
    max_memory = max(2000, mydf.max_memory-lib.current_memory()[0])
    fused_cell, fuse = fuse_auxcell_(mydf, mydf.auxcell)
    outcore.aux_e2(cell, fused_cell, mydf._cderi, 'cint3c2e_sph',
                   kptij_lst=kptij_lst, dataname='j3c', max_memory=max_memory)
    t1 = log.timer_debug1('3c2e', *t1)

    nao = cell.nao_nr()
    naux = auxcell.nao_nr()
    gs = mydf.gs
    gxyz = lib.cartesian_prod((numpy.append(range(gs[0]+1), range(-gs[0],0)),
                               numpy.append(range(gs[1]+1), range(-gs[1],0)),
                               numpy.append(range(gs[2]+1), range(-gs[2],0))))
    invh = numpy.linalg.inv(cell._h)
    Gv = 2*numpy.pi * numpy.dot(gxyz, invh)
    ngs = gxyz.shape[0]

    kptis = kptij_lst[:,0]
    kptjs = kptij_lst[:,1]
    kpt_ji = kptjs - kptis
    uniq_kpts, uniq_index, uniq_inverse = unique(kpt_ji)
    # j2c ~ (-kpt_ji | kpt_ji)
    j2c = fused_cell.pbc_intor('cint2c2e_sph', hermi=1, kpts=uniq_kpts)
    kLRs = []
    kLIs = []
    for k, kpt in enumerate(uniq_kpts):
        aoaux = ft_ao.ft_ao(fused_cell, Gv, None, invh, gxyz, gs, kpt).T
        aoaux = fuse(aoaux)
        coulG = numpy.sqrt(tools.get_coulG(cell, kpt, gs=gs) / cell.vol)
        kLR = (aoaux.real * coulG).T
        kLI = (aoaux.imag * coulG).T
        if not kLR.flags.c_contiguous: kLR = lib.transpose(kLR.T)
        if not kLI.flags.c_contiguous: kLI = lib.transpose(kLI.T)

        j2c[k] = fuse(fuse(j2c[k]).T).T.copy()
        if is_zero(kpt):  # kpti == kptj
            j2c[k] -= lib.dot(kLR.T, kLR)
            j2c[k] -= lib.dot(kLI.T, kLI)
        else:
             # aoaux ~ kpt_ij, aoaux.conj() ~ kpt_kl
            j2cR, j2cI = zdotCN(kLR.T, kLI.T, kLR, kLI)
            j2c[k] -= j2cR + j2cI * 1j

        kLR *= coulG.reshape(-1,1)
        kLI *= coulG.reshape(-1,1)
        kLRs.append(kLR)
        kLIs.append(kLI)
        aoaux = kLR = kLI = j2cR = j2cI = coulG = None

    feri = h5py.File(mydf._cderi)

    # Expand approx Lpq for aosym='s1'.  The approx Lpq are all in aosym='s2' mode
    if mydf.approx_sr_level > 0 and len(kptij_lst) > 1:
        Lpq_fake = _fake_Lpq_kpts(mydf, feri, naux, nao)

    def save(label, dat, col0, col1):
        nrow = dat.shape[0]
        feri[label][:nrow,col0:col1] = dat

    def make_kpt(uniq_kptji_id):  # kpt = kptj - kpti
        kpt = uniq_kpts[uniq_kptji_id]
        log.debug1('kpt = %s', kpt)
        adapted_ji_idx = numpy.where(uniq_inverse == uniq_kptji_id)[0]
        adapted_kptjs = kptjs[adapted_ji_idx]
        nkptj = len(adapted_kptjs)
        log.debug1('adapted_ji_idx = %s', adapted_ji_idx)
        kLR = kLRs[uniq_kptji_id]
        kLI = kLIs[uniq_kptji_id]

        if is_zero(kpt):  # kpti == kptj
            aosym = 's2'
            nao_pair = nao*(nao+1)//2

            vbar = fuse(mydf.auxbar(fused_cell))
            ovlp = cell.pbc_intor('cint1e_ovlp_sph', hermi=1, kpts=adapted_kptjs)
            for k, ji in enumerate(adapted_ji_idx):
                ovlp[k] = lib.pack_tril(ovlp[k])
        else:
            aosym = 's1'
            nao_pair = nao**2

        max_memory = max(2000, mydf.max_memory-lib.current_memory()[0])
        # nkptj for 3c-coulomb arrays plus 1 Lpq array
        buflen = min(max(int(max_memory*.6*1e6/16/naux/(nkptj+1)), 1), nao_pair)
        shranges = pyscf.df.outcore._guess_shell_ranges(cell, buflen, aosym)
        buflen = max([x[2] for x in shranges])
        # +1 for a pqkbuf
        if aosym == 's2':
            Gblksize = max(16, int(max_memory*.2*1e6/16/buflen/(nkptj+1)))
        else:
            Gblksize = max(16, int(max_memory*.4*1e6/16/buflen/(nkptj+1)))
        Gblksize = min(Gblksize, ngs)
        pqkRbuf = numpy.empty(buflen*Gblksize)
        pqkIbuf = numpy.empty(buflen*Gblksize)
        # buf for ft_aopair
        buf = numpy.zeros((nkptj,buflen*Gblksize), dtype=numpy.complex128)

        col1 = 0
        for istep, sh_range in enumerate(shranges):
            log.debug1('int3c2e [%d/%d], AO [%d:%d], ncol = %d', \
                       istep+1, len(shranges), *sh_range)
            bstart, bend, ncol = sh_range
            col0, col1 = col1, col1+ncol
            j3cR = []
            j3cI = []
            for k, idx in enumerate(adapted_ji_idx):
                v = fuse(numpy.asarray(feri['j3c/%d'%idx][:,col0:col1]))

                if mydf.approx_sr_level == 0:
                    Lpq = numpy.asarray(feri['Lpq/%d'%idx][:,col0:col1])
                elif aosym == 's2':
                    Lpq = numpy.asarray(feri['Lpq/0'][:,col0:col1])
                else:
                    Lpq = numpy.asarray(Lpq_fake[:,col0:col1])
                lib.dot(j2c[uniq_kptji_id], Lpq, -.5, v, 1)
                if is_zero(kpt):
                    for i, c in enumerate(vbar):
                        if c != 0:
                            v[i] -= c * ovlp[k][col0:col1]

                j3cR.append(numpy.asarray(v.real, order='C'))
                if is_zero(kpt) and gamma_point(adapted_kptjs[k]):
                    j3cI.append(None)
                else:
                    j3cI.append(numpy.asarray(v.imag, order='C'))
            v = Lpq = None

            if aosym == 's2':
                shls_slice = (bstart, bend, 0, bend)
                for p0, p1 in lib.prange(0, ngs, Gblksize):
                    ft_ao._ft_aopair_kpts(cell, Gv[p0:p1], shls_slice, aosym, invh,
                                          gxyz[p0:p1], gs, kpt, adapted_kptjs, out=buf)
                    nG = p1 - p0
                    for k, ji in enumerate(adapted_ji_idx):
                        aoao = numpy.ndarray((nG,ncol), dtype=numpy.complex128,
                                             order='F', buffer=buf[k])
                        pqkR = numpy.ndarray((ncol,nG), buffer=pqkRbuf)
                        pqkI = numpy.ndarray((ncol,nG), buffer=pqkIbuf)
                        pqkR[:] = aoao.real.T
                        pqkI[:] = aoao.imag.T
                        aoao[:] = 0
                        lib.dot(kLR[p0:p1].T, pqkR.T, -1, j3cR[k], 1)
                        lib.dot(kLI[p0:p1].T, pqkI.T, -1, j3cR[k], 1)
                        if not (is_zero(kpt) and gamma_point(adapted_kptjs[k])):
                            lib.dot(kLR[p0:p1].T, pqkI.T, -1, j3cI[k], 1)
                            lib.dot(kLI[p0:p1].T, pqkR.T,  1, j3cI[k], 1)
            else:
                shls_slice = (bstart, bend, 0, cell.nbas)
                ni = ncol // nao
                for p0, p1 in lib.prange(0, ngs, Gblksize):
                    ft_ao._ft_aopair_kpts(cell, Gv[p0:p1], shls_slice, aosym, invh,
                                          gxyz[p0:p1], gs, kpt, adapted_kptjs, out=buf)
                    nG = p1 - p0
                    for k, ji in enumerate(adapted_ji_idx):
                        aoao = numpy.ndarray((nG,ni,nao), dtype=numpy.complex128,
                                             order='F', buffer=buf[k])
                        pqkR = numpy.ndarray((ni,nao,nG), buffer=pqkRbuf)
                        pqkI = numpy.ndarray((ni,nao,nG), buffer=pqkIbuf)
                        pqkR[:] = aoao.real.transpose(1,2,0)
                        pqkI[:] = aoao.imag.transpose(1,2,0)
                        aoao[:] = 0
                        pqkR = pqkR.reshape(-1,nG)
                        pqkI = pqkI.reshape(-1,nG)
                        zdotCN(kLR[p0:p1].T, kLI[p0:p1].T, pqkR.T, pqkI.T,
                               -1, j3cR[k], j3cI[k], 1)

            for k, ji in enumerate(adapted_ji_idx):
                if is_zero(kpt) and gamma_point(adapted_kptjs[k]):
                    save('j3c/%d'%ji, j3cR[k], col0, col1)
                else:
                    save('j3c/%d'%ji, j3cR[k]+j3cI[k]*1j, col0, col1)


    for k, kpt in enumerate(uniq_kpts):
        make_kpt(k)

    feri.close()

def is_zero(kpt):
    return abs(kpt).sum() < KPT_DIFF_TOL
gamma_point = is_zero

class _load3c(object):
    def __init__(self, cderi, label, kpti_kptj):
        self.cderi = cderi
        self.label = label
        self.kpti_kptj = kpti_kptj
        self.feri = None

    def __enter__(self):
        self.feri = h5py.File(self.cderi, 'r')
        kpti_kptj = numpy.asarray(self.kpti_kptj)
        kptij_lst = self.feri['%s-kptij'%self.label].value
        dk = numpy.einsum('kij->k', abs(kptij_lst-kpti_kptj))
        k_id = numpy.where(dk < 1e-6)[0]
        if len(k_id) > 0:
            dat = self.feri['%s/%d' % (self.label,k_id[0])]
        else:
            # swap ki,kj due to the hermiticity
            kptji = kpti_kptj[[1,0]]
            dk = numpy.einsum('kij->k', abs(kptij_lst-kptji))
            k_id = numpy.where(dk < 1e-6)[0]
            if len(k_id) == 0:
                raise RuntimeError('%s for kpts %s is not initialized.\n'
                                   'Reset attribute .kpts then call '
                                   '.build() to initialize %s.'
                                   % (self.label, kpti_kptj, self.label))
            dat = self.feri['%s/%d' % (self.label, k_id[0])]
            dat = _load_and_unpack(dat)
        return dat

    def __exit__(self, type, value, traceback):
        self.feri.close()

def _fake_Lpq_kpts(mydf, feri, naux, nao):
    chunks = (min(mydf.blockdim,naux), min(mydf.blockdim,nao**2)) # 512K
    Lpq = feri.create_dataset('Lpq/1', (naux,nao**2), 'f8', chunks=chunks)
    for p0, p1 in lib.prange(0, naux, mydf.blockdim):
        Lpq[p0:p1] = lib.unpack_tril(feri['Lpq/0'][p0:p1]).reshape(-1,nao**2)
    return Lpq

class _load_and_unpack(object):
    def __init__(self, dat):
        self.dat = dat
        self.shape = self.dat.shape
    def __getslice__(self, p0, p1):
        nao = int(numpy.sqrt(self.shape[1]))
        v = numpy.asarray(self.dat[p0:p1])
        v = lib.transpose(v.reshape(-1,nao,nao), axes=(0,2,1)).conj()
        return v.reshape(-1,nao**2)
