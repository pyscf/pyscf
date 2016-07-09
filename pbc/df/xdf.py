#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Exact density fitting with Gaussian and planewaves
Ref:
'''

import time
import copy
import tempfile
import ctypes
import numpy
import h5py
from pyscf import lib
from pyscf import gto
from pyscf.lib import logger
import pyscf.df
import pyscf.df.xdf
from pyscf.pbc import gto as pgto
from pyscf.pbc.df import incore
from pyscf.pbc.df import outcore
from pyscf.pbc import tools
from pyscf.pbc.df import ft_ao
from pyscf.df.xdf import _uncontract_basis
from pyscf.pbc.df import xdf_jk
from pyscf.pbc.df import xdf_ao2mo
from pyscf.pbc.df import pwdf

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
def estimate_eta(cell, cutoff=1e-14):
    '''The exponent of the smooth gaussian model density, requiring that at
    boundary, density ~ 4pi rmax^2 exp(-eta*rmax^2) ~ 1e-14
    '''
    rmax = max(lib.norm(cell.lattice_vectors(), axis=0))
    eta = -numpy.log(cutoff/(4*numpy.pi*rmax**2))/rmax**2
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
    chgcell._bas = numpy.asarray(chg_bas, dtype=numpy.int32)
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
    auxcell = mydf.auxcell

    nao = cell.nao_nr()
    charge = -cell.atom_charges()
    j2c = pgto.intor_cross('cint2c2e_sph', auxcell, _fake_nuc(cell))
    jaux = j2c.dot(charge)
    jaux -= charge.sum() * mydf.auxbar(auxcell)
    Gv = cell.get_Gv(mydf.gs)
    SI = cell.get_SI(Gv)
# The normal nuclues have been considered in function get_gth_vlocG_part1
# The result vG is the potential in G-space for erf part of the pp nuclues and
# "numpy.dot(charge, SI) * coulG" for normal nuclues.
    vpplocG = pgto.pseudo.pp_int.get_gth_vlocG_part1(cell, Gv)
    vG = -1./cell.vol * numpy.einsum('ij,ij->j', SI, vpplocG)
    kpt_allow = numpy.zeros(3)
    gamma_point = abs(kpts).sum() < 1e-9

    if gamma_point:
        vj = numpy.zeros((nkpts,nao**2))
    else:
        vj = numpy.zeros((nkpts,nao**2), dtype=numpy.complex128)
    max_memory = mydf.max_memory - lib.current_memory()[0]
    for k, pqkR, LkR, pqkI, LkI, p0, p1 \
            in mydf.ft_loop(cell, auxcell, mydf.gs, kpt_allow, kpts_lst, max_memory):
        if not gamma_point:
            vj[k] += numpy.einsum('k,xk->x', vG.real, pqkI) * 1j
            vj[k] += numpy.einsum('k,xk->x', vG.imag, pqkR) *-1j
        vj[k] += numpy.einsum('k,xk->x', vG.real, pqkR)
        vj[k] += numpy.einsum('k,xk->x', vG.imag, pqkI)
        if k+1 == nkpts:
            jaux -= numpy.einsum('k,xk->x', vG.real, LkR)
            jaux -= numpy.einsum('k,xk->x', vG.imag, LkI)

    nao_pair = nao * (nao+1) // 2
    vj = vj.reshape(-1,nao,nao)
    for k, kpt in enumerate(kpts_lst):
        for Lpq in mydf.load_Lpq((kpt,kpt)):
            vpq = numpy.dot(jaux, Lpq)
            if vpq.shape == nao_pair:
                vpq = lib.unpack_tril(vpq)
            if abs(kpt).sum() < 1e-9:  # gamma point
                vj[k] += vpq.real
            else:
                vj[k] += vpq

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
    auxcell = mydf.auxcell
    nuccell = make_modchg_basis(cell, mydf.eta, 0)
    nuccell._bas = numpy.asarray(nuccell._bas[nuccell._bas[:,gto.ANG_OF]==0],
                                 dtype=numpy.int32, order='C')

    charge = -cell.atom_charges()
    nucbar = sum([z/nuccell.bas_exp(i)[0] for i,z in enumerate(charge)])
    nucbar *= numpy.pi/cell.vol

    vj = [v.ravel() for v in _int_nuc_vloc(cell, nuccell, kpts_lst)]
    t1 = log.timer_debug1('vnuc pass1: analytic int', *t1)
    j2c = pgto.intor_cross('cint2c2e_sph', auxcell, nuccell)
    jaux = j2c.dot(charge)

    kpt_allow = numpy.zeros(3)
    coulG = tools.get_coulG(cell, kpt_allow, gs=mydf.gs) / cell.vol

# Append nuccell to auxcell, so that they can be FT together in pw_loop
# the first [:naux] of ft_ao are aux fitting functions.
    nuccell._atm, nuccell._bas, nuccell._env = \
            gto.conc_env(auxcell._atm, auxcell._bas, auxcell._env,
                         nuccell._atm, nuccell._bas, nuccell._env)
    naux = auxcell.nao_nr()

    max_memory = mydf.max_memory - lib.current_memory()[0]
    for k, pqkR, LkR, pqkI, LkI, p0, p1 \
            in mydf.ft_loop(cell, nuccell, mydf.gs, kpt_allow, kpts_lst, max_memory):
# rho_ij(G) nuc(-G) / G^2
# = [Re(rho_ij(G)) + Im(rho_ij(G))*1j] [Re(nuc(G)) - Im(nuc(G))*1j] / G^2
        vGR = numpy.einsum('i,ix->x', charge, LkR[naux:]) * coulG[p0:p1]
        vGI = numpy.einsum('i,ix->x', charge, LkI[naux:]) * coulG[p0:p1]
        if abs(kpts_lst[k]).sum() > 1e-9:  # if not gamma point
            vj[k] += numpy.einsum('k,xk->x', vGR, pqkI) * 1j
            vj[k] += numpy.einsum('k,xk->x', vGI, pqkR) *-1j
        vj[k] += numpy.einsum('k,xk->x', vGR, pqkR)
        vj[k] += numpy.einsum('k,xk->x', vGI, pqkI)
        if k == 0:
            jaux -= numpy.einsum('k,xk->x', vGR, LkR[:naux])
            jaux -= numpy.einsum('k,xk->x', vGI, LkI[:naux])
    t1 = log.timer_debug1('contracting Vnuc', *t1)

    ovlp = cell.pbc_intor('cint1e_ovlp_sph', 1, lib.HERMITIAN, kpts_lst)
    nao = cell.nao_nr()
    nao_pair = nao * (nao+1) // 2
    jaux -= charge.sum() * mydf.auxbar(auxcell)
    for k, kpt in enumerate(kpts_lst):
        v = vj[k].reshape(nao,nao)
        v -= nucbar * ovlp[k]
        for Lpq in mydf.load_Lpq((kpt,kpt)):
            vpq = numpy.dot(jaux, Lpq)
            if vpq.shape == nao_pair:
                vpq = lib.unpack_tril(vpq)
            if abs(kpt).sum() < 1e-9:  # gamma point
                v = v.real + vpq.real
            else:
                v += vpq
        vj[k] = v

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


class XDF(lib.StreamObject):
    def __init__(self, cell, kpts=numpy.zeros((1,3))):
        self.cell = cell
        self.stdout = cell.stdout
        self.verbose = cell.verbose
        self.max_memory = cell.max_memory

        self.kpts = kpts  # default is gamma point
        self.gs = cell.gs
        self.metric = 'T'  # or 'S'
        self.approx_sr_level = 0  # approximate short range fitting level
        self.auxbasis = None
        self.eta = 1 #None

# Not input options
        self.exxdiv = None  # to mimic KRHF/KUHF object in function get_coulG
        self.auxcell = None
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

    def build(self, j_only=False, with_Lpq=True, with_j3c=True):
        log = logger.Logger(self.stdout, self.verbose)
        t1 = (time.clock(), time.time())
        cell = self.cell
        if self.eta is None:
            self.eta = estimate_eta(cell)
            log.debug('Set smooth gaussian eta to %.9g', self.eta)
        self.dump_flags()

        auxcell = make_modrho_basis(cell, self.auxbasis, self.eta)
        chgcell = make_modchg_basis(auxcell, self.eta)

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
        if len(self.kpts) == 1:
            aosym = 's2ij'
        else:
            aosym = 's1'

        if with_Lpq:
            if self.approx_sr_level == 0:
                build_Lpq_pbc(self, auxcell, chgcell, aosym, kptij_lst)
            elif self.approx_sr_level == 1:
                build_Lpq_pbc(self, auxcell, chgcell, aosym, numpy.zeros((1,2,3)))
            elif self.approx_sr_level == 2:
                build_Lpq_nonpbc(self, auxcell, chgcell)
            elif self.approx_sr_level == 3:
                build_Lpq_1c_approx(self, auxcell, chgcell)
            elif self.approx_sr_level == 4:
                build_Lpq_atomic(self, auxcell, chgcell, self.eta)

# Merge chgcell into auxcell
        auxcell._atm, auxcell._bas, auxcell._env = \
                gto.conc_env(auxcell._atm, auxcell._bas, auxcell._env,
                             chgcell._atm, chgcell._bas, chgcell._env)
        self.auxcell = auxcell
        t1 = log.timer_debug1('Lpq', *t1)

        if with_j3c:
            outcore.aux_e2(cell, auxcell, self._cderi, 'cint3c2e_sph',
                           aosym=aosym, kptij_lst=kptij_lst, dataname='j3c',
                           max_memory=self.max_memory)
            t1 = log.timer_debug1('3c2e', *t1)
        return self

    def auxbar(self, auxcell=None):
        '''
        Potential average = \sum_L V_L*Lpq

        The coulomb energy is computed with chargeless density
        \int (rho-C) V,  C = (\int rho) / vol = Tr(gamma,S)/vol
        It is equivalent to removing the averaged potential from the short range V
        vs = vs - (\int V)/vol * S
        '''
        if auxcell is None:
            auxcell = self.auxcell
        aux_loc = auxcell.ao_loc_nr()
        half_sph_norm = .5/numpy.sqrt(numpy.pi)
        vbar = numpy.zeros(aux_loc[-1])
        for i in range(auxcell.nbas):
            l = auxcell.bas_angular(i)
            if l == 0:
                es = auxcell.bas_exp(i)
                if es.size == 1:
                    vbar[aux_loc[i]] = -1/es[0]
                else:
# Remove the normalization to get the primitive contraction coeffcients
                    norms = half_sph_norm/gto.mole._gaussian_int(2, es)
                    cs = numpy.einsum('i,ij->ij', 1/norms, auxcell.bas_ctr_coeff(i))
                    vbar[aux_loc[i]:aux_loc[i+1]] = numpy.einsum('in,i->n', cs, -1/es)
        vbar *= numpy.pi/auxcell.vol
        return vbar

    def load_Lpq(self, kpti_kptj=numpy.zeros((2,3))):
        with h5py.File(self._cderi, 'r') as f:
            if self.approx_sr_level == 0:
                kptij_lst = f['Lpq-kptij'].value
                dk = numpy.einsum('kij->k', abs(kptij_lst-kpti_kptj))
                k_id = numpy.where(dk < 1e-6)[0]
                if len(k_id) > 0:
                    Lpq = f['Lpq/%d'%(k_id[0])].value
                else:
                    dk = numpy.einsum('kij->k', abs(kptij_lst-kpti_kptj[::-1]))
                    k_id = numpy.where(dk < 1e-6)[0]
                    if len(k_id) == 0:
                        raise RuntimeError('Lpq for kpts [%s,%s] is not initialized.\n'
                                           'Reset attribute .kpts then call '
                                           '.build() to initialize Lpq.'
                                           % tuple(kpti_kptj))
                    Lpq = f['Lpq/%d'%(k_id[0])].value
                    nao = int(numpy.sqrt(Lpq.shape[1]))
                    Lpq = Lpq.reshape(-1,nao,nao).transpose(0,2,1).conj()
                    Lpq = Lpq.reshape(-1,nao**2)
            elif self.approx_sr_level == 1:
                Lpq = f['Lpq/0'].value
            else:
                Lpq = f['Lpq'].value
            yield Lpq

    def load_j3c(self, kpti_kptj=numpy.zeros((2,3))):
        kpti, kptj = kpti_kptj
        nao = self.cell.nao_nr()
        with h5py.File(self._cderi, 'r') as f:
            kptij_lst = f['j3c-kptij'].value
            dk = numpy.einsum('kij->k', abs(kptij_lst-kpti_kptj))
            k_id = numpy.where(dk < 1e-6)[0]
            if len(k_id) > 0:
                j3c = f['j3c/%d'%(k_id[0])].value
            else:
                dk = numpy.einsum('kij->k', abs(kptij_lst-kpti_kptj[::-1]))
                k_id = numpy.where(dk < 1e-6)[0]
                if len(k_id) == 0:
                    raise RuntimeError('j3c for kpts [%s,%s] is not initialized.\n'
                                       'Reset attribute .kpts then call '
                                       '.build() to initialize j3c.'
                                       % tuple(kpti_kptj))
                j3c = f['j3c/%d'%(k_id[0])].value
                j3c = j3c.reshape(-1,nao,nao).transpose(0,2,1).conj()
                j3c = j3c.reshape(-1,nao**2)

            if abs(kpti-kptj).sum() < 1e-9:
                vbar = self.auxbar()
                ovlp = self.cell.pbc_intor('cint1e_ovlp_sph', hermi=1, kpts=kptj)
                if j3c.shape[1] == nao*(nao+1)//2:
                    ovlp = lib.pack_tril(ovlp)
                else:
                    ovlp = ovlp.ravel()
            else:
                vbar = []
                ovlp = 0
            for i, c in enumerate(vbar):
                if c != 0:
                    j3c[i] -= c * ovlp
            yield j3c

    def pw_loop(self, cell, auxcell, gs=None, kpti_kptj=None, max_memory=2000):
        '''Plane wave part'''
        if gs is None:
            gs = self.gs
        if kpti_kptj is None:
            kpti = kptj = numpy.zeros(3)
        else:
            kpti, kptj = kpti_kptj

        nao = cell.nao_nr()
        naux = auxcell.nao_nr()
        gxrange = numpy.append(range(gs[0]+1), range(-gs[0],0))
        gyrange = numpy.append(range(gs[1]+1), range(-gs[1],0))
        gzrange = numpy.append(range(gs[2]+1), range(-gs[2],0))
        gxyz = lib.cartesian_prod((gxrange, gyrange, gzrange))
        invh = numpy.linalg.inv(cell._h)
        Gv = 2*numpy.pi * numpy.dot(gxyz, invh)
        ngs = gxyz.shape[0]

# Theoretically, hermitian symmetry can be also found for kpti == kptj:
#       f_ji(G) = \int f_ji exp(-iGr) = \int f_ij^* exp(-iGr) = [f_ij(-G)]^*
# The hermi operation needs reordering the axis-0.  It is inefficient
        hermi = abs(kpti).sum() < 1e-9 and abs(kptj).sum() < 1e-9  # gamma point

        blksize = min(max(16, int(max_memory*1e6*.7/16/nao**2)), 16384)
        sublk = max(16, int(blksize//4))
        pqkRbuf = numpy.empty(nao*nao*sublk)
        pqkIbuf = numpy.empty(nao*nao*sublk)
        LkRbuf = numpy.empty(naux*sublk)
        LkIbuf = numpy.empty(naux*sublk)

        for p0, p1 in lib.prange(0, ngs, blksize):
            aoao = ft_ao.ft_aopair(cell, Gv[p0:p1], None, hermi, invh,
                                   gxyz[p0:p1], gs, (kpti, kptj))
            aoaux = ft_ao.ft_ao(auxcell, Gv[p0:p1], None, invh,
                                gxyz[p0:p1], gs, kptj-kpti)

            for i0, i1 in lib.prange(0, p1-p0, sublk):
                nG = i1 - i0
                pqkR = numpy.ndarray((nao,nao,nG), buffer=pqkRbuf)
                pqkI = numpy.ndarray((nao,nao,nG), buffer=pqkIbuf)
                pqkR[:] = aoao[i0:i1].real.transpose(1,2,0)
                pqkI[:] = aoao[i0:i1].imag.transpose(1,2,0)
                kLR = numpy.ndarray((nG,naux), buffer=LkRbuf)
                kLI = numpy.ndarray((nG,naux), buffer=LkIbuf)
                kLR [:] = aoaux[i0:i1].real
                kLI [:] = aoaux[i0:i1].imag
                yield (pqkR.reshape(-1,nG), kLR.T,
                       pqkI.reshape(-1,nG), kLI.T, p0+i0, p0+i1)

    def ft_loop(self, cell, auxcell, gs=None, kpt=numpy.zeros(3),
                kpts=None, max_memory=4000):
        '''
        Fourier transform iterator for all kpti which satisfy  kpt = kpts - kpti
        '''
        if gs is None: gs = self.gs
        if kpts is None:
            assert(abs(kpt).sum() < 1e-9)
            kpts = self.kpts
        kpts = numpy.asarray(kpts)
        nkpts = len(kpts)

        nao = cell.nao_nr()
        naux = auxcell.nao_nr()
        gxrange = numpy.append(range(gs[0]+1), range(-gs[0],0))
        gyrange = numpy.append(range(gs[1]+1), range(-gs[1],0))
        gzrange = numpy.append(range(gs[2]+1), range(-gs[2],0))
        gxyz = lib.cartesian_prod((gxrange, gyrange, gzrange))
        invh = numpy.linalg.inv(cell._h)
        Gv = 2*numpy.pi * numpy.dot(gxyz, invh)
        ngs = gxyz.shape[0]

        blksize = min(max(16, int(max_memory*1e6*.9/(nao**2*(nkpts+1)*16))), 16384)
        buf = [numpy.zeros(nao*nao*blksize, dtype=numpy.complex128)
               for k in range(nkpts)]
        pqkRbuf = numpy.empty(nao*nao*blksize)
        pqkIbuf = numpy.empty(nao*nao*blksize)
        LkRbuf = numpy.empty(naux*blksize)
        LkIbuf = numpy.empty(naux*blksize)

        for p0, p1 in lib.prange(0, ngs, blksize):
            aoaux = ft_ao.ft_ao(auxcell, Gv[p0:p1], None, invh,
                                gxyz[p0:p1], gs, kpt)
            nG = p1 - p0
            LkR = numpy.ndarray((naux,nG), buffer=LkRbuf)
            LkI = numpy.ndarray((naux,nG), buffer=LkIbuf)
            LkR [:] = aoaux.real.T
            LkI [:] = aoaux.imag.T

            ft_ao._ft_aopair_kpts(cell, Gv[p0:p1], None, True, invh,
                                  gxyz[p0:p1], gs, kpt, kpts, out=buf)
            for k in range(nkpts):
                aoao = numpy.ndarray((nG,nao,nao), dtype=numpy.complex128,
                                     order='F', buffer=buf[k])
                pqkR = numpy.ndarray((nao,nao,nG), buffer=pqkRbuf)
                pqkI = numpy.ndarray((nao,nao,nG), buffer=pqkIbuf)
                pqkR[:] = aoao.real.transpose(1,2,0)
                pqkI[:] = aoao.imag.transpose(1,2,0)
                yield (k, pqkR.reshape(-1,nG), LkR, pqkI.reshape(-1,nG), LkI, p0, p1)
                aoao[:] = 0

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
        else:
            kpts = numpy.asarray(kpts)

        # Use DF object to mimic KRHF/KUHF object in function get_coulG
        self.exxdiv = exxdiv

        if kpts.shape == (3,):
            return xdf_jk.get_jk(self, dm, hermi, kpts, kpt_band, with_j, with_k)

        vj = vk = None
        if with_k:
            vk = xdf_jk.get_k_kpts(self, dm, hermi, kpts, kpt_band)
        if with_j:
            vj = xdf_jk.get_j_kpts(self, dm, hermi, kpts, kpt_band)
        return vj, vk

    get_eri = get_ao_eri = xdf_ao2mo.get_eri
    ao2mo = get_mo_eri = xdf_ao2mo.general

    def update_mf(self, mf):
        return xdf_jk.density_fit(mf, with_df=self)

    def update_mp(self):
        pass

    def update_cc(self):
        pass

    def update(self):
        pass



def build_Lpq_pbc(mydf, auxcell, chgcell, aosym, kptij_lst):
    '''Fitting coefficients for auxiliary functions'''
    kpts_ji = kptij_lst[:,1] - kptij_lst[:,0]
    if mydf.metric.upper() == 'S':
        outcore.aux_e2(mydf.cell, auxcell, mydf._cderi, 'cint3c1e_sph',
                       aosym=aosym, kptij_lst=kptij_lst, dataname='Lpq',
                       max_memory=mydf.max_memory)
        j2c = auxcell.pbc_intor('cint1e_ovlp_sph', hermi=1, kpts=kpts_ji)
    else:  # mydf.metric.upper() == 'T'
        outcore.aux_e2(mydf.cell, auxcell, mydf._cderi, 'cint3c1e_p2_sph',
                       aosym=aosym, kptij_lst=kptij_lst, dataname='Lpq',
                       max_memory=mydf.max_memory)
        j2c = [x*2 for x in auxcell.pbc_intor('cint1e_kin_sph', hermi=1, kpts=kpts_ji)]

    with h5py.File(mydf._cderi) as feri:
        for k, j2c_k in enumerate(j2c):
            key = 'Lpq/%d' % k
            Lpq = feri[key].value
            del(feri[key])
            Lpq = lib.cho_solve(j2c_k, Lpq)
            feri[key] = compress_Lpq_to_chgcell(Lpq, auxcell, chgcell)

def build_Lpq_nonpbc(mydf, auxcell, chgcell):
    if mydf.metric.upper() == 'S':
        Lpq = pyscf.df.incore.aux_e2(mydf.cell, auxcell, 'cint3c1e_sph',
                                     aosym='s2ij')
        j2c = auxcell.intor_symmetric('cint1e_ovlp_sph')
    else:  # mydf.metric.upper() == 'T'
        Lpq = pyscf.df.incore.aux_e2(mydf.cell, auxcell, 'cint3c1e_p2_sph',
                                     aosym='s2ij')
        j2c = auxcell.intor_symmetric('cint1e_kin_sph') * 2

    with h5py.File(mydf._cderi) as feri:
        if 'Lpq' in feri:
            del(feri['Lpq'])
        Lpq = lib.cho_solve(j2c, Lpq.T)
        feri['Lpq'] = compress_Lpq_to_chgcell(Lpq, auxcell, chgcell)

def build_Lpq_1c_approx(mydf, auxcell, chgcell):
    def fsolve(mol, auxmol):
        if mydf.metric.upper() == 'S':
            j3c = pyscf.df.incore.aux_e2(mol, auxmol, 'cint3c1e_sph',
                                         aosym='s2ij')
            j2c = auxmol.intor_symmetric('cint1e_ovlp_sph')
        else:  # mydf.metric.upper() == 'T'
            j3c = pyscf.df.incore.aux_e2(mol, auxmol, 'cint3c1e_p2_sph',
                                         aosym='s2ij')
            j2c = auxmol.intor_symmetric('cint1e_kin_sph') * 2
        return lib.cho_solve(j2c, j3c.T)

    nao = mydf.cell.nao_nr()
    naux = auxcell.nao_nr()
    Lpq = numpy.zeros((naux, nao*(nao+1)//2))
    mol1 = copy.copy(mydf.cell)
    mol2 = copy.copy(auxcell)
    i0 = 0
    j0 = 0
    for ia in range(mydf.cell.natm):
        mol1._bas = mydf.cell._bas[mydf.cell._bas[:,gto.ATOM_OF] == ia]
        mol2._bas = auxcell._bas[auxcell._bas[:,gto.ATOM_OF] == ia]
        tmp = fsolve(mol1, mol2)
        di = tmp.shape[0]
        dj = mol1.nao_nr()
# idx is the diagonal block of the lower triangular indices
        idx = numpy.hstack([range(i*(i+1)//2+j0,i*(i+1)//2+i+1)
                            for i in range(j0, j0+dj)])
        Lpq[i0:i0+di,idx] = tmp
        i0 += di
        j0 += dj

    with h5py.File(mydf._cderi) as feri:
        if 'Lpq' in feri:
            del(feri['Lpq'])
        feri['Lpq'] = compress_Lpq_to_chgcell(Lpq, auxcell, chgcell)

def build_Lpq_atomic(mydf, auxcell, chgcell, drop_eta):
    '''Solve atomic fitting coefficients and drop the coefficients of smooth
    functions'''
    def solve_Lpq(mol, auxmol):
        if mydf.metric.upper() == 'S':
            j3c = pyscf.df.incore.aux_e2(mol, auxmol, 'cint3c1e_sph',
                                         aosym='s2ij')
            j2c = auxmol.intor_symmetric('cint1e_ovlp_sph')
        else:  # mydf.metric.upper() == 'T'
            j3c = pyscf.df.incore.aux_e2(mol, auxmol, 'cint3c1e_p2_sph',
                                         aosym='s2ij')
            j2c = auxmol.intor_symmetric('cint1e_kin_sph') * 2
        Lpq = lib.cho_solve(j2c, j3c.T)

        half_sph_norm = numpy.sqrt(.25/numpy.pi)
        mask = numpy.ones(Lpq.shape[0], dtype=bool)
        p0 = 0
        for ib in range(len(auxmol._bas)):
            l = auxmol.bas_angular(ib)
            np = auxmol.bas_nprim(ib)
            nc = auxmol.bas_nctr(ib)
            es = auxmol.bas_exp(ib)
            cs = auxmol.bas_ctr_coeff(ib)
            nd = (l*2+1) * nc
            norm1 = 1/numpy.einsum('pi,p->i', cs, gto.mole._gaussian_int(l+2, es))

            if numpy.any(es < drop_eta):
                es = es[es>=drop_eta]
                cs = cs[es>=drop_eta]
                np = len(es)
                pe = auxmol._bas[ib,gto.PTR_EXP]

            if np > 0:
                norm2 = 1/numpy.einsum('pi,p->i', cs, gto.mole._gaussian_int(l+2, es))
# scale Lpq, because it will contract to the truncated mol
                Lpq[p0:p0+nd] *= (norm1/norm2).reshape(-1,1)
            else:
                mask[p0:p0+nd] = False
            p0 += nd
        return Lpq[mask]

    with h5py.File(mydf._cderi) as feri:
        if 'Lpq' in feri:
            del(feri['Lpq'])
        Lpq = pyscf.df.xdf._make_Lpq_atomic_approx(mydf.cell, auxcell, solve_Lpq)
        feri['Lpq'] = compress_Lpq_to_chgcell(Lpq, auxcell, chgcell)

def compress_Lpq_to_chgcell(Lpq, auxcell, chgcell):
    aux_loc = auxcell.ao_loc_nr()
    modchg_offset = -numpy.ones((chgcell.natm,8), dtype=int)
    smooth_loc = chgcell.ao_loc_nr()
    for i in range(chgcell.nbas):
        ia = chgcell.bas_atom(i)
        l  = chgcell.bas_angular(i)
        modchg_offset[ia,l] = smooth_loc[i]
    naochg = chgcell.nao_nr()

    nao_pair = Lpq.shape[-1]
    auxchgs = numpy.zeros((naochg,nao_pair), dtype=Lpq.dtype)
    for i in range(auxcell.nbas):
        l  = auxcell.bas_angular(i)
        ia = auxcell.bas_atom(i)
        p0 = modchg_offset[ia,l]
        if p0 >= 0:
            nc = auxcell.bas_nctr(i)
            lchg = numpy.einsum('imn->mn', Lpq[aux_loc[i]:aux_loc[i+1]].reshape(nc,-1,nao_pair))
            auxchgs[p0:p0+len(lchg)] -= lchg

    if Lpq.flags.f_contiguous:
        Lpq = lib.transpose(Lpq.T)
    Lpq = numpy.vstack((Lpq, auxchgs))
    return Lpq

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
        if abs(kpt).sum() < 1e-9:  # gamma_point:
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
