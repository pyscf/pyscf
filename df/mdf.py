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
import numpy
import scipy.linalg
import h5py
from pyscf import lib
from pyscf import gto
from pyscf.lib import logger
from pyscf.gto import ANG_OF, PTR_COEFF
from pyscf.gto import ft_ao
from pyscf.df import addons
from pyscf.df import incore

#
# Split the Coulomb potential to two parts.  Computing short range part in
# real space, long range part in reciprocal space.
#

def make_modrho_basis(mol, auxbasis=None):
    auxmol = copy.copy(mol)   # DF basis to fit density

    if auxbasis is None:
        #_basis = addons.aug_etb_for_dfbasis(mol, beta=2.2, start_at=0)
        _basis = _uncontract_basis(mol)
    elif isinstance(auxbasis, str):
        uniq_atoms = set([a[0] for a in mol._atom])
        _basis = auxmol.format_basis(dict([(a, auxbasis) for a in uniq_atoms]))
    else:
        _basis = auxmol.format_basis(auxbasis)
    auxmol._basis = _basis
    auxmol._atm, auxmol._bas, auxmol._env = \
            auxmol.make_env(mol._atom, auxmol._basis, mol._env[:gto.PTR_ENV_START])

# Note libcint library will multiply the norm of the integration over spheric
# part sqrt(4pi) to the basis.
    half_sph_norm = numpy.sqrt(.25/numpy.pi)
    for ib in range(len(auxmol._bas)):
        l = auxmol.bas_angular(ib)
        np = auxmol.bas_nprim(ib)
        nc = auxmol.bas_nctr(ib)
        es = auxmol.bas_exp(ib)
        ptr = auxmol._bas[ib,PTR_COEFF]
        cs = auxmol._env[ptr:ptr+np*nc].reshape(nc,np).T

# int1 is the multipole value. l*2+2 is due to the radial part integral
# \int (r^l e^{-ar^2} * Y_{lm}) (r^l Y_{lm}) r^2 dr d\Omega
        int1 = gto.mole._gaussian_int(l*2+2, es)
        s = numpy.einsum('pi,p->i', cs, int1)
        cs = numpy.einsum('pi,i->pi', cs, half_sph_norm/s)
        auxmol._env[ptr:ptr+np*nc] = cs.T.reshape(-1)
    auxmol._built = True
    logger.debug(mol, 'aux basis, num shells = %d, num cGTO = %d',
                 auxmol.nbas, auxmol.nao_nr())
    return auxmol


def non_uniform_kgrids(gs):
    from pyscf.dft import gen_grid
    def plus_minus(n):
        #rs, ws = gen_grid.radi.delley(n)
        #rs, ws = gen_grid.radi.treutler_ahlrichs(n)
        #rs, ws = gen_grid.radi.mura_knowles(n)
        rs, ws = gen_grid.radi.gauss_chebyshev(n)
        return numpy.hstack((rs,-rs)), numpy.hstack((ws,ws))
    rx, wx = plus_minus(gs[0])
    ry, wy = plus_minus(gs[1])
    rz, wz = plus_minus(gs[2])
    Gvbase = (rx, ry, rz)
    Gv = lib.cartesian_prod(Gvbase)
    weights = numpy.einsum('i,j,k->ijk', wx, wy, wz).reshape(-1)
    return Gv, Gvbase, weights

def _uncontract_basis(mol, basis=None, eta=.1, l_max=None):
    if basis is None:
        basis = mol._basis
    elif isinstance(basis, str):
        uniq_atoms = set([a[0] for a in mol._atom])
        basis = mol.format_basis(dict([(a, basis) for a in uniq_atoms]))
    else:
        basis = mol.format_basis(basis)

    def setup(atom):
        if l_max is None:
            conf = lib.parameters.ELEMENTS[gto.mole._charge(atom)][2]
# pp exchange for p elements, dd exchange for d elements, ff exchange
# for f elements
            l = len([i for i in conf if i > 0]) - 1
            lmax = max(1, l*2)
        else:
            lmax = l_max
        return (atom, [(b[0], (b[1][0], 1)) for b in gto.uncontract(basis[atom])
                       if b[0] <= lmax and b[1][0] > eta])
    basis = dict([setup(a) for a in mol._basis])
    return basis


class MDF(lib.StreamObject):
    def __init__(self, mol):
        self.mol = mol
        self.stdout = mol.stdout
        self.verbose = mol.verbose
        self.max_memory = mol.max_memory

        self.gs = (10,10,10)
        self.metric = 'S'  # or 'T' or 'J'
        self.approx_sr_level = 0
        self.charge_constraint = True
        self.auxbasis = None
        self.auxmol = None
        self._cderi_file = tempfile.NamedTemporaryFile()
        self._cderi = None

    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)
        logger.info(self, '\n')
        logger.info(self, '******** %s flags ********', self.__class__)
        logger.info(self, 'gs = %s', self.gs)
        logger.info(self, 'metric = %s', self.metric)
        logger.info(self, 'approx_sr_level = %s', self.approx_sr_level)
        logger.info(self, 'charge_constraint = %s', self.charge_constraint)
        logger.info(self, 'auxbasis = %s', self.auxbasis)
        if isinstance(self._cderi, str):
            logger.info(self, '_cderi = %s', self._cderi)
        else:
            logger.info(self, '_cderi = %s', self._cderi_file.name)

    def build(self):
        self.dump_flags()

        mol = self.mol
        auxmol = self.auxmol = make_modrho_basis(self.mol, self.auxbasis)
        nao = mol.nao_nr()
        naux = auxmol.nao_nr()

        if not isinstance(self._cderi, str):
            if isinstance(self._cderi_file, str):
                self._cderi = self._cderi_file
            else:
                self._cderi = self._cderi_file.name

        fLpq = _Lpq_solver(self.metric, self.charge_constraint)
        if self.approx_sr_level == 0:
            with h5py.File(self._cderi, 'w') as f:
                f['Lpq'] = fLpq(mol, auxmol)
        else:
            with h5py.File(self._cderi, 'w') as f:
                f['Lpq'] = _make_Lpq_atomic_approx(mol, auxmol, fLpq)
        return self

    def load(self, key):
        return addons.load(self._cderi, key)

    def pw_loop(self, mol, auxmol, gs, max_memory=2000):
        '''Plane wave part'''
        if isinstance(gs, int):
            gs = [gs]*3
        nao = mol.nao_nr()
        naux = auxmol.nao_nr()
        Gv, Gvbase, kws = non_uniform_kgrids(gs)
        nxyz = [i*2 for i in gs]
        gxyz = lib.cartesian_prod([range(i) for i in nxyz])
        kk = numpy.einsum('ki,ki->k', Gv, Gv)
        idx = numpy.argsort(kk)[::-1]
#        idx = idx[(kk[idx] < 300.) & (kk[idx] > 1e-4)]  # ~ Cut high energy plain waves
#        log.debug('Cut grids %d to %d', Gv.shape[0], len(idx))
        kk = kk[idx]
        Gv = Gv[idx]
        kws = kws[idx]
        gxyz = gxyz[idx]

        blksize = min(max(16, int(max_memory*1e6*.7/16/nao**2)), 16384)
        sublk = max(16,int(blksize//4))
        pqkRbuf = numpy.empty(nao*nao*sublk)
        pqkIbuf = numpy.empty(nao*nao*sublk)
        LkRbuf = numpy.empty(naux*sublk)
        LkIbuf = numpy.empty(naux*sublk)

        for p0, p1 in lib.prange(0, kws.size, blksize):
            aoao = ft_ao.ft_aopair(mol, Gv[p0:p1], None, True,
                                   Gvbase, gxyz[p0:p1], nxyz)
            aoaux = ft_ao.ft_ao(auxmol, Gv[p0:p1], None, Gvbase, gxyz[p0:p1], nxyz)

            for i0, i1 in lib.prange(0, p1-p0, sublk):
                nG = i1 - i0
                coulG = .5/numpy.pi**2 * kws[p0+i0:p0+i1] / kk[p0+i0:p0+i1]
                pqkR = numpy.ndarray((nao,nao,nG), buffer=pqkRbuf)
                pqkI = numpy.ndarray((nao,nao,nG), buffer=pqkIbuf)
                LkR = numpy.ndarray((naux,nG), buffer=LkRbuf)
                LkI = numpy.ndarray((naux,nG), buffer=LkIbuf)
                pqkR[:] = aoao[i0:i1].real.transpose(1,2,0)
                pqkI[:] = aoao[i0:i1].imag.transpose(1,2,0)
                LkR [:] = aoaux[i0:i1].real.T
                LkI [:] = aoaux[i0:i1].imag.T
                yield pqkR.reshape(-1,nG), LkR, pqkI.reshape(-1,nG), LkI, coulG

    def sr_loop(self, mol, auxmol, max_memory=2000):
        '''Short range part'''
        j3c = incore.aux_e2(mol, auxmol, 'cint3c2e_sph', aosym='s2ij')
        j2c = incore.fill_2c2e(mol, auxmol)
        for i in range(1):
            yield j3c, j2c


    def get_jk(self, mol, dm, hermi=1, vhfopt=None, with_j=True, with_k=True):
        from pyscf.df import mdf_jk
        return mdf_jk.get_jk(self, mol, dm, hermi, vhfopt, with_j, with_k)

    def update_mf(self, mf):
        from pyscf.df import mdf_jk
        return mdf_jk.density_fit(mf, self.auxbasis, self.gs, with_df=self)

    def update_mc(self):
        pass

    def update_mp(self):
        pass

    def update_cc(self):
        pass

    def update(self):
        pass

    def update_mf_(self, mf):
        from pyscf.df import mdf_jk
        def get_j(mol, dm, hermi=1):
            return mdf_jk.get_jk(self, mol, dm, hermi, mf.opt,
                                 with_j=True, with_k=False)[0]
        def get_k(mol, dm, hermi=1):
            return mdf_jk.get_jk(self, mol, dm, hermi, mf.opt,
                                 with_j=False, with_k=True)[1]
        mf.get_j = get_j
        mf.get_k = get_k
        mf.get_jk = lambda mol, dm, hermi=1: mdf_jk.get_jk(self, mol, dm, hermi, mf.opt)
        return mf

    def gen_ft(self):
        '''Fourier transformation wrapper'''
        pass


def _Lpq_solver(metric, charge_constraint):
    def solve_Lpq(mol, auxmol):
        if metric.upper() == 'S':
            j3c = incore.aux_e2(mol, auxmol, 'cint3c1e_sph', aosym='s2ij')
            j2c = auxmol.intor_symmetric('cint1e_ovlp_sph')
        elif metric.upper() == 'T':
            j3c = incore.aux_e2(mol, auxmol, 'cint3c1e_p2_sph', aosym='s2ij')
            j2c = auxmol.intor_symmetric('cint1e_kin_sph') * 2
        else:  # metric.upper() == 'J'
            j3c = incore.aux_e2(mol, auxmol, 'cint3c2e_sph', aosym='s2ij')
            j2c = incore.fill_2c2e(mol, auxmol)

        if charge_constraint:
            ovlp = lib.pack_tril(mol.intor_symmetric('cint1e_ovlp_sph'))
            ao_loc = auxmol.ao_loc_nr()
            s_index = numpy.hstack([range(ao_loc[i],ao_loc[i+1])
                                    for i,l in enumerate(auxmol._bas[:,ANG_OF]) if l == 0])
            naux = ao_loc[-1]
            b = numpy.hstack((j3c,ovlp.reshape(-1,1)))
            a = numpy.zeros((naux+1,naux+1))
            a[:naux,:naux] = j2c
            a[naux,s_index] = a[s_index,naux] = 1
            try:
                Lpq = scipy.linalg.solve(a, b.T, sym_pos=True)[:naux]
            except scipy.linalg.LinAlgError:
                Lpq = scipy.linalg.solve(a, b.T)[:naux]
        else:
            Lpq = lib.cho_solve(j2c, j3c.T)
        return Lpq
    return solve_Lpq

def _atomic_envs(mol, auxmol, symbol):
    mol1 = copy.copy(mol)
    mol2 = copy.copy(auxmol)
    mol1._atm, mol1._bas, mol1._env = \
            mol1.make_env([[symbol, (0,0,0)]], mol._basis, mol._env[:gto.PTR_ENV_START])
    mol2._atm, mol2._bas, mol2._env = \
        mol2.make_env([[symbol, (0,0,0)]], auxmol._basis, mol._env[:gto.PTR_ENV_START])
    half_sph_norm = numpy.sqrt(.25/numpy.pi)
    for ib in range(len(mol2._bas)):
        l = mol2.bas_angular(ib)
        np = mol2.bas_nprim(ib)
        nc = mol2.bas_nctr(ib)
        es = mol2.bas_exp(ib)
        ptr = mol2._bas[ib,PTR_COEFF]
        cs = mol2._env[ptr:ptr+np*nc].reshape(nc,np).T
        s = numpy.einsum('pi,p->i', cs, gto.mole._gaussian_int(l*2+2, es))
        cs = numpy.einsum('pi,i->pi', cs, half_sph_norm/s)
        mol2._env[ptr:ptr+np*nc] = cs.T.reshape(-1)
    return mol1, mol2

def _make_Lpq_atomic_approx(mol, auxmol, fLpq):
    Lpqdb = {}
    for a in set([mol.atom_symbol(i) for i in range(mol.natm)]):
        mol1, mol2 = _atomic_envs(mol, auxmol, a)
        Lpqdb[a] = fLpq(mol1, mol2)

    nao = mol.nao_nr()
    naux = auxmol.nao_nr()
    Lpq = numpy.zeros((naux, nao*(nao+1)//2))
    i0 = 0
    j0 = 0
    for ia in range(mol.natm):
        atm_Lpq = Lpqdb[mol.atom_symbol(ia)]
        di, dj2 = atm_Lpq.shape
        dj = int(numpy.sqrt(dj2*2))
# idx is the diagonal block of the lower triangular indices
        idx = numpy.hstack([range(i*(i+1)//2+j0,i*(i+1)//2+i+1)
                            for i in range(j0, j0+dj)])
        Lpq[i0:i0+di,idx] = atm_Lpq
        i0 += di
        j0 += dj
    return Lpq

