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
from pyscf.gto import ATOM_OF, ANG_OF, PTR_COEFF
from pyscf.gto import ft_ao
from pyscf.df import addons
from pyscf.df import incore
from pyscf.df import outcore
from pyscf.df import _ri

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
        self.blockdim = 240
        self._keys = set(self.__dict__.keys())

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

        if not isinstance(self._cderi, str):
            if isinstance(self._cderi_file, str):
                self._cderi = self._cderi_file
            else:
                self._cderi = self._cderi_file.name

        _make_j3c(self, mol, auxmol)
        return self

    def load(self, key):
        return addons.load(self._cderi, key)

    def pw_loop(self, mol, auxmol, gs, shls_slice=None, max_memory=2000):
        '''Plane wave part'''
        if isinstance(gs, int):
            gs = [gs]*3
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
        coulG = .5/numpy.pi**2 * kws / kk

        if shls_slice is None:
            ni = nj = mol.nao_nr()
        else:
            ao_loc = mol.ao_loc_nr()
            ni = ao_loc[shls_slice[1]] - ao_loc[shls_slice[0]]
            nj = ao_loc[shls_slice[3]] - ao_loc[shls_slice[2]]
        nij = ni * nj
        blksize = min(max(16, int(max_memory*1e6*.7/16/nij)), 16384)
        sublk = max(16,int(blksize//4))
        pqkRbuf = numpy.empty(nij*sublk)
        pqkIbuf = numpy.empty(nij*sublk)
        LkRbuf = numpy.empty(naux*sublk)
        LkIbuf = numpy.empty(naux*sublk)

        for p0, p1 in lib.prange(0, coulG.size, blksize):
            aoao = ft_ao.ft_aopair(mol, Gv[p0:p1], shls_slice, 's1',
                                   Gvbase, gxyz[p0:p1], nxyz)
            aoaux = ft_ao.ft_ao(auxmol, Gv[p0:p1], None, Gvbase, gxyz[p0:p1], nxyz)

            for i0, i1 in lib.prange(0, p1-p0, sublk):
                nG = i1 - i0
                pqkR = numpy.ndarray((ni,nj,nG), buffer=pqkRbuf)
                pqkI = numpy.ndarray((ni,nj,nG), buffer=pqkIbuf)
                LkR = numpy.ndarray((naux,nG), buffer=LkRbuf)
                LkI = numpy.ndarray((naux,nG), buffer=LkIbuf)
                pqkR[:] = aoao[i0:i1].real.transpose(1,2,0)
                pqkI[:] = aoao[i0:i1].imag.transpose(1,2,0)
                LkR [:] = aoaux[i0:i1].real.T
                LkI [:] = aoaux[i0:i1].imag.T
                yield (pqkR.reshape(-1,nG), LkR, pqkI.reshape(-1,nG), LkI,
                       coulG[p0+i0:p0+i1])
            aoao = aoaux = None

    def sr_loop(self, mol, auxmol, max_memory=2000):
        '''Short range part'''
        with addons.load(self._cderi, 'j3c') as j3c:
            with addons.load(self._cderi, 'Lpq') as Lpq:
                naoaux = j3c.shape[0]
                for b0, b1 in self.prange(0, naoaux, self.blockdim):
                    yield (numpy.asarray(Lpq[b0:b1], order='C'),
                           numpy.asarray(j3c[b0:b1], order='C'))

    def prange(self, start, end, step):
        for i in range(start, end, step):
            yield i, min(i+step, end)

    def get_jk(self, dm, hermi=1, vhfopt=None, with_j=True, with_k=True):
        from pyscf.df import mdf_jk
        return mdf_jk.get_jk(self, dm, hermi, vhfopt, with_j, with_k)

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
            return mdf_jk.get_jk(self, dm, hermi, mf.opt, with_j=True, with_k=False)[0]
        def get_k(mol, dm, hermi=1):
            return mdf_jk.get_jk(self, dm, hermi, mf.opt, with_j=False, with_k=True)[1]
        mf.get_j = get_j
        mf.get_k = get_k
        mf.get_jk = lambda mol, dm, hermi=1: mdf_jk.get_jk(self, dm, hermi, mf.opt)
        return mf

    def gen_ft(self):
        '''Fourier transformation wrapper'''
        pass


def _make_Lpq(mydf, mol, auxmol):
    atm, bas, env, ao_loc = incore._env_and_aoloc('cint3c1e_sph', mol, auxmol)
    nao = ao_loc[mol.nbas]
    naux = ao_loc[-1] - nao
    nao_pair = nao * (nao+1) // 2

    if mydf.metric.upper() == 'S':
        intor = 'cint3c1e_sph'
        s_aux = auxmol.intor_symmetric('cint1e_ovlp_sph')
    elif mydf.metric.upper() == 'T':
        intor = 'cint3c1e_p2_sph'
        s_aux = auxmol.intor_symmetric('cint1e_kin_sph') * 2
    else:  # metric.upper() == 'J'
        intor = 'cint3c2e_sph'
        s_aux = incore.fill_2c2e(mol, auxmol)
    cintopt = gto.moleintor.make_cintopt(atm, bas, env, intor)

    if mydf.charge_constraint:
        ovlp = lib.pack_tril(mol.intor_symmetric('cint1e_ovlp_sph'))

        aux_loc = auxmol.ao_loc_nr()
        s_index = numpy.hstack([range(aux_loc[i],aux_loc[i+1])
                                for i,l in enumerate(auxmol._bas[:,ANG_OF]) if l == 0])
        a = numpy.zeros((naux+1,naux+1))
        a[:naux,:naux] = s_aux
        a[naux,s_index] = a[s_index,naux] = 1
        try:
            cd = scipy.linalg.cho_factor(a)
            def solve(Lpq):
                return scipy.linalg.cho_solve(cd, Lpq)
        except scipy.linalg.LinAlgError:
            def solve(Lpq):
                return scipy.linalg.solve(a, Lpq)
    else:
        cd = scipy.linalg.cho_factor(s_aux)
        def solve(Lpq):
            return scipy.linalg.cho_solve(cd, Lpq, overwrite_b=True)

    def get_Lpq(shls_slice, col0, col1, buf):
        # Be cautious here, _ri.nr_auxe2 assumes buf in F-order
        Lpq = _ri.nr_auxe2(intor, atm, bas, env, shls_slice, ao_loc,
                           's2ij', 1, cintopt, buf).T
        if mydf.charge_constraint:
            Lpq = numpy.ndarray(shape=(naux+1,col1-col0), buffer=buf)
            Lpq[naux,:] = ovlp[col0:col1]
            Lpq1 = solve(Lpq)
            assert(Lpq1.flags.f_contiguous)
            lib.transpose(Lpq1.T, out=Lpq)
            return Lpq[:naux]
        else:
            return solve(Lpq)
    return get_Lpq

def _atomic_envs(mol, auxmol, atm_id):
    mol1 = copy.copy(mol)
    mol2 = copy.copy(auxmol)
    mol1._atm = mol._atm   [atm_id:atm_id+1]
    mol2._atm = auxmol._atm[atm_id:atm_id+1]
    mol1._bas = numpy.copy(mol._bas   [mol._bas   [:,ATOM_OF] == atm_id])
    mol2._bas = numpy.copy(auxmol._bas[auxmol._bas[:,ATOM_OF] == atm_id])
    mol1._bas[:,ATOM_OF] = 0
    mol2._bas[:,ATOM_OF] = 0
    return mol1, mol2

def _make_Lpq_atomic_approx(mydf, mol, auxmol):
    Lpqdb = []
    for ia in range(mol.natm):
        mol1, auxmol1 = _atomic_envs(mol, auxmol, ia)
        nao = mol1.nao_nr()
        nao_pair = nao * (nao+1) // 2
        naux = auxmol1.nao_nr()
        buf = numpy.empty((naux+1,nao_pair))
        get_Lpq = _make_Lpq(mydf, mol1, auxmol1)
        shls_slice = (0, mol1.nbas, 0, mol1.nbas, mol1.nbas, mol1.nbas+auxmol1.nbas)
        Lpqdb.append(get_Lpq(shls_slice, 0, nao_pair, buf))

    naux = auxmol.nao_nr()
    def get_Lpq(shls_slice, col0, col1, buf=None):
        Lpq = numpy.ndarray((naux,col1-col0), buffer=buf)
        Lpq[:] = 0
        k0 = 0
        j0 = 0
        for ia in range(mol.natm):
            if j0*(j0+1)//2 >= col1:
                break
            atm_Lpq = Lpqdb[ia]
            dk, dj2 = atm_Lpq.shape
            dj = int(numpy.sqrt(dj2*2))
# idx is the diagonal block of the lower triangular indices
            idx = numpy.hstack([range(i*(i+1)//2+j0,i*(i+1)//2+i+1)
                                for i in range(j0, j0+dj)])
            mask = (col0 <= idx) & (idx < col1)
            idx = idx[mask] - col0
            Lpq[k0:k0+dk,idx] = atm_Lpq[:,mask]
            k0 += dk
            j0 += dj
        return Lpq
    return get_Lpq

def _make_j3c(mydf, mol, auxmol):
    log = logger.Logger(mydf.stdout, mydf.verbose)
    atm, bas, env, ao_loc = incore._env_and_aoloc('cint3c2e_sph', mol, auxmol)
    nao = ao_loc[mol.nbas]
    naux = ao_loc[-1] - nao
    nao_pair = nao * (nao+1) // 2
    cintopt = gto.moleintor.make_cintopt(atm, bas, env, 'cint3c2e_sph')
    if mydf.approx_sr_level == 0:
        get_Lpq = _make_Lpq(mydf, mol, auxmol)
    else:
        get_Lpq = _make_Lpq_atomic_approx(mydf, mol, auxmol)

    feri = h5py.File(mydf._cderi)
    chunks = (min(256,naux), min(256,nao_pair)) # 512K
    feri.create_dataset('j3c', (naux,nao_pair), 'f8', chunks=chunks)
    feri.create_dataset('Lpq', (naux,nao_pair), 'f8', chunks=chunks)

    def save(label, dat, col0, col1):
        feri[label][:,col0:col1] = dat

    Gv, Gvbase, kws = non_uniform_kgrids(mydf.gs)
    nxyz = [i*2 for i in mydf.gs]
    gxyz = lib.cartesian_prod([range(i) for i in nxyz])
    kk = numpy.einsum('ki,ki->k', Gv, Gv)
    idx = numpy.argsort(kk)[::-1]
    kk = kk[idx]
    Gv = Gv[idx]
    kws = kws[idx]
    gxyz = gxyz[idx]
    coulG = .5/numpy.pi**2 * kws / kk

    aoaux = ft_ao.ft_ao(auxmol, Gv, None, Gvbase, gxyz, nxyz)
    kLR = numpy.asarray(aoaux.real, order='C')
    kLI = numpy.asarray(aoaux.imag, order='C')
    j2c = auxmol.intor('cint2c2e_sph', hermi=1).T  # .T to C-ordr
    lib.dot(kLR.T*coulG, kLR, -1, j2c, 1)
    lib.dot(kLI.T*coulG, kLI, -1, j2c, 1)

    kLR *= coulG.reshape(-1,1)
    kLI *= coulG.reshape(-1,1)
    aoaux = coulG = kk = kws = idx = None

    max_memory = max(2000, mydf.max_memory-lib.current_memory()[0])
    buflen = min(max(int(max_memory*.3*1e6/8/naux), 1), nao_pair)
    shranges = outcore._guess_shell_ranges(mol, buflen, 's2ij')
    buflen = max([x[2] for x in shranges])
    blksize = max(16, int(max_memory*.15*1e6/16/buflen))
    pqkbuf = numpy.empty(buflen*blksize)
    bufs1 = numpy.empty((buflen*naux))
    # bufs2 holds either Lpq and ft_aopair
    bufs2 = numpy.empty(max(buflen*(naux+1),buflen*blksize*2)) # *2 for cmplx

    col1 = 0
    for istep, sh_range in enumerate(shranges):
        log.debug('int3c2e [%d/%d], AO [%d:%d], ncol = %d', \
                  istep+1, len(shranges), *sh_range)
        bstart, bend, ncol = sh_range
        col0, col1 = col1, col1+ncol
        shls_slice = (bstart, bend, 0, bend, mol.nbas, mol.nbas+auxmol.nbas)

        Lpq = get_Lpq(shls_slice, col0, col1, bufs2)
        save('Lpq', Lpq, col0, col1)

        j3c = _ri.nr_auxe2('cint3c2e_sph', atm, bas, env, shls_slice, ao_loc,
                           's2ij', 1, cintopt, bufs1)
        j3c = j3c.T  # -> (L|pq) in C-order
        lib.dot(j2c, Lpq, -.5, j3c, 1)
        Lpq = None

        for p0, p1 in lib.prange(0, Gv.shape[0], blksize):
            aoao = ft_ao.ft_aopair(mol, Gv[p0:p1], shls_slice[:4], 's2',
                                   Gvbase, gxyz[p0:p1], nxyz, buf=bufs2)
            nG = p1 - p0
            pqkR = numpy.ndarray((ncol,nG), buffer=pqkbuf)
            pqkR[:] = aoao.real.T
            lib.dot(kLR[p0:p1].T, pqkR.T, -1, j3c, 1)
            pqkI = numpy.ndarray((ncol,nG), buffer=pqkbuf)
            pqkI[:] = aoao.imag.T
            lib.dot(kLI[p0:p1].T, pqkI.T, -1, j3c, 1)
            aoao = aoaux = None
        save('j3c', j3c, col0, col1)

    feri.close()
