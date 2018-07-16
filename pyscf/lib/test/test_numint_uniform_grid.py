import ctypes
import unittest
import numpy
import scipy.linalg

from pyscf import lib
from pyscf.pbc import gto
from pyscf.dft.numint import libdft
from pyscf.pbc.dft import gen_grid

PTR_EXPDROP = 16
def eval_mat(cell, weights, shls_slice=None, comp=1, hermi=0,
             xctype='LDA', kpts=None, offset=None, submesh=None):
    assert(all(cell._bas[:,gto.mole.NPRIM_OF] == 1))
    atm, bas, env = gto.conc_env(cell._atm, cell._bas, cell._env,
                                 cell._atm, cell._bas, cell._env)
    env[PTR_EXPDROP] = 1e-18
    ao_loc = gto.moleintor.make_loc(bas, 'cart')
    if shls_slice is None:
        shls_slice = (0, cell.nbas, 0, cell.nbas)
    i0, i1, j0, j1 = shls_slice
    j0 += cell.nbas
    j1 += cell.nbas
    naoi = ao_loc[i1] - ao_loc[i0]
    naoj = ao_loc[j1] - ao_loc[j0]

    if cell.dimension > 0:
        Ls = numpy.asarray(cell.get_lattice_Ls(), order='C')
    else:
        Ls = numpy.zeros((1,3))
    nimgs = len(Ls)

    mesh = cell.mesh
    weights = numpy.asarray(weights, order='C')
    if xctype.upper() == 'LDA':
        weights = weights.reshape(-1, numpy.prod(mesh))
    elif xctype.upper() == 'GGA':
        if hermi == 1:
            raise RuntimeError('hermi=1 is not supported by GGA functional')
        weights = weights.reshape(-1, 4, numpy.prod(mesh))
    else:
        raise NotImplementedError
    n_mat = weights.shape[0]

    a = cell.lattice_vectors()
    b = numpy.linalg.inv(a.T)
    if offset is None:
        offset = (0, 0, 0)
    if submesh is None:
        submesh = mesh

    if abs(a-numpy.diag(a.diagonal())).max() < 1e-12:
        lattice_type = '_orth'
    else:
        lattice_type = '_nonorth'
    eval_fn = 'NUMINTeval_' + xctype.lower() + lattice_type
    drv = libdft.NUMINT_fill2c
    def make_mat(weights):
        if comp == 1:
            mat = numpy.zeros((nimgs,naoj,naoi))
        else:
            mat = numpy.zeros((nimgs,comp,naoj,naoi))
        drv(getattr(libdft, eval_fn),
            weights.ctypes.data_as(ctypes.c_void_p),
            mat.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(comp), ctypes.c_int(hermi),
            (ctypes.c_int*4)(i0, i1, j0, j1),
            ao_loc.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_double(numpy.log(cell.precision)),
            ctypes.c_int(cell.dimension),
            ctypes.c_int(nimgs),
            Ls.ctypes.data_as(ctypes.c_void_p),
            a.ctypes.data_as(ctypes.c_void_p),
            b.ctypes.data_as(ctypes.c_void_p),
            (ctypes.c_int*3)(*offset), (ctypes.c_int*3)(*submesh),
            (ctypes.c_int*3)(*mesh),
            atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(atm)),
            bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(bas)),
            env.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(env)))
        return mat

    out = []
    for wv in weights:
        if cell.dimension == 0:
            mat = numpy.rollaxis(make_mat(wv)[0], -1, -2)
        elif kpts is None:
            mat = numpy.rollaxis(make_mat(wv).sum(axis=0), -1, -2)
        else:
            mat = make_mat(wv)
            mat_shape = mat.shape
            expkL = numpy.exp(1j*kpts.reshape(-1,3).dot(Ls.T))
            mat = numpy.dot(expkL, mat.reshape(nimgs,-1))
            mat = numpy.rollaxis(mat.reshape((-1,)+mat_shape[1:]), -1, -2)
        out.append(mat)

    if n_mat == 1:
        out = out[0]
    return out

def uncontract(cell):
    pcell, contr_coeff = cell.to_uncontracted_cartesian_basis()
    return pcell, scipy.linalg.block_diag(*contr_coeff)

def eval_rho(cell, dm, shls_slice=None, hermi=0, xctype='LDA', kpts=None,
             offset=None, submesh=None):
    assert(all(cell._bas[:,gto.mole.NPRIM_OF] == 1))
    atm, bas, env = gto.conc_env(cell._atm, cell._bas, cell._env,
                                 cell._atm, cell._bas, cell._env)
    env[PTR_EXPDROP] = 1e-18
    ao_loc = gto.moleintor.make_loc(bas, 'cart')
    if shls_slice is None:
        shls_slice = (0, cell.nbas, 0, cell.nbas)
    i0, i1, j0, j1 = shls_slice
    if hermi:
        assert(i0 == j0 and i1 == j1)
    j0 += cell.nbas
    j1 += cell.nbas
    naoi = ao_loc[i1] - ao_loc[i0]
    naoj = ao_loc[j1] - ao_loc[j0]
    assert(dm.shape[-2:] == (naoi, naoj))

    if cell.dimension > 0:
        Ls = numpy.asarray(cell.get_lattice_Ls(), order='C')
    else:
        Ls = numpy.zeros((1,3))

    if cell.dimension == 0 or kpts is None:
        nkpts, nimgs = 1, Ls.shape[0]
        dm = dm.reshape(-1,1,naoi,naoj).transpose(0,1,3,2)
    else:
        expkL = numpy.exp(1j*kpts.reshape(-1,3).dot(Ls.T))
        nkpts, nimgs = expkL.shape
        dm = dm.reshape(-1,nkpts,naoi,naoj).transpose(0,1,3,2)
    n_dm = dm.shape[0]

    a = cell.lattice_vectors()
    b = numpy.linalg.inv(a.T)
    mesh = numpy.asarray(cell.mesh, dtype=numpy.int32)
    if offset is None:
        offset = (0, 0, 0)
    if submesh is None:
        submesh = mesh

    if abs(a-numpy.diag(a.diagonal())).max() < 1e-12:
        lattice_type = '_orth'
    else:
        lattice_type = '_nonorth'
    if xctype.upper() == 'LDA':
        comp = 1
    elif xctype.upper() == 'GGA':
        if hermi == 1:
            raise RuntimeError('hermi=1 is not supported by GGA functional')
        comp = 4
    else:
        raise NotImplementedError('meta-GGA')
    eval_fn = 'NUMINTrho_' + xctype.lower() + lattice_type
    drv = libdft.NUMINT_rho_drv
    def make_rho(dm):
        if comp == 1:
            rho = numpy.zeros((numpy.prod(submesh)))
        else:
            rho = numpy.zeros((comp, numpy.prod(submesh)))
        drv(getattr(libdft, eval_fn),
            rho.ctypes.data_as(ctypes.c_void_p),
            dm.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(comp), ctypes.c_int(hermi),
            (ctypes.c_int*4)(i0, i1, j0, j1),
            ao_loc.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_double(numpy.log(cell.precision)),
            ctypes.c_int(cell.dimension),
            ctypes.c_int(nimgs),
            Ls.ctypes.data_as(ctypes.c_void_p),
            a.ctypes.data_as(ctypes.c_void_p),
            b.ctypes.data_as(ctypes.c_void_p),
            (ctypes.c_int*3)(*offset), (ctypes.c_int*3)(*submesh),
            mesh.ctypes.data_as(ctypes.c_void_p),
            atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(atm)),
            bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(bas)),
            env.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(env)))
        return rho

    rho = []
    for dm_i in dm:
        if cell.dimension == 0:
            # make a copy because the dm may be overwritten in the
            # NUMINT_rho_drv inplace
            rho.append(make_rho(numpy.array(dm_i, order='C', copy=True)))
        elif kpts is None:
            rho.append(make_rho(numpy.repeat(dm_i, nimgs, axis=0)))
        else:
            dm_i = numpy.dot(expkL.T, dm_i.reshape(nkpts,-1)).reshape(nimgs,naoj,naoi)
            dmR = numpy.asarray(dm_i.real, order='C')
            dmI = numpy.asarray(dm_i.imag, order='C')

            has_imag = (hermi == 0 and abs(dmI).max() > 1e-8)
            if (has_imag and xctype.upper() == 'LDA' and
                naoi == naoj and
# For hermitian density matrices, the anti-symmetry character of the imaginary
# part of the density matrices can be found by rearranging the repeated images.
                abs(dmI + dmI[::-1].transpose(0,2,1)).max() < 1e-8):
                has_imag = False

            if has_imag:
                rho.append(make_rho(dmR) + make_rho(dmI)*1j)
            else:
                rho.append(make_rho(dmR))
            dmR = dmI = None

    if n_dm == 1:
        rho = rho[0]
    return rho

numpy.random.seed(2)
cell_orth = gto.M(atom='H1 1 1 0; H2 0 0 1',
                  basis={'H1':[[0, ( 1, 1, .1), (.5, .1, 1)],
                               [1, (.8, 1, .2), (.3, .2, 1)]],
                         'H2':[[0, (.9, .6, .3), (.4, .1, 1)],
                               [2, (.7, .8, .2), (.2, .2, 1)]]},
                  unit='B',
                  mesh=[7,6,5],
                  a=numpy.eye(3)*4)

mol_orth = cell_orth.copy()
mol_orth.dimension = 0

cell_north = gto.M(atom='H1 1 1 0; H2 0 0 1',
                   basis={'H1':[[0, ( 1, 1, .1), (.5, .1, 1)],
                                [1, (.8, 1, .2), (.3, .2, 1)]],
                          'H2':[[0, (.9, .6, .3), (.4, .1, 1)],
                                [2, (.7, .8, .2), (.2, .2, 1)]]},
                   unit='B',
                   mesh=[7,6,5],
                   a=numpy.eye(3)*4+numpy.random.rand(3,3)*.3)

mol_north = cell_north.copy()
mol_north.dimension = 0

vxc = numpy.random.random((4,numpy.prod(cell_orth.mesh)))
kpts = (numpy.random.random((2,3))-.5) * 2
nkpts = len(kpts)
nao = cell_orth.nao
dm = numpy.random.random((nkpts,nao,nao))
dm = dm + dm.transpose(0,2,1)

# FIXME when kpts != 0 and dm is not hermitian
dm_kpts = cell_orth.pbc_intor('int1e_ovlp', kpts=kpts)

grids_orth = gen_grid.UniformGrids(cell_orth)
grids_north = gen_grid.UniformGrids(cell_north)

ao_kpts_orth = cell_orth.pbc_eval_gto('GTOval_sph_deriv1', grids_orth.coords, kpts=kpts)
ao_kpts_north = cell_north.pbc_eval_gto('GTOval_sph_deriv1', grids_north.coords, kpts=kpts)
ao_orth = mol_orth.eval_gto('GTOval_sph_deriv1', grids_orth.coords, kpts=kpts)
ao_north = mol_north.eval_gto('GTOval_sph_deriv1', grids_north.coords, kpts=kpts)
ao_gamma_orth = cell_orth.pbc_eval_gto('GTOval_sph_deriv1', grids_orth.coords)
ao_gamma_north = cell_north.pbc_eval_gto('GTOval_sph_deriv1', grids_north.coords)

def tearDownModule():
    global cell_orth, cell_north, mol_orth, mol_north
    del cell_orth, cell_north, mol_orth, mol_north

class KnownValues(unittest.TestCase):
    def test_pbc_orth_lda_ints(self):
        ref = numpy.array([numpy.einsum('gi,gj,g->ij', ao[0].conj(), ao[0], vxc[0])
                           for ao in ao_kpts_orth])
        pcell, contr_coeff = uncontract(cell_orth)
        out = eval_mat(pcell, vxc[0], hermi=0, kpts=kpts)
        out = numpy.einsum('pi,kpq,qj->kij', contr_coeff, out, contr_coeff)
        self.assertAlmostEqual(abs(out-ref).max(), 0, 9)

        out = eval_mat(pcell, vxc[0], hermi=1, kpts=kpts)
        out = numpy.einsum('pi,kpq,qj->kij', contr_coeff, out, contr_coeff)
        self.assertAlmostEqual(abs(out-ref).max(), 0, 9)

    def test_pbc_orth_gga_ints(self):
        ref = numpy.array([numpy.einsum('ngi,gj,ng->ij', ao.conj(), ao[0], vxc)
                           for ao in ao_kpts_orth])
        pcell, contr_coeff = uncontract(cell_orth)
        out = eval_mat(pcell, vxc, xctype='GGA', hermi=0, kpts=kpts)
        out = numpy.einsum('pi,kpq,qj->kij', contr_coeff, out, contr_coeff)
        self.assertAlmostEqual(abs(out-ref).max(), 0, 9)

        #out = eval_mat(pcell, vxc, xctype='GGA', hermi=1, kpts=kpts)
        self.assertRaises(RuntimeError, eval_mat, pcell, vxc, xctype='GGA', hermi=1, kpts=kpts)
        #out = numpy.einsum('pi,kpq,qj->kij', contr_coeff, out, contr_coeff)
        #self.assertAlmostEqual(abs(out-ref).max(), 0, 9)

    def test_pbc_orth_overlap(self):
        ref = cell_orth.pbc_intor('int1e_ovlp', kpts=kpts)
        pcell, contr_coeff = uncontract(cell_orth)
        pcell.mesh = [9]*3
        w = gen_grid.UniformGrids(pcell).weights
        out = eval_mat(pcell, w, hermi=0, kpts=kpts)
        out = numpy.einsum('pi,kpq,qj->kij', contr_coeff, out, contr_coeff)
        self.assertAlmostEqual(abs(out-ref).max(), 0, 9)

        out = eval_mat(pcell, w, hermi=1, kpts=kpts)
        out = numpy.einsum('pi,kpq,qj->kij', contr_coeff, out, contr_coeff)
        self.assertAlmostEqual(abs(out-ref).max(), 0, 9)

    def test_pbc_nonorth_lda_ints(self):
        ref = numpy.array([numpy.einsum('gi,gj,g->ij', ao[0].conj(), ao[0], vxc[0])
                           for ao in ao_kpts_north])
        pcell, contr_coeff = uncontract(cell_north)
        out = eval_mat(pcell, vxc[0], hermi=0, kpts=kpts)
        out = numpy.einsum('pi,kpq,qj->kij', contr_coeff, out, contr_coeff)
        self.assertAlmostEqual(abs(out-ref).max(), 0, 9)

        out = eval_mat(pcell, vxc[0], hermi=1, kpts=kpts)
        out = numpy.einsum('pi,kpq,qj->kij', contr_coeff, out, contr_coeff)
        self.assertAlmostEqual(abs(out-ref).max(), 0, 9)

    def test_pbc_nonorth_gga_ints(self):
        ref = numpy.array([numpy.einsum('ngi,gj,ng->ij', ao.conj(), ao[0], vxc)
                           for ao in ao_kpts_north])
        pcell, contr_coeff = uncontract(cell_north)
        out = eval_mat(pcell, vxc, xctype='GGA', hermi=0, kpts=kpts)
        out = numpy.einsum('pi,kpq,qj->kij', contr_coeff, out, contr_coeff)
        self.assertAlmostEqual(abs(out-ref).max(), 0, 9)

        #out = eval_mat(pcell, vxc, xctype='GGA', hermi=1, kpts=kpts)
        self.assertRaises(RuntimeError, eval_mat, pcell, vxc, xctype='GGA', hermi=1, kpts=kpts)
        #out = numpy.einsum('pi,kpq,qj->kij', contr_coeff, out, contr_coeff)
        #self.assertAlmostEqual(abs(out-ref).max(), 0, 9)

    def test_pbc_nonorth_overlap(self):
        ref = cell_north.pbc_intor('int1e_ovlp', kpts=kpts)
        pcell, contr_coeff = uncontract(cell_north)
        pcell.mesh = [9]*3
        w = gen_grid.UniformGrids(pcell).weights
        out = eval_mat(pcell, w, hermi=0, kpts=kpts)
        out = numpy.einsum('pi,kpq,qj->kij', contr_coeff, out, contr_coeff)
        self.assertAlmostEqual(abs(out-ref).max(), 0, 9)

        out = eval_mat(pcell, w, hermi=1, kpts=kpts)
        out = numpy.einsum('pi,kpq,qj->kij', contr_coeff, out, contr_coeff)
        self.assertAlmostEqual(abs(out-ref).max(), 0, 9)

    def test_orth_lda_ints(self):
        ao = ao_orth
        ref = numpy.einsum('gi,gj,g->ij', ao[0], ao[0], vxc[0])
        pcell, contr_coeff = uncontract(mol_orth)
        out = eval_mat(pcell, vxc[0], hermi=0)
        out = numpy.einsum('pi,pq,qj->ij', contr_coeff, out, contr_coeff)
        self.assertAlmostEqual(abs(out-ref).max(), 0, 9)

        out = eval_mat(pcell, vxc[0], hermi=1)
        out = numpy.einsum('pi,pq,qj->ij', contr_coeff, out, contr_coeff)
        self.assertAlmostEqual(abs(out-ref).max(), 0, 9)

    def test_orth_gga_ints(self):
        ao = ao_orth
        ref = numpy.einsum('ngi,gj,ng->ij', ao, ao[0], vxc)
        pcell, contr_coeff = uncontract(mol_orth)
        out = eval_mat(pcell, vxc, xctype='GGA', hermi=0)
        out = numpy.einsum('pi,pq,qj->ij', contr_coeff, out, contr_coeff)
        self.assertAlmostEqual(abs(out-ref).max(), 0, 9)

    def test_nonorth_lda_ints(self):
        ao = ao_north
        ref = numpy.einsum('gi,gj,g->ij', ao[0], ao[0], vxc[0])
        pcell, contr_coeff = uncontract(mol_north)
        out = eval_mat(pcell, vxc[0])
        out = numpy.einsum('pi,pq,qj->ij', contr_coeff, out, contr_coeff)
        self.assertAlmostEqual(abs(out-ref).max(), 0, 9)

        out = eval_mat(pcell, vxc[0], hermi=1)
        out = numpy.einsum('pi,pq,qj->ij', contr_coeff, out, contr_coeff)
        self.assertAlmostEqual(abs(out-ref).max(), 0, 9)

    def test_nonorth_gga_ints(self):
        ao = ao_north
        ref = numpy.einsum('ngi,gj,ng->ij', ao, ao[0], vxc)
        pcell, contr_coeff = uncontract(mol_north)
        out = eval_mat(pcell, vxc, xctype='GGA', hermi=0)
        out = numpy.einsum('pi,pq,qj->ij', contr_coeff, out, contr_coeff)
        self.assertAlmostEqual(abs(out-ref).max(), 0, 9)

    def test_pbc_orth_lda_rho(self):
        ref = sum([numpy.einsum('gi,ij,gj->g', ao[0], dm[k], ao[0].conj())
                   for k,ao in enumerate(ao_kpts_orth)])
        pcell, contr_coeff = uncontract(cell_orth)
        dm1 = numpy.einsum('pi,kij,qj->kpq', contr_coeff, dm, contr_coeff)
        out = eval_rho(pcell, dm1, kpts=kpts)
        self.assertTrue(out.dtype == numpy.double)
        self.assertAlmostEqual(abs(out-ref).max(), 0, 9)

        out = eval_rho(pcell, dm1, hermi=1, kpts=kpts)
        self.assertTrue(out.dtype == numpy.double)
        self.assertAlmostEqual(abs(out-ref).max(), 0, 9)

    def test_pbc_orth_lda_rho_kpts(self):
        ref = sum([numpy.einsum('gi,ij,gj->g', ao[0], dm_kpts[k], ao[0].conj())
                   for k,ao in enumerate(ao_kpts_orth)])
        pcell, contr_coeff = uncontract(cell_orth)
        dm1 = numpy.einsum('pi,kij,qj->kpq', contr_coeff, dm_kpts, contr_coeff)
        out = eval_rho(pcell, dm1, hermi=0, kpts=kpts)
        self.assertTrue(out.dtype == numpy.double)
        self.assertAlmostEqual(abs(out-ref).max(), 0, 9)

        out = eval_rho(pcell, dm1, hermi=1, kpts=kpts)
        self.assertTrue(out.dtype == numpy.double)
        self.assertAlmostEqual(abs(out-ref).max(), 0, 9)

    def test_pbc_orth_gga_rho(self):
        ao = ao_gamma_orth
        ref = numpy.einsum('ngi,ij,gj->ng', ao, dm[0], ao[0].conj())
        pcell, contr_coeff = uncontract(cell_orth)
        dm1 = numpy.einsum('pi,ij,qj->pq', contr_coeff, dm[0], contr_coeff)
        out = eval_rho(pcell, dm1, xctype='GGA')
        self.assertTrue(out.dtype == numpy.double)
        self.assertAlmostEqual(abs(out-ref).max(), 0, 9)

        ref = sum([numpy.einsum('ngi,ij,gj->ng', ao, dm[k], ao[0].conj())
                   for k,ao in enumerate(ao_kpts_orth)])
        dm1 = numpy.einsum('pi,kij,qj->kpq', contr_coeff, dm, contr_coeff)
        out = eval_rho(pcell, dm1, kpts=kpts, xctype='GGA')
        self.assertAlmostEqual(abs(out-ref).max(), 0, 9)

    def test_pbc_orth_gga_rho_kpts(self):
        pcell, contr_coeff = uncontract(cell_orth)
        ref = sum([numpy.einsum('ngi,ij,gj->ng', ao, dm_kpts[k], ao[0].conj())
                   for k,ao in enumerate(ao_kpts_orth)])
        dm1 = numpy.einsum('pi,kij,qj->kpq', contr_coeff, dm_kpts, contr_coeff)
        out = eval_rho(pcell, dm1, kpts=kpts, xctype='GGA')
        self.assertAlmostEqual(abs(out-ref).max(), 0, 9)

    def test_pbc_nonorth_lda_rho(self):
        ref = sum([numpy.einsum('gi,ij,gj->g', ao[0], dm[k], ao[0].conj())
                   for k,ao in enumerate(ao_kpts_north)])
        pcell, contr_coeff = uncontract(cell_north)
        dm1 = numpy.einsum('pi,kij,qj->kpq', contr_coeff, dm, contr_coeff)
        out = eval_rho(pcell, dm1, kpts=kpts)
        self.assertAlmostEqual(abs(out-ref).max(), 0, 9)

        out = eval_rho(pcell, dm1, hermi=1, kpts=kpts)
        self.assertTrue(out.dtype == numpy.double)
        self.assertAlmostEqual(abs(out-ref).max(), 0, 9)

    def test_pbc_nonorth_gga_rho(self):
        ao = ao_gamma_north
        ref = numpy.einsum('ngi,ij,gj->ng', ao, dm[0], ao[0].conj())
        pcell, contr_coeff = uncontract(cell_north)
        dm1 = numpy.einsum('pi,ij,qj->pq', contr_coeff, dm[0], contr_coeff)
        out = eval_rho(pcell, dm1, xctype='GGA')
        self.assertAlmostEqual(abs(out-ref).max(), 0, 9)

        ref = sum([numpy.einsum('ngi,ij,gj->ng', ao, dm[k], ao[0].conj())
                   for k,ao in enumerate(ao_kpts_north)])
        dm1 = numpy.einsum('pi,kij,qj->kpq', contr_coeff, dm, contr_coeff)
        out = eval_rho(pcell, dm1, kpts=kpts, xctype='GGA')
        self.assertAlmostEqual(abs(out-ref).max(), 0, 9)

    def test_pbc_nonorth_gga_rho_kpts(self):
        pcell, contr_coeff = uncontract(cell_north)
        ref = sum([numpy.einsum('ngi,ij,gj->ng', ao, dm_kpts[k], ao[0].conj())
                   for k,ao in enumerate(ao_kpts_north)])
        dm1 = numpy.einsum('pi,kij,qj->kpq', contr_coeff, dm_kpts, contr_coeff)
        out = eval_rho(pcell, dm1, kpts=kpts, xctype='GGA')
        self.assertAlmostEqual(abs(out-ref).max(), 0, 9)

    def test_orth_lda_rho(self):
        ao = ao_orth
        ref = numpy.einsum('gi,ij,gj->g', ao[0], dm[0], ao[0].conj())
        pcell, contr_coeff = uncontract(mol_orth)
        dm1 = numpy.einsum('pi,ij,qj->pq', contr_coeff, dm[0], contr_coeff)
        out = eval_rho(pcell, dm1)
        self.assertAlmostEqual(abs(out-ref).max(), 0, 9)

        out = eval_rho(pcell, dm1, hermi=1)
        self.assertAlmostEqual(abs(out-ref).max(), 0, 9)

    def test_orth_gga_rho(self):
        ao = ao_orth
        ref = numpy.einsum('ngi,ij,gj->ng', ao, dm[0], ao[0].conj())
        pcell, contr_coeff = uncontract(mol_orth)
        dm1 = numpy.einsum('pi,ij,qj->pq', contr_coeff, dm[0], contr_coeff)
        out = eval_rho(pcell, dm1, xctype='GGA')
        self.assertAlmostEqual(abs(out-ref).max(), 0, 9)

        #out = eval_rho(pcell, dm1, hermi=1, xctype='GGA')
        self.assertRaises(RuntimeError, eval_rho, pcell, dm1, xctype='GGA', hermi=1)
        #self.assertAlmostEqual(abs(out-ref).max(), 0, 9)

    def test_nonorth_lda_rho(self):
        ao = ao_north
        ref = numpy.einsum('gi,ij,gj->g', ao[0], dm[0], ao[0].conj())
        pcell, contr_coeff = uncontract(mol_north)
        dm1 = numpy.einsum('pi,ij,qj->pq', contr_coeff, dm[0], contr_coeff)
        out = eval_rho(pcell, dm1)
        self.assertAlmostEqual(abs(out-ref).max(), 0, 9)

        out = eval_rho(pcell, dm1, hermi=1)
        self.assertAlmostEqual(abs(out-ref).max(), 0, 9)

    def test_nonorth_gga_rho(self):
        ao = ao_north
        ref = numpy.einsum('ngi,ij,gj->ng', ao, dm[0], ao[0].conj())
        pcell, contr_coeff = uncontract(mol_north)
        dm1 = numpy.einsum('pi,ij,qj->pq', contr_coeff, dm[0], contr_coeff)
        out = eval_rho(pcell, dm1, xctype='GGA')
        self.assertAlmostEqual(abs(out-ref).max(), 0, 9)

        #out = eval_rho(pcell, dm1, hermi=1, xctype='GGA')
        self.assertRaises(RuntimeError, eval_rho, pcell, dm1, xctype='GGA', hermi=1)
        #self.assertAlmostEqual(abs(out-ref).max(), 0, 9)


if __name__ == '__main__':
    print("Full Tests for numint_uniform_grid")
    unittest.main()
