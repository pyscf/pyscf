import ctypes
import unittest
import numpy
import scipy.linalg

from pyscf import lib
from pyscf.pbc import gto
from pyscf.dft.numint import libdft
from pyscf.pbc.dft import gen_grid

def eval_mat(cell, weights, shls_slice=None, comp=1, hermi=0,
             xctype='LDA', kpts=None):
    assert(all(cell._bas[:,gto.mole.NPRIM_OF] == 1))
    atm, bas, env = gto.conc_env(cell._atm, cell._bas, cell._env,
                                 cell._atm, cell._bas, cell._env)
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

    weights = numpy.asarray(weights, order='C')
    if xctype.upper() == 'LDA':
        if weights.ndim > 1:
            comp *= weights.shape[0]
    elif xctype.upper() == 'GGA':
        if weights.ndim > 2:
            comp *= weights.shape[1]
    else:
        raise NotImplementedError

    out = numpy.zeros((nimgs,comp,naoj,naoi))

    a = cell.lattice_vectors()
    b = numpy.linalg.inv(a.T)
    mesh = numpy.asarray(cell.mesh, dtype=numpy.int32)

    if abs(a-numpy.diag(a.diagonal())).max() < 1e-12:
        lattice_type = '_orth'
    else:
        lattice_type = '_nonorth'
    eval_fn = 'NUMINTeval_' + xctype.lower() + lattice_type
    drv = libdft.NUMINT_fill2c
    drv(getattr(libdft, eval_fn),
        out.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(comp), ctypes.c_int(hermi),
        (ctypes.c_int*4)(i0, i1, j0, j1),
        ao_loc.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_double(numpy.log(cell.precision)),
        ctypes.c_int(cell.dimension),
        ctypes.c_int(nimgs),
        Ls.ctypes.data_as(ctypes.c_void_p),
        a.ctypes.data_as(ctypes.c_void_p),
        b.ctypes.data_as(ctypes.c_void_p),
        mesh.ctypes.data_as(ctypes.c_void_p),
        weights.ctypes.data_as(ctypes.c_void_p),
        atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(atm)),
        bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(bas)),
        env.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(env)))

    if cell.dimension == 0:
        out = out[0].transpose(0,2,1)
    elif kpts is None:
        out = out.sum(axis=0).transpose(0,2,1)
    else:
        expkL = numpy.exp(1j*kpts.reshape(-1,3).dot(Ls.T))
        out = numpy.dot(expkL, out.reshape(nimgs,-1))
        out = out.reshape(-1,comp,naoj,naoi).transpose(1,0,3,2)

    if comp == 1:
        out = out[0]
    return out

def uncontract(cell):
    pcell, contr_coeff = cell.to_uncontracted_cartesian_basis()
    return pcell, scipy.linalg.block_diag(*contr_coeff)

def eval_rho(cell, dm, shls_slice=None, hermi=0, xctype='LDA', kpts=None):
    assert(all(cell._bas[:,gto.mole.NPRIM_OF] == 1))
    atm, bas, env = gto.conc_env(cell._atm, cell._bas, cell._env,
                                 cell._atm, cell._bas, cell._env)
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

    has_imag = False
    if cell.dimension == 0:
        nkpts = nimgs = 1
        dm = dm.reshape(1,-1,naoi,naoj).transpose(0,1,3,2)
    elif kpts is None:
        nkpts, nimgs = 1, Ls.shape[0]
        dm = dm.reshape(1,-1,naoi,naoj).transpose(0,1,3,2)
        dm = numpy.vstack([dm]*nimgs)
    else:
        # FIXME: has_imag?
        has_imag = (hermi == 0 and abs(kpts).max() > 1e-8)
        dm = dm.reshape(-1,naoi,naoj)
        if xctype.upper() == 'LDA' and abs(dm - dm.transpose(0,2,1).conj()).max() < 1e-9:
            has_imag = False

        expkL = numpy.exp(1j*kpts.reshape(-1,3).dot(Ls.T))
        nkpts, nimgs = expkL.shape
        dm = dm.reshape(-1,nkpts,naoi,naoj).transpose(1,0,3,2)
        dm = numpy.dot(expkL.T, dm.reshape(nkpts,-1)).reshape(nimgs,-1,naoj,naoi)
        if has_imag:
            dm = numpy.concatenate((dm.real, dm.imag), axis=1)
        else:
            dm = dm.real
    dm = numpy.asarray(dm, order='C')
    n_dm = dm.shape[1]

    a = cell.lattice_vectors()
    b = numpy.linalg.inv(a.T)
    mesh = numpy.asarray(cell.mesh, dtype=numpy.int32)
    offset = (0, 0, 0)
    ngrids = mesh

    if abs(a-numpy.diag(a.diagonal())).max() < 1e-12:
        lattice_type = '_orth'
    else:
        lattice_type = '_nonorth'
    if xctype.upper() == 'LDA':
        comp = 1
        rho = numpy.zeros((n_dm, numpy.prod(ngrids)))
    elif xctype.upper() == 'GGA':
        comp = 4
        rho = numpy.zeros((n_dm, comp, numpy.prod(ngrids)))
    else:
        raise NotImplementedError('meta-GGA')
    eval_fn = 'NUMINTrho_' + xctype.lower() + lattice_type
    drv = libdft.NUMINT_rho_drv
    if n_dm > 1:
        raise NotImplementedError("n_dm > 1")
    if hermi == 1:
        raise NotImplementedError("hermi == 1")
    drv(getattr(libdft, eval_fn),
        rho.ctypes.data_as(ctypes.c_void_p),
        (ctypes.c_int*3)(*offset),
        (ctypes.c_int*3)(*ngrids),
        dm.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(n_dm), ctypes.c_int(comp), ctypes.c_int(hermi),
        (ctypes.c_int*4)(i0, i1, j0, j1),
        ao_loc.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_double(numpy.log(cell.precision)),
        ctypes.c_int(cell.dimension),
        ctypes.c_int(nimgs),
        Ls.ctypes.data_as(ctypes.c_void_p),
        a.ctypes.data_as(ctypes.c_void_p),
        b.ctypes.data_as(ctypes.c_void_p),
        mesh.ctypes.data_as(ctypes.c_void_p),
        atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(atm)),
        bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(bas)),
        env.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(env)))

    if has_imag:
        n_dm /= 2
        rho = rho[:n_dm] + rho[n_dm:] * 1j

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
dm = dm + dm.transpose(0,2,1) # FIXME when kpts != 0 and dm is not hermitian

grids_orth = gen_grid.UniformGrids(cell_orth)
grids_north = gen_grid.UniformGrids(cell_north)

ao_kpts_orth = cell_orth.pbc_eval_gto('GTOval_sph_deriv1', grids_orth.coords, kpts=kpts)
ao_kpts_north = cell_north.pbc_eval_gto('GTOval_sph_deriv1', grids_north.coords, kpts=kpts)
ao_orth = mol_orth.eval_gto('GTOval_sph_deriv1', grids_orth.coords, kpts=kpts)
ao_north = mol_north.eval_gto('GTOval_sph_deriv1', grids_north.coords, kpts=kpts)
ao_gamma_orth = cell_orth.pbc_eval_gto('GTOval_sph_deriv1', grids_orth.coords)
ao_gamma_north = cell_orth.pbc_eval_gto('GTOval_sph_deriv1', grids_orth.coords)

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

        #out = eval_mat(pcell, vxc[0], hermi=1, kpts=kpts)
        #out = numpy.einsum('pi,kpq,qj->kij', contr_coeff, out, contr_coeff)
        #self.assertAlmostEqual(abs(out-ref).max(), 0, 9)

    def test_pbc_orth_gga_ints(self):
        ref = numpy.array([numpy.einsum('ngi,gj,ng->ij', ao.conj(), ao[0], vxc)
                           for ao in ao_kpts_orth])
        pcell, contr_coeff = uncontract(cell_orth)
        out = eval_mat(pcell, vxc, xctype='GGA', hermi=0, kpts=kpts)
        out = numpy.einsum('pi,kpq,qj->kij', contr_coeff, out, contr_coeff)
        self.assertAlmostEqual(abs(out-ref).max(), 0, 9)

        #out = eval_mat(pcell, vxc, xctype='GGA', hermi=1, kpts=kpts)
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

        #out = eval_mat(pcell, w, hermi=1, kpts=kpts)
        #out = numpy.einsum('pi,kpq,qj->kij', contr_coeff, out, contr_coeff)
        #self.assertAlmostEqual(abs(out-ref).max(), 0, 9)

    def test_pbc_nonorth_lda_ints(self):
        ref = numpy.array([numpy.einsum('gi,gj,g->ij', ao[0].conj(), ao[0], vxc[0])
                           for ao in ao_kpts_north])
        pcell, contr_coeff = uncontract(cell_north)
        out = eval_mat(pcell, vxc[0], hermi=0, kpts=kpts)
        out = numpy.einsum('pi,kpq,qj->kij', contr_coeff, out, contr_coeff)
        self.assertAlmostEqual(abs(out-ref).max(), 0, 9)

        #out = eval_mat(pcell, vxc[0], hermi=1, kpts=kpts)
        #out = numpy.einsum('pi,kpq,qj->kij', contr_coeff, out, contr_coeff)
        #self.assertAlmostEqual(abs(out-ref).max(), 0, 9)

    def test_pbc_nonorth_gga_ints(self):
        ref = numpy.array([numpy.einsum('ngi,gj,ng->ij', ao.conj(), ao[0], vxc)
                           for ao in ao_kpts_north])
        pcell, contr_coeff = uncontract(cell_north)
        out = eval_mat(pcell, vxc, xctype='GGA', hermi=0, kpts=kpts)
        out = numpy.einsum('pi,kpq,qj->kij', contr_coeff, out, contr_coeff)
        self.assertAlmostEqual(abs(out-ref).max(), 0, 9)

        #out = eval_mat(pcell, vxc, xctype='GGA', hermi=1, kpts=kpts)
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

        #out = eval_mat(pcell, w, hermi=1, kpts=kpts)
        #out = numpy.einsum('pi,kpq,qj->kij', contr_coeff, out, contr_coeff)
        #self.assertAlmostEqual(abs(out-ref).max(), 0, 9)

    def test_orth_lda_ints(self):
        ao = ao_orth
        ref = numpy.einsum('gi,gj,g->ij', ao[0], ao[0], vxc[0])
        pcell, contr_coeff = uncontract(mol_orth)
        out = eval_mat(pcell, vxc[0], hermi=0)
        out = numpy.einsum('pi,pq,qj->ij', contr_coeff, out, contr_coeff)
        self.assertAlmostEqual(abs(out-ref).max(), 0, 9)

        #out = eval_mat(pcell, vxc[0], hermi=1)
        #out = numpy.einsum('pi,pq,qj->ij', contr_coeff, out, contr_coeff)
        #self.assertAlmostEqual(abs(out-ref).max(), 0, 9)

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
        self.assertAlmostEqual(abs(out-ref).max(), 0, 8)

        #out = eval_rho(pcell, dm1, hermi=1, kpts=kpts)
        #self.assertAlmostEqual(abs(out-ref).max(), 0, 8)

    def test_pbc_orth_gga_rho(self):
        ao = ao_gamma_orth
        ref = numpy.einsum('ngi,ij,gj->ng', ao, dm[0], ao[0].conj())
        pcell, contr_coeff = uncontract(cell_orth)
        dm1 = numpy.einsum('pi,ij,qj->pq', contr_coeff, dm[0], contr_coeff)
        out = eval_rho(pcell, dm1, xctype='GGA')
        self.assertAlmostEqual(abs(out-ref).max(), 0, 8)

        #ref = sum([numpy.einsum('ngi,ij,gj->ng', ao, dm[k], ao[0].conj())
        #           for k,ao in enumerate(ao_kpts_orth)])
        #dm1 = numpy.einsum('pi,kij,qj->kpq', contr_coeff, dm, contr_coeff)
        #out = eval_rho(pcell, dm1, kpts=kpts, xctype='GGA')
        #self.assertAlmostEqual(abs(out-ref.real).max(), 0, 8)
        #self.assertAlmostEqual(abs(out-ref).max(), 0, 8)

        #out = eval_rho(pcell, dm1, hermi=1, kpts=kpts, xctype='GGA')
        #self.assertAlmostEqual(abs(out-ref).max(), 0, 9)

    def test_pbc_nonorth_lda_rho(self):
        ref = sum([numpy.einsum('gi,ij,gj->g', ao[0], dm[k], ao[0].conj())
                   for k,ao in enumerate(ao_kpts_north)])
        pcell, contr_coeff = uncontract(cell_north)
        dm1 = numpy.einsum('pi,kij,qj->kpq', contr_coeff, dm, contr_coeff)
        out = eval_rho(pcell, dm1, kpts=kpts)
        self.assertAlmostEqual(abs(out-ref).max(), 0, 9)

        #out = eval_rho(pcell, hermi=1, dm1, kpts=kpts)
        #self.assertAlmostEqual(abs(out-ref).max(), 0, 9)

    def test_pbc_nonorth_gga_rho(self):
        ao = ao_gamma_north
        ref = numpy.einsum('ngi,ij,gj->ng', ao, dm[0], ao[0].conj())
        pcell, contr_coeff = uncontract(cell_orth)
        dm1 = numpy.einsum('pi,ij,qj->pq', contr_coeff, dm[0], contr_coeff)
        out = eval_rho(pcell, dm1, xctype='GGA')
        self.assertAlmostEqual(abs(out-ref).max(), 0, 8)

        #ref = sum([numpy.einsum('ngi,ij,gj->ng', ao, dm[k], ao[0].conj())
        #           for k,ao in enumerate(ao_kpts_north)])
        #dm1 = numpy.einsum('pi,kij,qj->kpq', contr_coeff, dm, contr_coeff)
        #out = eval_rho(pcell, dm1, kpts=kpts, xctype='GGA')
        #self.assertAlmostEqual(abs(out-ref).max(), 0, 9)

        #out = eval_rho(pcell, dm1, hermi=1, kpts=kpts, xctype='GGA')
        #self.assertAlmostEqual(abs(out-ref).max(), 0, 9)

    def test_orth_lda_rho(self):
        ao = ao_orth
        ref = numpy.einsum('gi,ij,gj->g', ao[0], dm[0], ao[0].conj())
        pcell, contr_coeff = uncontract(mol_orth)
        dm1 = numpy.einsum('pi,ij,qj->pq', contr_coeff, dm[0], contr_coeff)
        out = eval_rho(pcell, dm1)
        self.assertAlmostEqual(abs(out-ref).max(), 0, 9)

        #out = eval_rho(pcell, dm1, hermi=1)
        #self.assertAlmostEqual(abs(out-ref).max(), 0, 9)

    def test_orth_gga_rho(self):
        ao = ao_orth
        ref = numpy.einsum('ngi,ij,gj->ng', ao, dm[0], ao[0].conj())
        pcell, contr_coeff = uncontract(mol_orth)
        dm1 = numpy.einsum('pi,ij,qj->pq', contr_coeff, dm[0], contr_coeff)
        out = eval_rho(pcell, dm1, xctype='GGA')
        self.assertAlmostEqual(abs(out-ref).max(), 0, 9)

        #out = eval_rho(pcell, dm1, hermi=1, xctype='GGA')
        #self.assertAlmostEqual(abs(out-ref).max(), 0, 9)

    def test_nonorth_lda_rho(self):
        ao = ao_north
        ref = numpy.einsum('gi,ij,gj->g', ao[0], dm[0], ao[0].conj())
        pcell, contr_coeff = uncontract(mol_north)
        dm1 = numpy.einsum('pi,ij,qj->pq', contr_coeff, dm[0], contr_coeff)
        out = eval_rho(pcell, dm1)
        self.assertAlmostEqual(abs(out-ref).max(), 0, 9)

        #out = eval_rho(pcell, dm1, hermi=1)
        #self.assertAlmostEqual(abs(out-ref).max(), 0, 9)

    def test_nonorth_gga_rho(self):
        ao = ao_north
        ref = numpy.einsum('ngi,ij,gj->ng', ao, dm[0], ao[0].conj())
        pcell, contr_coeff = uncontract(mol_north)
        dm1 = numpy.einsum('pi,ij,qj->pq', contr_coeff, dm[0], contr_coeff)
        out = eval_rho(pcell, dm1, xctype='GGA')
        self.assertAlmostEqual(abs(out-ref).max(), 0, 9)

        #out = eval_rho(pcell, dm1, hermi=1, xctype='GGA')
        #self.assertAlmostEqual(abs(out-ref).max(), 0, 9)

#TODO: test multiple dms and vxcs


if __name__ == '__main__':
    print("Full Tests for numint_uniform_grid")
    unittest.main()
