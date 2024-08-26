import unittest
import numpy
import scipy.linalg

from pyscf import lib
from pyscf.pbc import gto
from pyscf.pbc.dft import gen_grid
from pyscf.pbc.dft import multigrid

from pyscf.pbc.dft.multigrid.multigrid import eval_mat, eval_rho

def uncontract(cell):
    pcell, contr_coeff = cell.to_uncontracted_cartesian_basis()
    return pcell, scipy.linalg.block_diag(*contr_coeff)

def setUpModule():
    global cell_orth, cell_north, mol_orth, mol_north
    global bak_EXPDROP, bak_EXTRA_PREC
    global vxc, kpts, nkpts, nao, dm, dm_kpts, grids_orth, grids_north
    global ao_kpts_orth, ao_kpts_north, ao_orth, ao_north, ao_gamma_orth, ao_gamma_north
    multigrid.multigrid.EXPDROP, bak_EXPDROP = 1e-14, multigrid.multigrid.EXPDROP
    multigrid.multigrid.EXTRA_PREC, bak_EXTRA_PREC = 1e-3, multigrid.multigrid.EXTRA_PREC

    numpy.random.seed(2)
    cell_orth = gto.M(atom='H1 1 1 0; H2 0 0 1',
                      basis={'H1':[[0, ( 1, 1, .1), (.5, .1, 1)],
                                   [1, (.8, 1, .2), (.3, .2, 1)]],
                             'H2':[[0, (.9, .6, .3), (.4, .1, 1)],
                                   [2, (.7, .8, .2), (.2, .2, 1)]]},
                      unit='B',
                      mesh=[7,6,5],
                      a=numpy.eye(3)*8,
                      precision=1e-9)

    mol_orth = cell_orth.copy()
    mol_orth.dimension = 0

    cell_north = gto.M(atom='H1 1 1 0; H2 0 0 1',
                       basis={'H1':[[0, ( 1, 1, .1), (.5, .1, 1)],
                                    [1, (.8, 1, .2), (.3, .2, 1)]],
                              'H2':[[0, (.9, .6, .3), (.4, .1, 1)],
                                    [2, (.7, .8, .2), (.2, .2, 1)]]},
                       unit='B',
                       mesh=[7,6,5],
                       a=numpy.eye(3)*8+numpy.random.rand(3,3),
                       precision=1e-9)

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

    grids_orth = gen_grid.UniformGrids(cell_orth).run()
    grids_north = gen_grid.UniformGrids(cell_north).run()
    grids_orth.coords = cell_orth.get_uniform_grids(wrap_around=False)
    grids_north.coords = cell_north.get_uniform_grids(wrap_around=False)

    ao_kpts_orth = cell_orth.pbc_eval_gto('GTOval_sph_deriv1', grids_orth.coords, kpts=kpts)
    ao_kpts_north = cell_north.pbc_eval_gto('GTOval_sph_deriv1', grids_north.coords, kpts=kpts)
    ao_orth = mol_orth.eval_gto('GTOval_sph_deriv1', grids_orth.coords, kpts=kpts)
    ao_north = mol_north.eval_gto('GTOval_sph_deriv1', grids_north.coords, kpts=kpts)
    ao_gamma_orth = cell_orth.pbc_eval_gto('GTOval_sph_deriv1', grids_orth.coords)
    ao_gamma_north = cell_north.pbc_eval_gto('GTOval_sph_deriv1', grids_north.coords)

def tearDownModule():
    global cell_orth, cell_north, mol_orth, mol_north
    global bak_EXPDROP, bak_EXTRA_PREC
    del cell_orth, cell_north, mol_orth, mol_north
    # They affect multigrid tests when using nosetests
    multigrid.EXPDROP = bak_EXPDROP
    multigrid.EXTRA_PREC = bak_EXTRA_PREC

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
        pcell.mesh = [30]*3
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
        self.assertAlmostEqual(abs(out-ref).max(), 0, 7)

        #out = eval_mat(pcell, vxc, xctype='GGA', hermi=1, kpts=kpts)
        self.assertRaises(RuntimeError, eval_mat, pcell, vxc, xctype='GGA', hermi=1, kpts=kpts)
        #out = numpy.einsum('pi,kpq,qj->kij', contr_coeff, out, contr_coeff)
        #self.assertAlmostEqual(abs(out-ref).max(), 0, 9)

    def test_pbc_nonorth_overlap(self):
        ref = cell_north.pbc_intor('int1e_ovlp', kpts=kpts)
        pcell, contr_coeff = uncontract(cell_north)
        pcell.mesh = [30]*3
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

    def test_pbc_orth_lda_rho_submesh(self):
        cell = gto.M(atom='H 2 3 4; H 3 4 3',
                  basis=[[0, (2.2, 1)],
                         [1, (1.9, 1)]],
                  unit='B',
                  mesh=[7,6,5],
                  a=numpy.eye(3)*8)
        grids = cell.get_uniform_grids(wrap_around=False)
        ao = cell.pbc_eval_gto('GTOval', grids)
        nao = cell.nao_nr()
        dm = numpy.random.random((nao,nao))
        dm = dm + dm.T
        ref = numpy.einsum('gi,ij,gj->g', ao, dm, ao.conj())
        ref = ref.reshape(cell.mesh)[1:6,1:5,1:4].ravel()
        out = eval_rho(cell, dm, offset=[1,1,1], submesh=[5,4,3])
        self.assertAlmostEqual(abs(out-ref).max(), 0, 9)

    def test_pbc_nonorth_lda_rho_submesh(self):
        cell = gto.M(atom='H 2 3 4; H 3 4 3',
                  basis=[[0, (2.2, 1)],
                         [1, (1.9, 1)]],
                  unit='B',
                  mesh=[7,6,5],
                  a=numpy.eye(3)*8+numpy.random.rand(3,3))
        grids = cell.get_uniform_grids(wrap_around=False)
        ao = cell.pbc_eval_gto('GTOval', grids)
        nao = cell.nao_nr()
        dm = numpy.random.random((nao,nao))
        dm = dm + dm.T
        ref = numpy.einsum('gi,ij,gj->g', ao, dm, ao.conj())
        ref = ref.reshape(cell.mesh)[1:6,2:5,2:4].ravel()
        out = eval_rho(cell, dm, offset=[1,2,2], submesh=[5,3,2])
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
