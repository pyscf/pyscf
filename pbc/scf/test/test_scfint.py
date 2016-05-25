import unittest
import numpy
import numpy as np
from pyscf.pbc.scf import scfint
from pyscf.pbc import gto as pbcgto
import pyscf.pbc.dft as pdft
import pyscf.pbc.scf.hf as phf


def make_cell1(L, n, nimgs=None):
    cell = pbcgto.Cell()
    cell.verbose = 0
    cell.unit = 'B'
    cell.h = ((L,0,0),(0,L,0),(0,0,L))
    cell.gs = [n,n,n]

    cell.atom = [['He', (L/2.,L/2.,L/2.)], ]
    cell.basis = { 'He': [[0, (0.8, 1.0)],
                         [0, (1.0, 1.0)],
                         [0, (1.2, 1.0)]] }
    cell.pseudo = None
    cell.nimgs = nimgs
    cell.build(False, False)
    return cell

def make_cell2(L, n, nimgs=None):
    cell = pbcgto.Cell()
    cell.build(False, False,
               unit = 'B',
               verbose = 0,
               h = ((L,0,0),(0,L,0),(0,0,L)),
               gs = [n,n,n],
               atom = [['He', (L/2.-.5,L/2.,L/2.-.5)],
                       ['He', (L/2.   ,L/2.,L/2.+.5)]],
               basis = { 'He': [[0, (0.8, 1.0)],
                                [0, (1.0, 1.0)],
                                [0, (1.2, 1.0)]] })
    return cell

k = numpy.ones(3) * .25

def finger(mat):
    w = numpy.cos(numpy.arange(mat.size))
    return numpy.dot(mat.ravel(), w)

def get_ovlp(cell, kpt=np.zeros(3)):
    '''Get the overlap AO matrix.
    '''
    coords = pdft.gen_grid.gen_uniform_grids(cell)
    aoR = pdft.numint.eval_ao(cell, coords, kpt)
    ngs = len(aoR)
    s = (cell.vol/ngs) * np.dot(aoR.T.conj(), aoR)
    return s

def get_t(cell, kpt=np.zeros(3)):
    '''Get the kinetic energy AO matrix.

    Note: Evaluated in real space using orbital gradients, for improved accuracy.
    '''
    coords = pdft.gen_grid.gen_uniform_grids(cell)
    aoR = pdft.numint.eval_ao(cell, coords, kpt, deriv=1)
    ngs = aoR.shape[1]  # because we requested deriv=1, aoR.shape[0] = 4

    t = 0.5*(np.dot(aoR[1].T.conj(), aoR[1]) +
             np.dot(aoR[2].T.conj(), aoR[2]) +
             np.dot(aoR[3].T.conj(), aoR[3]))
    t *= (cell.vol/ngs)
    return t

def get_pp(cell, kpt=np.zeros(3)):
    '''Get the periodic pseudotential nuc-el AO matrix, with G=0 removed.
    '''
    from pyscf.pbc import tools
    from pyscf.pbc.gto import pseudo
    coords = pdft.gen_grid.gen_uniform_grids(cell)
    aoR = pdft.numint.eval_ao(cell, coords, kpt)
    nao = cell.nao_nr()

    SI = cell.get_SI()
    vlocG = pseudo.get_vlocG(cell)
    vpplocG = -np.sum(SI * vlocG, axis=0)

    # vpploc evaluated in real-space
    vpplocR = tools.ifft(vpplocG, cell.gs)
    vpploc = np.dot(aoR.T.conj(), vpplocR.reshape(-1,1)*aoR)

    # vppnonloc evaluated in reciprocal space
    aokG = np.empty(aoR.shape, np.complex128)
    for i in range(nao):
        aokG[:,i] = tools.fftk(aoR[:,i], cell.gs, coords, kpt)
    ngs = len(aokG)

    vppnl = np.zeros((nao,nao), dtype=np.complex128)
    hs, projGs = pseudo.get_projG(cell, kpt)
    for ia, [h_ia,projG_ia] in enumerate(zip(hs,projGs)):
        for l, h in enumerate(h_ia):
            nl = h.shape[0]
            for m in range(-l,l+1):
                SPG_lm_aoG = np.zeros((nl,nao), dtype=np.complex128)
                for i in range(nl):
                    SPG_lmi = SI[ia,:] * projG_ia[l][m][i]
                    SPG_lm_aoG[i,:] = np.einsum('g,gp->p', SPG_lmi.conj(), aokG)
                for i in range(nl):
                    for j in range(nl):
                        # Note: There is no (-1)^l here.
                        vppnl += h[i,j]*np.einsum('p,q->pq',
                                                   SPG_lm_aoG[i,:].conj(),
                                                   SPG_lm_aoG[j,:])
    vppnl *= (1./ngs**2)

    return vpploc + vppnl


class KnowValues(unittest.TestCase):
    def test_olvp(self):
        cell = make_cell1(4, 20, [2,2,2])
        s0 = get_ovlp(cell)
        s1 = scfint.get_ovlp(cell)
        self.assertAlmostEqual(numpy.linalg.norm(s0-s1), 0, 8)
        self.assertAlmostEqual(finger(s1), 1.3229918679678208, 10)

        s0 = get_ovlp(cell, kpt=k)
        s1 = scfint.get_ovlp(cell, kpt=k)
        self.assertAlmostEqual(numpy.linalg.norm(s0-s1), 0, 8)

    def test_t(self):
        numpy.random.seed(3)
        cell = make_cell1(4, 20, [2,2,2])
        t0 = get_t(cell, kpt=k)
        t1 = scfint.get_t(cell, kpt=k)
        self.assertAlmostEqual(numpy.linalg.norm(t0-t1), 0, 8)

    def test_pp(self):
        cell = pbcgto.Cell()
        cell.verbose = 0
        cell.atom = 'C 0 0 0; C 1 1 1; C 0 2 2; C 2 0 2'
        cell.h = np.diag([4, 4, 4])
        cell.basis = 'gth-szv'
        cell.pseudo = 'gth-pade'
        cell.gs = [10, 10, 10]
        cell.build()
        v0 = get_pp(cell, k)
        v1 = phf.get_pp(cell, k)
        self.assertAlmostEqual(numpy.linalg.norm(v0-v1), 0, 8)


if __name__ == '__main__':
    print("Full Tests for pbc.scf.hf.scfint")
    unittest.main()

