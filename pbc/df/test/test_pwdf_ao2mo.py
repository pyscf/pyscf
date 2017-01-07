import unittest
import numpy
from pyscf.pbc.df import pwdf
import pyscf.pbc.gto as pgto
from pyscf import ao2mo

L = 5.
n = 1
cell = pgto.Cell()
cell.a = numpy.diag([L,L,L])
cell.gs = numpy.array([n,n,n])

cell.atom = '''He    3.    2.       3.
               He    1.    1.       1.'''
cell.basis = 'ccpvdz'
cell.verbose = 0
cell.build(0,0)
nao = cell.nao_nr()


def finger(a):
    w = np.cos(np.arange(a.size))
    return np.dot(w, a.ravel())

class KnowValues(unittest.TestCase):
    def test_eri1111(self):
        kpts = numpy.random.random((4,3)) * .25
        kpts[3] = -numpy.einsum('ij->j', kpts[:3])
        with_df = pwdf.PWDF(cell)
        with_df.kpts = kpts
        mo =(numpy.random.random((nao,nao)) +
             numpy.random.random((nao,nao))*1j)
        eri = with_df.get_eri(kpts).reshape((nao,)*4)
        eri0 = numpy.einsum('pjkl,pi->ijkl', eri , mo.conj())
        eri0 = numpy.einsum('ipkl,pj->ijkl', eri0, mo       )
        eri0 = numpy.einsum('ijpl,pk->ijkl', eri0, mo.conj())
        eri0 = numpy.einsum('ijkp,pl->ijkl', eri0, mo       )
        eri1 = with_df.ao2mo(mo, kpts)
        self.assertAlmostEqual(abs(eri1.reshape(eri0.shape)-eri0).sum(), 0, 9)

    def test_eri0110(self):
        kpts = numpy.random.random((4,3)) * .25
        kpts[3] = kpts[0]
        kpts[2] = kpts[1]
        with_df = pwdf.PWDF(cell)
        with_df.kpts = kpts
        mo =(numpy.random.random((nao,nao)) +
             numpy.random.random((nao,nao))*1j)
        eri = with_df.get_eri(kpts).reshape((nao,)*4)
        eri0 = numpy.einsum('pjkl,pi->ijkl', eri , mo.conj())
        eri0 = numpy.einsum('ipkl,pj->ijkl', eri0, mo       )
        eri0 = numpy.einsum('ijpl,pk->ijkl', eri0, mo.conj())
        eri0 = numpy.einsum('ijkp,pl->ijkl', eri0, mo       )
        eri1 = with_df.ao2mo(mo, kpts)
        self.assertAlmostEqual(abs(eri1.reshape(eri0.shape)-eri0).sum(), 0, 9)

    def test_eri0000(self):
        with_df = pwdf.PWDF(cell)
        with_df.kpts = numpy.zeros((4,3))
        mo =(numpy.random.random((nao,nao)) +
             numpy.random.random((nao,nao))*1j)
        eri = ao2mo.restore(1, with_df.get_eri(with_df.kpts), nao)
        eri0 = numpy.einsum('pjkl,pi->ijkl', eri , mo.conj())
        eri0 = numpy.einsum('ipkl,pj->ijkl', eri0, mo       )
        eri0 = numpy.einsum('ijpl,pk->ijkl', eri0, mo.conj())
        eri0 = numpy.einsum('ijkp,pl->ijkl', eri0, mo       )
        eri1 = with_df.ao2mo(mo, with_df.kpts)
        self.assertAlmostEqual(abs(eri1.reshape(eri0.shape)-eri0).sum(), 0, 9)

        mo = mo.real
        eri0 = numpy.einsum('pjkl,pi->ijkl', eri , mo.conj())
        eri0 = numpy.einsum('ipkl,pj->ijkl', eri0, mo       )
        eri0 = numpy.einsum('ijpl,pk->ijkl', eri0, mo.conj())
        eri0 = numpy.einsum('ijkp,pl->ijkl', eri0, mo       )
        eri1 = with_df.ao2mo(mo, with_df.kpts, compact=False)
        self.assertAlmostEqual(abs(eri1.reshape(eri0.shape)-eri0).sum(), 0, 9)

if __name__ == '__main__':
    print("Full Tests for pwdf ao2mo")
    unittest.main()

