#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import numpy
import unittest
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf.scf import _vhf

mol = gto.Mole()
mol.build(
    verbose = 5,
    output = '/dev/null',
    atom = '''
O     0    0        0
H     0    -0.757   0.587
H     0    0.757    0.587''',
    basis = 'cc-pvdz',
)

mf = scf.RHF(mol)
mf.scf()
nao, nmo = mf.mo_coeff.shape


class KnowValues(unittest.TestCase):
    def test_incore_s4(self):
        eri4 = ao2mo.restore(4, mf._eri, nmo)
        dm = mf.make_rdm1()
        vj0, vk0 = _vhf.incore(eri4, dm, hermi=1)
        vj1, vk1 = scf.hf.get_jk(mol, dm, hermi=1)
        self.assertTrue(numpy.allclose(vj0,vj1))
        self.assertTrue(numpy.allclose(vk0,vk1))

    def test_direct_mapdm(self):
        numpy.random.seed(1)
        dm = numpy.random.random((nao,nao))
        eri0 = numpy.zeros((3,nmo,nmo,nmo,nmo))
        c_atm = numpy.array(mol._atm, dtype=numpy.int32)
        c_bas = numpy.array(mol._bas, dtype=numpy.int32)
        c_env = numpy.array(mol._env)
        i0 = 0
        for i in range(mol.nbas):
            j0 = 0
            for j in range(mol.nbas):
                k0 = 0
                for k in range(mol.nbas):
                    l0 = 0
                    for l in range(mol.nbas):
                        buf = gto.getints_by_shell('cint2e_ip1_sph', (i,j,k,l),
                                                   c_atm, c_bas, c_env, 3)
                        di,dj,dk,dl = buf.shape[1:]
                        eri0[:,i0:i0+di,j0:j0+dj,k0:k0+dk,l0:l0+dl] = buf
                        l0 += dl
                    k0 += dk
                j0 += dj
            i0 += di
        vj0 = numpy.einsum('nijkl,kl->nij', eri0, dm)
        vk0 = numpy.einsum('nijkl,kj->nil', eri0, dm)
        vj1, vk1 = _vhf.direct_mapdm('cint2e_ip1_sph', 's2kl',
                                     ('kl->s1ij', 'kj->s1il'),
                                     dm, 3, mol._atm, mol._bas, mol._env)
        self.assertTrue(numpy.allclose(vj0,vj1))
        self.assertTrue(numpy.allclose(vk0,vk1))

    def test_direct_bindm(self):
        numpy.random.seed(1)
        dm = numpy.random.random((nao,nao))
        vj0, vk0 = _vhf.direct_mapdm('cint2e_ip1_sph', 's2kl',
                                     ('kl->s1ij', 'kj->s1il'),
                                     dm, 3, mol._atm, mol._bas, mol._env)
        dms = (dm,dm)
        vj1, vk1 = _vhf.direct_bindm('cint2e_ip1_sph', 's2kl',
                                     ('kl->s1ij', 'kj->s1il'),
                                     dms, 3, mol._atm, mol._bas, mol._env)
        self.assertTrue(numpy.allclose(vj0,vj1))
        self.assertTrue(numpy.allclose(vk0,vk1))

    def test_rdirect_mapdm(self):
        numpy.random.seed(1)
        n2c = nao*2
        dm = numpy.random.random((n2c,n2c)) + \
             numpy.random.random((n2c,n2c)) * 1j
        eri0 = numpy.zeros((3,n2c,n2c,n2c,n2c),dtype=numpy.complex)
        c_atm = numpy.array(mol._atm, dtype=numpy.int32)
        c_bas = numpy.array(mol._bas, dtype=numpy.int32)
        c_env = numpy.array(mol._env)
        i0 = 0
        for i in range(mol.nbas):
            j0 = 0
            for j in range(mol.nbas):
                k0 = 0
                for k in range(mol.nbas):
                    l0 = 0
                    for l in range(mol.nbas):
                        buf = gto.getints_by_shell('cint2e_g1', (i,j,k,l),
                                                   c_atm, c_bas, c_env, 3)
                        di,dj,dk,dl = buf.shape[1:]
                        eri0[:,i0:i0+di,j0:j0+dj,k0:k0+dk,l0:l0+dl] = buf
                        l0 += dl
                    k0 += dk
                j0 += dj
            i0 += di
        vk0 = numpy.einsum('nijkl,jk->nil', eri0, dm)
        vj1, vk1 = _vhf.rdirect_mapdm('cint2e_g1', 'a4ij',
                                      ('lk->s2ij', 'jk->s1il'),
                                      dm, 3, mol._atm, mol._bas, mol._env)
        self.assertTrue(numpy.allclose(vk0,vk1))

    def test_rdirect_bindm(self):
        n2c = nao*2
        eri0 = numpy.zeros((n2c,n2c,n2c,n2c),dtype=numpy.complex)
        mfr = scf.DHF(mol)
        mfr.scf()
        dm = mfr.make_rdm1()[:n2c,:n2c].copy()
        c_atm = numpy.array(mol._atm, dtype=numpy.int32)
        c_bas = numpy.array(mol._bas, dtype=numpy.int32)
        c_env = numpy.array(mol._env)
        i0 = 0
        for i in range(mol.nbas):
            j0 = 0
            for j in range(mol.nbas):
                k0 = 0
                for k in range(mol.nbas):
                    l0 = 0
                    for l in range(mol.nbas):
                        buf = gto.getints_by_shell('cint2e_spsp1', (i,j,k,l),
                                                   c_atm, c_bas, c_env, 1)
                        di,dj,dk,dl = buf.shape
                        eri0[i0:i0+di,j0:j0+dj,k0:k0+dk,l0:l0+dl] = buf
                        l0 += dl
                    k0 += dk
                j0 += dj
            i0 += di

        vk0 = numpy.einsum('ijkl,jk->il', eri0, dm)
        vk1 = _vhf.rdirect_bindm('cint2e_spsp1', 's4', ('jk->s1il',),
                                 (dm,), 1, mol._atm, mol._bas, mol._env)
        self.assertTrue(numpy.allclose(vk0,vk1))


if __name__ == "__main__":
    print("Full Tests for _vhf")
    unittest.main()


