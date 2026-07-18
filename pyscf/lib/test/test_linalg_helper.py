# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import unittest
import numpy
import scipy.linalg
import tempfile
from pyscf import gto
from pyscf import scf
from pyscf import fci
from pyscf.lib import linalg_helper

class KnownValues(unittest.TestCase):
    def test_davidson(self):
        mol = gto.Mole()
        mol.verbose = 0
        mol.atom = [['H', (0,0,i)] for i in range(8)]
        mol.basis = {'H': 'sto-3g'}
        mol.build()
        mf = scf.RHF(mol)
        mf.scf()
        myfci = fci.FCI(mol, mf.mo_coeff)
        myfci.max_memory = .001
        myfci.max_cycle = 100
        e = myfci.kernel()[0]
        self.assertAlmostEqual(e, -11.579978414933732+mol.energy_nuc(), 9)

    def test_davidson_large_dx(self):
        mol = gto.M(atom='''
                    O 0 0 0
                    H 1.92 1.38 0
                    H -1.92 1.38 0''', verbose=0)
        ci = fci.FCI(mol.RHF().run()).run()
        self.assertAlmostEqual(ci.e_tot, -74.74294263255416, 8)

    def test_linalg_qr(self):
        a = numpy.random.random((9,5))+numpy.random.random((9,5))*1j
        q, r = linalg_helper._qr(a.T, numpy.dot)
        self.assertAlmostEqual(abs(r.T.dot(q)-a.T).max(), 0, 8)

    def test_davidson1(self):
        numpy.random.seed(12)
        n = 100
        a = numpy.random.rand(n,n)
        a = a + a.conj().T + numpy.diag(numpy.random.random(n))*10
        eref, u = scipy.linalg.eigh(a)

        def aop(x):
            return numpy.dot(a, x)
        x0 = a[0]
        e0, x0 = linalg_helper.dsyev(aop, x0, a.diagonal(), max_cycle=100,
                                     nroots=3, follow_state=True)
        self.assertAlmostEqual(abs(e0[:3] - eref[:3]).max(), 0, 8)
        self.assertAlmostEqual(abs(numpy.abs(x0[:3]) - abs(u[:,:3].T)).max(), 0, 5)

        x0 = a[0]
        e0, x0 = linalg_helper.dsyev(aop, x0, a.diagonal(), max_cycle=100,
                                     max_memory=1e-4, nroots=3, follow_state=True)
        self.assertAlmostEqual(abs(e0[:3] - eref[:3]).max(), 0, 8)
        self.assertAlmostEqual(abs(numpy.abs(x0[:3]) - abs(u[:,:3].T)).max(), 0, 5)

    def test_davidson_diag_matrix(self):
        numpy.random.seed(12)
        n = 100
        a = numpy.diag(numpy.random.random(n))
        eref = numpy.sort(a.diagonal())

        def aop(x):
            return numpy.dot(a, x)
        x0 = numpy.random.rand(n)
        e0, x0 = linalg_helper.dsyev(aop, x0, a.diagonal(), nroots=3)
        self.assertAlmostEqual(abs(e0 - eref[:3]).max(), 0, 8)

        a = numpy.eye(n) * 2
        def aop(x):
            return numpy.dot(a, x)
        x0 = numpy.random.rand(n)
        e0, x0 = linalg_helper.dsyev(aop, x0, a.diagonal(), nroots=3)
        self.assertEqual(e0.size, 1)
        self.assertAlmostEqual(e0, 2, 8)

    def test_solve(self):
        numpy.random.seed(12)
        n = 100
        a = numpy.random.rand(n,n)
        a += numpy.diag(numpy.random.random(n))* 10
        b = numpy.random.random(n)
        def aop(x):
            return numpy.dot(a,x)
        def precond(x, *args):
            return x / a.diagonal()
        xref = numpy.linalg.solve(a, b)
        x1 = linalg_helper.dsolve(aop, b, precond, max_cycle=80)
        self.assertAlmostEqual(abs(xref - x1).max(), 0, 3)

    def test_krylov_with_level_shift(self):
        numpy.random.seed(10)
        n = 100
        a = numpy.random.rand(n,n) * .1
        a = a.dot(a.T)
        a_diag = numpy.random.rand(n)
        b = numpy.random.rand(n)
        ref = numpy.linalg.solve(numpy.diag(a_diag) + a, b)

        #((diag+shift) + (a-shift)) x = b
        shift = .1
        a_diag += shift
        a -= numpy.eye(n)*shift

        aop = lambda x: (a.dot(x.T).T/a_diag)
        c = linalg_helper.krylov(aop, b/a_diag, max_cycle=18, lindep=1e-15)
        self.assertAlmostEqual(abs(ref - c).max(), 0, 9)
        c = linalg_helper.krylov(aop, b/a_diag, max_cycle=17)
        self.assertAlmostEqual(abs(ref - c).max(), 0, 8)

    def test_krylov_multiple_roots(self):
        numpy.random.seed(10)
        n = 100
        a = numpy.random.rand(n,n) * .1
        b = numpy.random.rand(4, n)
        ref = numpy.linalg.solve(numpy.eye(n) + a, b.T).T

        aop = lambda x: x.dot(a.T)
        c = linalg_helper.krylov(aop, b, lindep=1e-14)
        self.assertAlmostEqual(abs(ref - c).max(), 0, 7)

        a = numpy.random.rand(n,n) * .1 + numpy.random.rand(n,n) * .1j
        b = numpy.random.rand(4, n) + numpy.random.rand(4, n) * .5j
        ref = numpy.linalg.solve(numpy.eye(n) + a, b.T).T

        aop = lambda x: x.dot(a.T)
        c = linalg_helper.krylov(aop, b, lindep=1e-14)
        self.assertAlmostEqual(abs(ref - c).max(), 0, 7)

    def test_krylov_zero_force(self):
        numpy.random.seed(10)
        n = 2
        a = numpy.random.rand(n,n) * .1
        a = a.dot(a.T)
        a_diag = numpy.random.rand(n)
        b = numpy.zeros((3, n))
        a -= numpy.eye(n)

        aop = lambda x: (a.dot(x.T).T/a_diag)
        c = linalg_helper.krylov(aop, b/a_diag, max_cycle=18, lindep=1e-15)
        self.assertAlmostEqual(abs(c).max(), 0, 9)

    def test_dgeev(self):
        numpy.random.seed(12)
        n = 100
        a = numpy.random.rand(n,n)
        a = a + a.conj().T
        a += numpy.diag(numpy.random.random(n))* 10
        b = numpy.random.random((n,n))
        b = numpy.dot(b,b.T)

        def abop(x):
            return numpy.dot(numpy.asarray(x), a.T), numpy.dot(numpy.asarray(x), b.T)

        eref, u = scipy.linalg.eigh(a, b)
        x0 = a[0]
        e0,x0 = linalg_helper.dgeev1(abop, x0, a.diagonal(), type=1,
                                     max_cycle=100, nroots=3)[1:]
        self.assertAlmostEqual(abs(e0 - eref[:3]).max(), 0, 8)
        self.assertAlmostEqual(abs(numpy.abs(x0) - abs(u[:,:3].T)).max(), 0, 4)

        eref, u = scipy.linalg.eigh(a, b, type=2)
        x0 = a[0]
        e0,x0 = linalg_helper.dgeev1(abop, x0, a.diagonal(), type=2,
                                     max_cycle=100, nroots=3)[1:]
        self.assertAlmostEqual(abs(e0 - eref[:3]).max(), 0, 8)
        self.assertAlmostEqual(abs(numpy.abs(x0) - abs(u[:,:3].T)).max(), 0, 4)

    def test_eig1(self):
        numpy.random.seed(12)
        n = 100
        a = numpy.random.rand(n,n)
        a = a + a.conj().T
        a += numpy.diag(numpy.random.random(n))* 10
        b = numpy.random.random((n,n))
        b = numpy.dot(b,b.T) + numpy.eye(n)*5

        def abop(x):
            return numpy.dot(numpy.asarray(x), a.T), numpy.dot(numpy.asarray(x), b.T)

        eref, u = scipy.linalg.eigh(a, b, type=2)
        u /= numpy.linalg.norm(u, axis=0)
        x0 = a[0]
        def abop(x):
            x = numpy.asarray(x).T
            return numpy.dot(a, numpy.dot(b, x)).T.copy()
        e0, x0 = linalg_helper.eig(abop, x0, a.diagonal(), max_cycle=100,
                                   nroots=3, pick=linalg_helper.pick_real_eigs)
        self.assertAlmostEqual(abs(e0 - eref[:3]).max(), 0, 7)

    def test_eig2(self):
        numpy.random.seed(12)
        n = 100
        a = numpy.random.rand(n,n)
        a = a + a.conj().T
        a += numpy.diag(numpy.random.random(n))* 10
        b = numpy.random.random((n,n))
        b = numpy.dot(b,b.T) + numpy.eye(n)*5

        def abop(x):
            x = numpy.asarray(x).T
            return numpy.dot(a, numpy.dot(b, x)).T.copy()

        e, ul, u = scipy.linalg.eig(numpy.dot(a, b), left=True)
        idx = numpy.argsort(e)
        e = e[idx]
        ul = ul[:,idx]
        u  = u [:,idx]
        u  /= numpy.linalg.norm(u, axis=0)
        ul /= numpy.linalg.norm(ul, axis=0)
        x0 = a[0]
        e0, vl, vr = linalg_helper.eig(abop, x0, a.diagonal(), max_cycle=100,
                         nroots=3, pick=linalg_helper.pick_real_eigs, left=True)
        self.assertAlmostEqual(abs(e0 - e[:3]).max(), 0, 7)
        self.assertAlmostEqual(abs(numpy.abs(vr) - abs(u[:,:3].T)).max(), 0, 5)
        # FIXME: left eigenvectors do not agree with scipy results
        print((abs(vl[0]) - abs(ul[:,0])).max())
        print((abs(vl[1]) - abs(ul[:,1])).max())
        print((abs(vl[2]) - abs(ul[:,2])).max())

    def test_eig_difficult_problem(self):
        N = 40
        neig = 4
        A = numpy.zeros((N,N))
        k = N/2
        for ii in range(N):
            i = ii+1
            for jj in range(N):
                j = jj+1
                if j <= k:
                    A[ii,jj] = i*(i==j)-(i-j-k**2)
                else:
                    A[ii,jj] = i*(i==j)+(i-j-k**2)
        def matvec(x):
            return numpy.dot(A,x)

        def precond(r, e0, x0):
            return (r+e0*x0) / A.diagonal()  # Converged
            #return (r+e0*x0) / (A.diagonal()-e0)  # Does not converge
            #return r / (A.diagonal()-e0)  # Does not converge
        x0 = A[0]
        e, c = linalg_helper.eig(matvec, x0, precond, nroots=2, tol=1e-6, verbose=5)

if __name__ == "__main__":
    print("Full Tests for linalg_helper")
    unittest.main()
