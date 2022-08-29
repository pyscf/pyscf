# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
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

'''
Extension to scipy.linalg module developed for PBC branch.
'''

import numpy as np
import scipy.linalg

def davidson_nosymm(matvec,size,nroots,Adiag=None):
    '''Davidson diagonalization method to solve A c = E c
    when A is not Hermitian.
    '''

    # We don't pass args
    def matvec_args(vec, args):
        return matvec(vec)

    nroots = min(nroots,size)
    #if Adiag == None:
    #   Adiag = matvec(numpy.ones(size))

    # Currently not used:
    x = np.ones((size,1))
    P = np.ones((size,1))

    arnold = Arnoldi(matvec_args, x, P, nroots=nroots)
    return arnold.solve()


VERBOSE = False

class Arnoldi:
    def __init__(self,matr_multiply,xStart,inPreCon,nroots=1,tol=1e-6):
        self.matrMultiply = matr_multiply
        self.size = xStart.shape[0]
        self.nEigen = min(nroots, self.size)
        self.maxM = min(30, self.size)
        self.maxOuterLoop = 10
        self.tol = tol

        #
        #  Creating initial guess and preconditioner
        #
        self.x0 = xStart.real.copy()

        self.iteration = 0
        self.totalIter = 0
        self.converged = False
        self.preCon = inPreCon.copy()
        #
        #  Allocating other vectors
        #
        self.allocateVecs()

    def solve(self):
        while self.converged == 0:
            if self.totalIter == 0:
                self.guessInitial()
            for i in range(self.maxM):
                if self.deflated == 1:
                    self.currentSize = self.nEigen

                if self.deflated == 0 and self.totalIter > 0:
                    self.hMult()
                    self.push_Av()
                    self.constructSubspace()

                self.solveSubspace()
                self.constructSol()
                self.computeResidual()
                self.checkConvergence()
                self.deflated = 0
                if self.converged:
                    break

                self.updateVecs()
                self.checkDeflate()
                self.constructDeflatedSub()

                self.totalIter += 1
                self.currentSize += 1
        print("")
        print("Converged in %3d cycles" % self.totalIter)
        self.constructAllSolV()
        return self.outeigs, self.outevecs

    def allocateVecs(self):
        self.subH = np.zeros( shape=(self.maxM,self.maxM), dtype=complex )
        self.sol = np.zeros( shape=(self.maxM), dtype=complex )
        self.dgks = np.zeros( shape=(self.maxM), dtype=complex )
        self.nConv = np.zeros( shape=(self.maxM), dtype=int )
        self.eigs = np.zeros( shape=(self.maxM), dtype=complex )
        self.evecs = np.zeros( shape=(self.maxM,self.maxM), dtype=complex )
        self.oldeigs = np.zeros( shape=(self.maxM), dtype=complex )
        self.deigs = np.zeros( shape=(self.maxM), dtype=complex )
        self.outeigs = np.zeros( shape=(self.nEigen), dtype=complex )
        self.outevecs = np.zeros( shape=(self.size,self.nEigen), dtype=complex)
        self.currentSize = 0

        self.Ax = np.zeros( shape=(self.size), dtype=complex )
        self.res = np.zeros( shape=(self.size), dtype=complex )
        self.vlist = np.zeros( shape=(self.maxM,self.size), dtype=complex )
        self.cv = np.zeros( shape = (self.size), dtype = complex )
        self.cAv = np.zeros( shape = (self.size), dtype = complex )
        self.Avlist = np.zeros( shape=(self.maxM,self.size), dtype=complex )
        self.dres = 999.9
        self.resnorm = 999.9
        self.cvEig = 0.1
        self.ciEig = 0
        self.deflated = 0

    def guessInitial(self):
        nrm = np.linalg.norm(self.x0)
        self.x0 *= 1./nrm
        self.currentSize = self.nEigen
        for i in range(self.currentSize):
            self.vlist[i] *= 0.0
            self.vlist[i,i] = 1.0 + 0.0*1j
            self.vlist[i] /= np.linalg.norm(self.vlist[i])
        for i in range(self.currentSize):
            self.cv = self.vlist[i].copy()
            self.hMult()
            self.Avlist[i] = self.cAv.copy()
        self.constructSubspace()

    def hMult(self):
        args = 0
        self.cAv = self.matrMultiply(self.cv.reshape(self.size),args)

    def push_Av(self):
        self.Avlist[self.currentSize-1] = self.cAv.reshape(self.size)

    def constructSubspace(self):
        if self.totalIter == 0 or self.deflated == 1: # construct the full block of v^*Av
            for i in range(self.currentSize):
                for j in range(self.currentSize):
                    val = np.vdot(self.vlist[i],self.Avlist[j])
                    self.subH[i,j] = val
        else:
            for j in range(self.currentSize):
                if j <= (self.currentSize-1):
                    val = np.vdot(self.vlist[j],self.Avlist[self.currentSize-1])
                    self.subH[j,self.currentSize-1] = val
                if j < (self.currentSize-1):
                    val = np.vdot(self.vlist[self.currentSize-1],self.Avlist[j])
                    self.subH[self.currentSize-1,j] = val

    def solveSubspace(self):
        w, v = scipy.linalg.eig(self.subH[:self.currentSize,:self.currentSize])
        idx = w.real.argsort()
        #imag_norm = np.linalg.norm(w.imag)
        #if imag_norm > 1e-12:
        #    print " *************************************************** "
        #    print " WARNING  IMAGINARY EIGENVALUE OF NORM %.15g " % (imag_norm)
        #    print " *************************************************** "
        #print "Imaginary norm eigenvectors = ", np.linalg.norm(v.imag)
        #print "Imaginary norm eigenvalue   = ", np.linalg.norm(w.imag)
        v = v[:,idx]
        w = w[idx].real
        self.sol[:self.currentSize] = v[:,self.ciEig]
        self.evecs[:self.currentSize,:self.currentSize] = v
        self.eigs[:self.currentSize] = w[:self.currentSize]
        self.outeigs[:self.nEigen] = w[:self.nEigen]
        self.cvEig = self.eigs[self.ciEig]

    def constructAllSolV(self):
        for i in range(self.nEigen):
            self.sol[:] = self.evecs[:,i]
            self.cv = np.dot(self.vlist[:self.currentSize].transpose(),self.sol[:self.currentSize])
            self.outevecs[:,i] = self.cv

    def constructSol(self):
        self.constructSolV()
        self.constructSolAv()

    def constructSolV(self):
        self.cv = np.dot(self.vlist[:self.currentSize].transpose(),self.sol[:self.currentSize])

    def constructSolAv(self):
        self.cAv = np.dot(self.Avlist[:self.currentSize].transpose(),self.sol[:self.currentSize])

    def computeResidual(self):
        self.res = self.cAv - self.cvEig * self.cv
        self.dres = np.vdot(self.res,self.res)**0.5
        #
        # gram-schmidt for residual vector
        #
        for i in range(self.currentSize):
            self.dgks[i] = np.vdot( self.vlist[i], self.res )
            self.res -= self.dgks[i]*self.vlist[i]
        #
        # second gram-schmidt to make them really orthogonal
        #
        for i in range(self.currentSize):
            self.dgks[i] = np.vdot( self.vlist[i], self.res )
            self.res -= self.dgks[i]*self.vlist[i]
        self.resnorm = np.linalg.norm(self.res)
        self.res /= self.resnorm

        orthog = 0.0
        for i in range(self.currentSize):
            orthog += np.vdot(self.res,self.vlist[i])**2.0
        orthog = orthog ** 0.5
        if not self.deflated:
            if VERBOSE:
                print("%3d %20.14f %20.14f  %10.4g" % (self.ciEig, self.cvEig.real, self.resnorm.real, orthog.real))
        #else:
        #    print "%3d %20.14f %20.14f %20.14f (deflated)" % (self.ciEig, self.cvEig,
        #                                                      self.resnorm, orthog)

        self.iteration += 1

    def updateVecs(self):
        self.vlist[self.currentSize] = self.res.copy()
        self.cv = self.vlist[self.currentSize]

    def checkConvergence(self):
        if self.resnorm < self.tol:
            if VERBOSE:
                print("Eigenvalue %3d converged! (res = %.15g)" % (self.ciEig, self.resnorm))
            self.ciEig += 1
        if self.ciEig == self.nEigen:
            self.converged = True
        if self.resnorm < self.tol and not self.converged:
            if VERBOSE:
                print("")
                print("")
                print("%-3s %-20s %-20s %-8s" % ("#", "  Eigenvalue", "  Res. Norm.", "  Ortho. (should be ~0)"))

    def gramSchmidtCurrentVec(self,northo):
        for i in range(northo):
            self.dgks[i] = np.vdot( self.vlist[i], self.cv )
            self.cv -= self.dgks[i]*self.vlist[i] #/ np.vdot(self.vlist[i],self.vlist[i])
        self.cv /= np.linalg.norm(self.cv)


    def checkDeflate(self):
        if self.currentSize == self.maxM-1:
            self.deflated = 1
            #print "deflating..."
            for i in range(self.nEigen):
                self.sol[:self.currentSize] = self.evecs[:self.currentSize,i]
                # Finds the "best" eigenvector for this eigenvalue
                self.constructSolV()
                # Puts this guess in self.Avlist rather than self.vlist for now...
                # since this would mess up self.constructSolV()'s solution
                self.Avlist[i] = self.cv.copy()
            for i in range(self.nEigen):
                # This is actually the "best" eigenvector v, not A*v (see above)
                self.cv = self.Avlist[i].copy()
                self.gramSchmidtCurrentVec(i)
                self.vlist[i] = self.cv.copy()

            for i in range(self.nEigen):
                # This is actually the "best" eigenvector v, not A*v (see above)
                self.cv = self.vlist[i].copy()
                # Use current vector cv to create cAv
                self.hMult()
                self.Avlist[i] = self.cAv.copy()

    def constructDeflatedSub(self):
        if self.deflated == 1:
            self.currentSize = self.nEigen
            self.constructSubspace()
