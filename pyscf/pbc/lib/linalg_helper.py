#!/usr/bin/env python
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
# Author: Timothy Berkelbach <tim.berkelbach@gmail.com>
#

from functools import reduce
import time
import numpy as np
import scipy.linalg

from pyscf import lib
from pyscf.lib import logger

'''
Extension to scipy.linalg module developed for PBC branch.
'''

method = 'davidson'
#method = 'arnoldi'

VERBOSE = False

try:
# Temporary fix for davidson when using mpicc
    from mpi4py import MPI
    FOUND_MPI4PY = True
    MPI_RANK = MPI.COMM_WORLD.Get_rank()
    MPI_COMM = MPI.COMM_WORLD
except (ImportError, OSError):
    FOUND_MPI4PY = False

def eigs(matvec, size, nroots, x0=None, Adiag=None, guess=False, verbose=logger.INFO):
    '''Davidson diagonalization method to solve A c = E c
    when A is not Hermitian.
    '''

    # We don't pass args
    def matvec_args(vec, args=None):
        return matvec(vec)

    nroots = min(nroots, size)

    if method == 'davidson':
        if guess == False:
            conv, e, c, niter = davidson(matvec, size, nroots, x0, Adiag, verbose)
        else:
            conv, e, c, niter = davidson_guess(matvec, size, nroots, Adiag)
        return conv, e, c
    elif method == 'arnoldi':
        # Currently not used:
        x = np.ones((size,1))
        P = np.ones((size,1))
        arnold = Arnoldi(matvec_args, x, P, nroots=nroots)
        e,c = arnold.solve()
        print("Arnoldi converged in %d iterations"%(arnold.totalIter))
        return e,c
    else:
        david = DavidsonZL()
        david.ndim = size
        david.neig = nroots
        david.diag = matvec(np.ones(size))
        david.matvec = matvec

        return david.solve_iter()


def davidson(mult_by_A, N, neig, x0=None, Adiag=None, verbose=logger.INFO):
    """Diagonalize a matrix via non-symmetric Davidson algorithm.

    mult_by_A() is a function which takes a vector of length N
        and returns a vector of length N.
    neig is the number of eigenvalues requested
    """

    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        import sys
        log = logger.Logger(sys.stdout, verbose)

    cput1 = (time.clock(), time.time())

    Mmin = min(neig,N)
    Mmax = min(N,2000)
    tol = 1e-6

    #Adiagcheck = np.zeros(N,np.complex)
    #for i in range(N):
    #    test = np.zeros(N,np.complex)
    #    test[i] = 1.0
    #    Adiagcheck[i] = mult_by_A(test)[i]
    #print "Analytical Adiag == numerical Adiag?", np.allclose(Adiag,Adiagcheck)

    if Adiag is None:
        Adiag = np.zeros(N,np.complex)
        for i in range(N):
            test = np.zeros(N,np.complex)
            test[i] = 1.0
            Adiag[i] = mult_by_A(test)[i]

    xi = np.zeros(N,np.complex)

    lamda_k_old = 0
    lamda_k = 0
    target = 0
    conv = False
    if x0 is not None:
        assert x0.shape == (N, Mmin)
        b = x0.copy()

        Ab = np.zeros((N,Mmin),np.complex)
        for m in range(Mmin):
            Ab[:,m] = mult_by_A(b[:,m])

    for istep,M in enumerate(range(Mmin,Mmax+1)):
        if M == Mmin:
            # Set of M unit vectors from lowest Adiag (NxM)
            b = np.zeros((N,M))
            idx = Adiag.argsort()
            for m,i in zip(range(M),idx):
                b[i,m] = 1.0
            ## Add random noise and orthogonalize
            #for m in range(M):
            #    b[:,m] += 0.01*np.random.random(N)
            #    b[:,m] /= np.linalg.norm(b[:,m])
            #    b,R = np.linalg.qr(b)

            Ab = np.zeros((N,M),np.complex)
            for m in range(M):
                Ab[:,m] = mult_by_A(b[:,m])
        else:
            Ab = np.column_stack( (Ab,mult_by_A(b[:,M-1])) )

        Atilde = np.dot(b.conj().transpose(),Ab)
        lamda, alpha = diagonalize_asymm(Atilde)
        lamda_k_old, lamda_k = lamda_k, lamda[target]
        alpha_k = alpha[:,target]

        if M == Mmax:
            break

        q = np.dot( Ab-lamda_k*b, alpha_k )
        log.info('davidson istep = %d  root = %d  E = %.15g  dE = %.9g  residual = %.6g',
                 istep, target, lamda_k.real, (lamda_k - lamda_k_old).real, np.linalg.norm(q))
        cput1 = log.timer('davidson iter', *cput1)
        if np.linalg.norm(q) < tol:
            if target == neig-1:
                conv = True
                break
            else:
                target += 1
        #for i in range(N):
            #eps = 0.
            #if np.allclose(lamda_k,Adiag[i]):
            #    eps = 1e-10
            #xi[i] = q[i]/(lamda_k-Adiag[i]+eps)
        eps = 1e-10
        xi = q/(lamda_k-Adiag+eps)

        # orthonormalize xi wrt b
        bxi,R = np.linalg.qr(np.column_stack((b,xi)))
        if FOUND_MPI4PY:  # Ensure all processes search in same direction
            bxi = MPI_COMM.bcast(bxi)

        # append orthonormalized xi to b
        b = np.column_stack((b,bxi[:,-1]))

    #if M > Mmin and M == Mmax:
    #    print("WARNING: Davidson algorithm reached max basis size "
    #          "M = %d without converging."%(M))

    # Express alpha in original basis
    evecs = np.dot(b,alpha) # b is N x M, alpha is M x M
    return conv, lamda[:neig], evecs[:,:neig], istep


def davidson_guess(mult_by_A,N,neig,Adiag=None):
    """Diagonalize a matrix via non-symmetric Davidson algorithm.

    mult_by_A() is a function which takes a vector of length N
        and returns a vector of length N.
    neig is the number of eigenvalues requested

    Targets the first nroots closest in amplitude to the first
    nroots of the basis. This is useful for the QP peaks in IP/EA.
    """
    Mmax = min(N,2000)
    tol = 1e-6

    #Adiagcheck = np.zeros(N,np.complex)
    #for i in range(N):
    #    test = np.zeros(N,np.complex)
    #    test[i] = 1.0
    #    Adiagcheck[i] = mult_by_A(test)[i]
    #print "Analytical Adiag == numerical Adiag?", np.allclose(Adiag,Adiagcheck)

    if Adiag is None:
        Adiag = np.zeros(N,np.complex)
        for i in range(N):
            test = np.zeros(N,np.complex)
            test[i] = 1.0
            Adiag[i] = mult_by_A(test)[i]

    xi = np.zeros(N,np.complex)

    evals = np.zeros(neig,np.complex)
    evecs = np.zeros((N,neig),np.complex)

    Mtot = 0
    for guess in range(neig):
        print("Working on guess =", guess+1, "/", neig)
        for M in range(1,Mmax+1):
            if M == 1:
                # Unit vector 'target' as the guess
                b = np.zeros((N,1))
                b[guess,0] = 1.0
                Ab = np.zeros((N,1),np.complex)
                Ab[:,0] = mult_by_A(b[:,0])
            else:
                Ab = np.column_stack( (Ab,mult_by_A(b[:,M-1])) )

            Atilde = np.dot(b.conj().transpose(),Ab)
            lamda, alpha = diagonalize_asymm(Atilde)

            overlap_guess_j_max = -99
            target = 0
            for j, overlap_guess_j in enumerate(alpha[0,:]):
                if overlap_guess_j > overlap_guess_j_max:
                    overlap_guess_j_max = overlap_guess_j
                    target = j

            lamda_k = lamda[target]
            alpha_k = alpha[:,target]

            if M == Mmax:
                print(" -- M reached Mmax")
                break

            q = np.dot( Ab-lamda_k*b, alpha_k )
            if np.linalg.norm(q) < tol:
                evals[guess] = lamda_k
                evecs[:,guess] = np.dot(b,alpha_k)
                Mtot += M
                print(" -- Converged in", M, "iterations")
                break

            for i in range(N):
                eps = 0.
                if np.allclose(lamda_k,Adiag[i]):
                    eps = 1e-8
                xi[i] = q[i]/(lamda_k-Adiag[i]+eps)

            # orthonormalize xi wrt b
            bxi,R = np.linalg.qr(np.column_stack((b,xi)))
            # append orthonormalized xi to b
            b = np.column_stack((b,bxi[:,-1]))

    if M > 1 and M == Mmax:
        print("WARNING: Davidson algorithm reached max basis size "
              "M = %d without converging."%(M))

    return evals, evecs, Mtot


def diagonalize_asymm(H):
    """
    Diagonalize a real, *asymmetric* matrix and return sorted results.

    Return the eigenvalues and eigenvectors (column matrix)
    sorted from lowest to highest eigenvalue.
    """
    E,C = np.linalg.eig(H)
    #if np.allclose(E.imag, 0*E.imag):
    #    E = np.real(E)
    #else:
    #    print "WARNING: Eigenvalues are complex, will be returned as such."

    idx = E.real.argsort()
    E = E[idx]
    C = C[:,idx]

    return E,C


class Arnoldi(object):
    def __init__(self,matr_multiply,xStart,inPreCon,nroots=1,tol=1e-10):
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
            for i in xrange(self.maxM):
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
        if VERBOSE:
            print("\nConverged in %3d cycles" % self.totalIter)
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
        for i in xrange(self.currentSize):
            self.vlist[i] *= 0.0
            self.vlist[i,:] += np.random.rand(self.size)
            self.vlist[i,i] /= np.linalg.norm(self.vlist[i,i])
            self.vlist[i,i] *= 12.
            for j in xrange(i):
                fac = np.vdot( self.vlist[j,:], self.vlist[i,:] )
                self.vlist[i,:] -= fac * self.vlist[j,:]
            self.vlist[i,:] /= np.linalg.norm(self.vlist[i,:])
        for i in xrange(self.currentSize):
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
            for i in xrange(self.currentSize):
                for j in xrange(self.currentSize):
                   val = np.vdot(self.vlist[i],self.Avlist[j])
                   self.subH[i,j] = val
        else:
            for j in xrange(self.currentSize):
                if j <= (self.currentSize-1):
                    val = np.vdot(self.vlist[j,:],self.Avlist[self.currentSize-1,:])
                    self.subH[j,self.currentSize-1] = val
                if j < (self.currentSize-1):
                    val = np.vdot(self.vlist[self.currentSize-1,:],self.Avlist[j,:])
                    self.subH[self.currentSize-1,j] = val

    def solveSubspace(self):
        w, v = scipy.linalg.eig(self.subH[:self.currentSize,:self.currentSize])
        idx = w.real.argsort()
        v = v[:,idx]
        w = w[idx].real
#
        imag_norm = np.linalg.norm(w.imag)
        if imag_norm > 1e-12:
            print(" *************************************************** ")
            print(" WARNING  IMAGINARY EIGENVALUE OF NORM %.15g " % (imag_norm))
            print(" *************************************************** ")
        #print "Imaginary norm eigenvectors = ", np.linalg.norm(v.imag)
        #print "Imaginary norm eigenvalue   = ", np.linalg.norm(w.imag)
        #print "eigenvalues = ", w[:min(self.currentSize,7)]
#
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
        import sys
        self.res = self.cAv - self.cvEig * self.cv
        self.dres = np.vdot(self.res,self.res)**0.5
        #
        # gram-schmidt for residual vector
        #
        for i in xrange(self.currentSize):
            self.dgks[i] = np.vdot( self.vlist[i,:], self.res )
            self.res -= self.dgks[i]*self.vlist[i,:]
        #
        # second gram-schmidt to make them really orthogonal
        #
        for i in xrange(self.currentSize):
            self.dgks[i] = np.vdot( self.vlist[i,:], self.res )
            self.res -= self.dgks[i]*self.vlist[i,:]
        self.resnorm = np.linalg.norm(self.res)
        self.res /= self.resnorm

        orthog = 0.0
        for i in xrange(self.currentSize):
            orthog += np.vdot(self.res,self.vlist[i])**2.0
            if orthog > 1e-8:
                sys.exit( "Exiting davidson procedure ... orthog = %24.16f" % orthog )
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
        for k in xrange(northo):
            fac = np.vdot( self.vlist[k,:], self.cv )
            self.cv -= fac * self.vlist[k,:] #/ np.vdot(self.vlist[i],self.vlist[i])
        cvnorm = np.linalg.norm(self.cv)
        if cvnorm < 1e-4:
            self.cv = np.random.rand(self.size)
            for k in xrange(northo):
                fac = np.vdot( self.vlist[k,:], self.cv )
                self.cv -= fac * self.vlist[k,:] #/ np.vdot(self.vlist[i],self.vlist[i])
            ########################################################################
            #  Sometimes some of the created vectors are linearly dependent.  i
            #  To get around this I'm not sure what to do other than to throw that
            #  solution out and start at that eigenvalue
            ########################################################################
            print(" ERROR!!!! ... restarting at eigenvalue #%" % \
                        (northo, cvnorm))
            self.ciEig = northo
        self.cv /= np.linalg.norm(self.cv)


    def checkDeflate(self):
        if self.currentSize == self.maxM-1:
            self.deflated = 1
            for i in xrange(self.nEigen):
                self.sol[:self.currentSize] = self.evecs[:self.currentSize,i]
                self.constructSolV()            # Finds the "best" eigenvector for this eigenvalue
                self.Avlist[i,:] = self.cv.copy() # Puts this guess in self.Avlist rather than self.vlist for now...
                                                # since this would mess up self.constructSolV()'s solution
            for i in xrange(self.nEigen):
                self.cv = self.Avlist[i,:].copy() # This is actually the "best" eigenvector v, not A*v (see above)
                self.gramSchmidtCurrentVec(i)
                self.vlist[i,:] = self.cv.copy()

            orthog = 0.0
            for j in xrange(self.nEigen):
                for i in xrange(j):
                    orthog += np.vdot(self.vlist[j,:],self.vlist[i,:])**2.0

            for i in xrange(self.nEigen):
                self.cv = self.vlist[i].copy() # This is actually the "best" eigenvector v, not A*v (see above)
                self.hMult()                   # Use current vector cv to create cAv
                self.Avlist[i] = self.cAv.copy()

    def constructDeflatedSub(self):
        if self.deflated == 1:
            self.currentSize = self.nEigen
            self.constructSubspace()


class DavidsonZL(object):
    def __init__(self):
        self.maxcycle = 200
        self.crit_e = 1.e-7
        self.crit_vec = 1.e-5
        self.crit_demo = 1.e-10
        self.crit_indp = 1.e-10
        # Basic setting
        self.iprt = 1
        self.ndim = 0
        self.neig = 5
        self.matvec = None
        self.v0 = None
        self.diag = None
        self.matrix = None

    def matvecs(self,vlst):
        n = len(vlst)
        wlst = [0]*n
        for i in range(n):
            wlst[i] = self.matvec(vlst[i])
        return wlst

    def genMatrix(self):
        v = np.identity(self.ndim)
        vlst = list(v)
        wlst = self.matvecs(vlst)
        Hmat = np.array(vlst).dot(np.array(wlst).T)
        self.matrix = Hmat
        return Hmat

    def solve_full(self):
        Hmat = self.matrix
        eig,vl,vr = eigGeneral(Hmat)
        return eig,vr

    def genV0(self):
        index = np.argsort(self.diag)[:self.neig]
        self.v0 = [0]*self.neig
        for i in range(self.neig):
            v = np.zeros(self.ndim)
            v[index[i]] = 1.0
            self.v0[i] = v.copy()
        return self.v0

    def solve_iter(self):
        if VERBOSE:
            print('Davidson solver for AX = wX')
            print(' ndim = ', self.ndim)
            print(' neig = ', self.neig)
            print(' maxcycle = ', self.maxcycle)
        #
        # Generate v0
        #
        vlst = self.genV0()
        wlst = self.matvecs(vlst)
        #
        # Begin to solve
        #
        ifconv = False
        neig = self.neig
        iconv = [False]*neig
        ediff = 0.0
        eigs = np.zeros(neig)
        ndim = neig
        rlst = []
        for niter in range(self.maxcycle):
            if self.iprt > 0:
                if VERBOSE: print('\n --- niter=',niter,'ndim0=',self.ndim,\
			           'ndim=',ndim,'---')

            # List[n,N] -> Max[N,n]
            vbas = np.array(vlst).transpose(1,0)
            wbas = np.array(wlst).transpose(1,0)
            iden = vbas.T.dot(vbas)
            diff = np.linalg.norm(iden-np.identity(ndim))
            if diff > 1.e-10:
                if VERBOSE: print('diff=',diff)
                if VERBOSE: print(iden)
                exit(1)
            tmpH = vbas.T.dot(wbas)
            eig,vl,vr = eigGeneral(tmpH)
            teig = eig[:neig]

            # Eigenvalue convergence
            nconv1 = 0
            for i in range(neig):
                tmp = abs(teig[i]-eigs[i])
                if VERBOSE: print(' i,eold,enew,ediff=',i,eigs[i],teig[i],tmp)
                if tmp <= self.crit_e: nconv1+=1
            if VERBOSE: print(' No. of converged eigval:',nconv1)
            if nconv1 == neig:
                if VERBOSE: print(' Cong: all eignvalues converged ! ')
            eigs = teig.copy()

            # Full Residuals: Res[i]=Res'[i]-w[i]*X[i]
            vr = vr[:,:neig].copy()
            jvec = vbas.dot(vr)
            rbas = wbas.dot(vr) - jvec.dot(np.diag(eigs))
            nconv2 = 0
            for i in range(neig):
                tmp = np.linalg.norm(rbas[:,i])
                if tmp <= self.crit_vec:
                    nconv2 += 1
                    iconv[i] = True
                else:
                    iconv[i] = False
                if VERBOSE: print(' i, norm=', i, tmp, iconv[i])
            if VERBOSE: print(' No. of converged eigvec:', nconv2)
            if nconv2 == neig:
                if VERBOSE: print(' Cong: all eigenvectors converged ! ')

            ifconv = (nconv1 == neig) or (nconv2 == neig)
            if ifconv:
                if VERBOSE: print(' Cong: ALL are converged !\n')
                break

            # Rotated basis to minimal subspace that
            # can give the exact [neig] eigenvalues
            nkeep = ndim #neig
            qbas = realRepresentation(vl,vr,nkeep)
            vbas = vbas.dot(qbas)
            wbas = wbas.dot(qbas)
            vlst = list(vbas.transpose(1,0))
            wlst = list(wbas.transpose(1,0))

            # New directions from residuals
            rlst = []
            for i in range(neig):
                if iconv[i] == True: continue
                for j in range(self.ndim):
                    tmp = self.diag[j] - eigs[i]
                    if abs(tmp) < self.crit_demo:
                        rbas[j,i] = rbas[j,i]/self.crit_demo
                    else:
                        rbas[j,i] = rbas[j,i]/tmp
                rlst.append(rbas[:,i].real)
                rlst.append(rbas[:,i].imag)

            # Re-orthogonalization and get Nindp
            nindp,vlst2 = mgs_ortho(vlst,rlst,self.crit_indp)

            if nindp != 0:
                wlst2 = self.matvecs(vlst2)
                vlst = vlst + vlst2
                wlst = wlst + wlst2
                ndim = len(vlst)
            else:
                if VERBOSE: print('Convergence Failure: Nindp=0 !')
                exit(1)

        if not ifconv:
            if VERBOSE: print('Convergence Failure: Out of Nmaxcycle !')

        return eigs, jvec


def svd_cut(mat,thresh):
    if len(mat.shape) != 2:
        print("NOT A MATRIX in SVD_CUT !", mat.shape)
        exit(1)
    d1, d2 = mat.shape
    u, sig, v = scipy.linalg.svd(mat, full_matrices=False)
    r = len(sig)
    for i in range(r):
        if sig[i] < thresh*1.01:
            r = i
            break
    # return empty list
    if r == 0: return [[],[],[]]
    bdim = r
    rkep = r
    u2 = np.zeros((d1,bdim))
    s2 = np.zeros((bdim))
    v2 = np.zeros((bdim,d2))
    u2[:,:rkep] = u[:,:rkep]
    s2[:rkep] = sig[:rkep]
    v2[:rkep,:] = v[:rkep,:]
    return u2, s2, v2

def eigGeneral(Hmat):
    n = Hmat.shape[0]
    eig,vl,vr = scipy.linalg.eig(Hmat,left=True)
    order = np.argsort(eig.real)
    for i in range(n-1):
        rdiff = eig[order[i]].real-eig[order[i+1]].real
        idiff = eig[order[i]].imag-eig[order[i+1]].imag
        # swap to a-w,a+w
        if abs(rdiff) < 1.e-14 and idiff>0.0:
            j = i + 1
            torder = order[i]
            order[i] = order[j]
            order[j] = torder
    eig = eig[order]
    vl = vl[:,order]
    vr = vr[:,order]
    # Normalize
    for i in range(n):
        ova = vl[:,i].T.conj().dot(vr[:,i])
        vl[:,i] = vl[:,i]/ova.conj()
        #test
        #ova = vl[:,i].T.conj().dot(vr[:,i])
    # A=RwL^+T
    tmpH = reduce(np.dot,(vr,np.diag(eig),vl.T.conj()))
    diff = np.linalg.norm(tmpH-Hmat)
    if diff > 1.e-8:
        if VERBOSE: print('error: A=R*w*L^+ !',diff)
        #exit(1)
    return eig, vl, vr

def realRepresentation(vl,vr,nred):
    vbas = np.hstack((vl[:,:nred].real,vl[:,:nred].imag,
                      vr[:,:nred].real,vl[:,:nred].imag))
    u,w,v=svd_cut(vbas,thresh=1.e-12)
    vbas =u.copy()
    return vbas

def mgs_ortho(vlst,rlst,crit_indp,iop=0):
    debug = False
    ndim = len(vlst)
    nres = len(rlst)
    maxtimes = 2
    # [N,n]
    vbas = np.array(vlst).transpose(1,0)
    rbas = np.array(rlst).transpose(1,0)
    # res=I-VVt*rlst
    for k in range(maxtimes):
        #rbas = rbas - reduce(np.dot,(vbas,vbas.T,rbas))
        tmp = np.dot(vbas.T,rbas)
        rbas -= np.dot(vbas,tmp)
    nindp = 0
    vlst2 = []
    if iop == 1:
        u,w,v = svd_cut(rbas,thresh=1.e-12)
        nindp = len(w)
        if nindp == 0: return nindp,vlst2
        vbas = np.hstack((vbas,u))
        vlst2 = list(u.transpose(1,0))
    else:
        # orthogonalization
        # - MORE STABLE since SVD sometimes does not converge !!!
        for k in range(maxtimes):
            for i in range(nres):
                rvec = rbas[:,i].copy()
                rii = np.linalg.norm(rbas[:,i])
                if debug: print(' ktime,i,rii=', k, i, rii)
                # TOO SMALL
                if rii <= crit_indp*10.0**(-k):
                    if debug: print(' unable to normalize:', i, ' norm=', rii,\
                                   ' thresh=', crit_indp)
                    continue
                # NORMALIZE
                rvec = rvec / rii
                rii = np.linalg.norm(rvec)
                rvec = rvec / rii
                nindp = nindp + 1
                vlst2.append(rvec)
                # Substract all things
                # [N,n]
                vbas = np.hstack((vbas,rvec.reshape(-1,1)))
                for j in range(i,nres):
                    #rbas[:,j]=rbas[:,j]-reduce(np.dot,(vbas,vbas.T,rbas[:,j]))
                    tmp = np.dot(vbas.T,rbas[:,j])
                    rbas[:,j] -= np.dot(vbas,tmp)
        # iden
    iden = vbas.T.dot(vbas)
    diff = np.linalg.norm(iden-np.identity(ndim+nindp))
    if diff > 1.e-10:
        if VERBOSE: print(' error in mgs_ortho: diff=', diff)
        if VERBOSE: print(iden)
        exit(1)
    else:
        if VERBOSE: print(' final nindp from mgs_ortho =', nindp, \
                          ' diffIden=', diff)
    return nindp, vlst2

if __name__ == '__main__':
    N = 200
    neig = 4
    A = np.zeros((N,N))
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
        return np.dot(A,x)

    e,c = eigs(matvec,N,neig,Adiag=np.diag(A))
    print("# davidson evals =", e)

    e,c = diagonalize_asymm(A)
    print("# numpy evals =", e.real[:neig])
