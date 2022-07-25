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
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

"""
DIIS
"""

from functools import reduce
import numpy
import numpy.linalg
import numpy.testing
import scipy.linalg
import scipy.optimize
import pyscf.scf._vhf
from pyscf import lib
from pyscf.lib import logger
from pyscf.ao2mo import addons
import time

DEBUG = False

# J. Mol. Struct. 114, 31-34 (1984); DOI:10.1016/S0022-2860(84)87198-7
# PCCP, 4, 11 (2002); DOI:10.1039/B108658H
# GEDIIS, JCTC, 2, 835 (2006); DOI:10.1021/ct050275a
# C2DIIS, IJQC, 45, 31 (1993); DOI:10.1002/qua.560450106
# SCF-EDIIS, JCP 116, 8255 (2002); DOI:10.1063/1.1470195

# error vector = SDF-FDS
# error vector = F_ai ~ (S-SDS)*S^{-1}FDS = FDS - SDFDS ~ FDS-SDF in converge
class CDIIS(lib.diis.DIIS):
    def __init__(self, mf=None, filename=None):
        lib.diis.DIIS.__init__(self, mf, filename)
        self.rollback = False
        self.space = 8

    def update(self, s, d, f, *args, **kwargs):
        errvec = get_err_vec(s, d, f)
        logger.debug1(self, 'diis-norm(errvec)=%g', numpy.linalg.norm(errvec))
        xnew = lib.diis.DIIS.update(self, f, xerr=errvec)
        if self.rollback > 0 and len(self._bookkeep) == self.space:
            self._bookkeep = self._bookkeep[-self.rollback:]
        return xnew

    def get_num_vec(self):
        if self.rollback:
            return self._head
        else:
            return len(self._bookkeep)

SCFDIIS = SCF_DIIS = DIIS = CDIIS

def get_err_vec(s, d, f):
    '''error vector = SDF - FDS'''
    if isinstance(f, numpy.ndarray) and f.ndim == 2:
        sdf = reduce(numpy.dot, (s,d,f))
        errvec = sdf.T.conj() - sdf

    elif isinstance(f, numpy.ndarray) and f.ndim == 3 and s.ndim == 3:
        errvec = []
        for i in range(f.shape[0]):
            sdf = reduce(numpy.dot, (s[i], d[i], f[i]))
            errvec.append((sdf.T.conj() - sdf))
        errvec = numpy.vstack(errvec)

    elif f.ndim == s.ndim+1 and f.shape[0] == 2:  # for UHF
        nao = s.shape[-1]
        s = lib.asarray((s,s)).reshape(-1,nao,nao)
        return get_err_vec(s, d.reshape(s.shape), f.reshape(s.shape))
    else:
        raise RuntimeError('Unknown SCF DIIS type')
    return errvec


class EDIIS(lib.diis.DIIS):
    '''SCF-EDIIS
    Ref: JCP 116, 8255 (2002); DOI:10.1063/1.1470195
    '''
    def update(self, s, d, f, mf, h1e, vhf):
        if self._head >= self.space:
            self._head = 0
        if not self._buffer:
            shape = (self.space,) + f.shape
            self._buffer['dm'  ] = numpy.zeros(shape, dtype=f.dtype)
            self._buffer['fock'] = numpy.zeros(shape, dtype=f.dtype)
            self._buffer['etot'] = numpy.zeros(self.space)
        self._buffer['dm'  ][self._head] = d
        self._buffer['fock'][self._head] = f
        self._buffer['etot'][self._head] = mf.energy_elec(d, h1e, vhf)[0]
        self._head += 1

        ds = self._buffer['dm'  ]
        fs = self._buffer['fock']
        es = self._buffer['etot']
        etot, c = ediis_minimize(es, ds, fs)
        logger.debug1(self, 'E %s  diis-c %s', etot, c)
        fock = numpy.einsum('i,i...pq->...pq', c, fs)
        return fock


class FDIIS(lib.diis.DIIS):
    '''SCF-EDIIS
    Ref: JCP 116, 8255 (2002); DOI:10.1063/1.1470195
    '''
    # 2 electron integrals
    _int2e_as_mat = None
    
    # FDIIS latency
    _use_dynamic_latency = True
    _dynamic_latency_thresh = 3
    _static_latency_thresh = 2
    _fdiis_octahedral_breadth = 0.5
    _fdiis_mo_coeff_norm = 0

    def update(self, s, d, f, mf, h1e, vhf):
        chkTime1 = time.time()
        self._static_latency_thresh = self.space

        if self._head >= self.space:
            self._head = 0
        if not self._buffer:
            shape = (self.space,) + f.shape
            self._buffer['dm'  ] = numpy.zeros(shape, dtype=f.dtype)
            self._buffer['fock'] = numpy.zeros(shape, dtype=f.dtype)
            self._buffer['etot'] = numpy.zeros(self.space)
            # DMs for calculated minima in each iterative step
            # self._buffer['mins'] = numpy.zeros(shape, dtype=f.dtype)
            # DMs for distances to calculated minima in each iterative step
            # self._buffer['mindists'] = numpy.zeros(self.space)
        # initialising antisymmetrised 2e integral matrix
        #if self._int2e_as_mat is None:
        #    self._int2e_as_mat = fdiis_getInt2eAsMat(mf)
        #    print(self._int2e_as_mat)
        self._buffer['dm'  ][self._head] = d
        self._buffer['fock'][self._head] = f
        self._buffer['etot'][self._head] = mf.energy_elec(d, h1e, vhf)[0]
        self._head += 1

        lowerIndex = self._head - 2
        if lowerIndex == -1:
            lowerIndex = self.space-1

        
        energy_improvement = abs(self._buffer['etot'][self._head-1] - self._buffer['etot'][lowerIndex]) 

        ds = self._buffer['dm'  ]
        fs = self._buffer['fock']
        es = self._buffer['etot']
        
        chkTime2 = time.time()
        if energy_improvement <= mf.conv_tol * 10**self._dynamic_latency_thresh:
            print("FDIIS active")
            #mo_energy, mo_coeffs = fdiis_getPertubedMOCoeffs(mf, f, s, self._fdiis_octahedral_breadth, self._fdiis_mo_coeff_norm)
            ds = fdiis_genPertubationMatrices(mf, d, self._fdiis_octahedral_breadth)
            #fdiis_verifyDms(mf, ds)
            fs, es = fdiis_getPertubedFocksAndEnergies(mf, ds, f, self._buffer['etot'][self._head-1])
        #else:
        #    mo_energy, mo_coeffs = mf.eig(f, s)
        #    self._fdiis_mo_coeff_norm = (numpy.linalg.norm(mo_coeffs) * numpy.linalg.norm(mo_coeffs.T))**(0.5)

        chkTime3 = time.time()

        etot, c = ediis_minimize(es, ds, fs)
        chkTime4 = time.time()
        logger.debug1(self, 'F %s  diis-c %s', etot, c)
        fock = numpy.einsum('i,i...pq->...pq', c, fs)
        chkTime5 = time.time()

        #print(len(fdiis_convert_to_intrinsic_coordinates(ds[self._head-1])))
        #print(len(ds[0])**2)

        #self._buffer['mins'][self._head-1] = fdiis_findMinimumByQuadraticApproximation(d, mf, self._int2e_as_mat)
        #self._buffer['mindists'][self._head-1] = fdiis_getMinimumDistance(self._buffer['mins'][self._head-1], d)

        #print(self._buffer['mindists'])

        # DEBUG FOR TIMESCALE JUDGEMENT
        print("ENERGY: " + str(etot))
        print("Energy improvement: " + str(energy_improvement))
        
        print("DIIS Initialising time: " + str(chkTime2-chkTime1))
        print("FDIIS time: " + str(chkTime3-chkTime2))
        print("DIIS Minimisation time: " + str(chkTime4-chkTime3))
        print("DIIS Fock time: " + str(chkTime5-chkTime4))

        return fock

def fdiis_verifyDms(mf, ds):
    for i in range(len(ds)):
        print("(" + str(i) + ") / Tr: " + str(numpy.trace(mf.get_ovlp() @ ds[i])))

def fdiis_getPertubedMOCoeffs2(mf, dm, f, s, pertubationStrength):
    mo_occ, mo_unit = scipy.linalg.eig(mf.get_ovlp() @ dm)
    matrix = 0.5 * mf.get_ovlp() @ dm

    numpy.testing.assert_allclose(matrix @ matrix, matrix, atol=10**-10)

def fdiis_getPertubedMOCoeffs(mf, f, s, pertubationStrength, normalisationConst):
    mo_energy, mo_coeffs = mf.eig(f, s)


    nb = mf.mol.nao_nr()
    pertubed_mo_coeffs = numpy.zeros((2*nb, nb, nb))
    orientationCoeff = 1
    for i in range(2*nb):
        pertubed_mo_coeffs[i] = mo_coeffs
        pertubed_mo_coeffs[i][int(0.5*i+0.25*orientationCoeff-0.25)] += pertubationStrength * orientationCoeff
        orientationCoeff *= -1
    
    pertubed_mo_coeffs *= (numpy.linalg.norm(pertubed_mo_coeffs) * numpy.linalg.norm(pertubed_mo_coeffs.T))**(-0.5) * normalisationConst


    return mo_energy, pertubed_mo_coeffs

def fdiis_getPertubedDMFromMOCoeffs(mf, pertubedCoeffs, energies, d):
    nb = mf.mol.nao_nr()
    mo_occ = mf.get_occ(energies, pertubedCoeffs)
    pertubedDms = numpy.zeros((2*nb+1, nb, nb))

    for i in range(2*nb):
        mo_occ = mf.get_occ(energies, pertubedCoeffs[i])
        pertubedDms[i] = mf.make_rdm1(pertubedCoeffs[i], mo_occ)

    pertubedDms[2*nb] = d
    
    return pertubedDms


def fdiis_getPertubedFocksAndEnergies(mf1, pertubedMatrices, f, etot):
    pertubedFocks = numpy.zeros(pertubedMatrices.shape)
    pertubedEnergies = numpy.zeros((mf1.mol.nao_nr()*2-1))
    for i in range(mf1.mol.nao_nr()*2-2):
        pertubedFocks[i] = mf1.get_fock(None, None, None, pertubedMatrices[i], -1, None, None, None, None)
        pertubedEnergies[i] = mf1.energy_elec(pertubedMatrices[i], None, None)[0]

    pertubedFocks[mf1.mol.nao_nr()*2-2] = f
    pertubedEnergies[mf1.mol.nao_nr()*2-2] = etot

    return pertubedFocks, pertubedEnergies

def fdiis_genPertubationMatrices(mf, dm, pertubative_strength):

    ovlp = mf.get_ovlp()
    ovlp_inv = numpy.linalg.inv(ovlp)

    pseudo_dm = 0.5 * ovlp @ dm


    s = numpy.sin(pertubative_strength)
    c = numpy.cos(pertubative_strength)
    bn = mf.mol.nao_nr()


    pertubationMatricesArray = numpy.zeros((bn * 2-1, bn, bn))

    #print(pertubationMatricesArray.shape)
    #print(dm.shape)

    orientationCoeff = 1
    orientationDir = 0

    for orientation in range(bn * 2-2):
        for icoord in range(bn):
            for jcoord in range(bn):
                # 'OCTAHEDRAL' ansatz
                

                # THE o-th MATRIX is a ROTATION MATRIX AROUND THE SPACE BUILT FROM ALL COORDS EXCEPT o AND o+1
                if (icoord == orientationDir and jcoord == orientationDir) or (icoord == orientationDir+1 and jcoord == orientationDir+1):
                    pertubationMatricesArray[orientation][icoord][jcoord] = c
                elif (icoord == orientationDir and jcoord == orientationDir+1):
                    pertubationMatricesArray[orientation][icoord][jcoord] = orientationCoeff * s
                elif (icoord == orientationDir+1 and jcoord == orientationDir):
                    pertubationMatricesArray[orientation][icoord][jcoord] = -orientationCoeff * s
                elif icoord == jcoord:
                    pertubationMatricesArray[orientation][icoord][jcoord] = 1.0



        
        if orientationCoeff == -1:
            orientationDir += 1
        orientationCoeff *= -1

    #print(pertubationMatricesArray)
    pertubedMatrices = numpy.zeros(pertubationMatricesArray.shape)

    for i in range(bn * 2-2):
        pertubedMatrices[i] = 2 * ovlp_inv @ numpy.linalg.inv(pertubationMatricesArray[i]) @ pseudo_dm @ pertubationMatricesArray[i]
    pertubedMatrices[bn*2-2] = dm
    
    
    return pertubedMatrices


def fdiis_getMinimumDistance(min_dm, dm):
    # Two possible Ansatz: Frobenius (L2) norm or L_inf norm

    # L2 ansatz:
    #return numpy.linalg.norm(dm - min_dm)

    # L_inf ansatz:
    return numpy.linalg.norm(numpy.subtract(min_dm, dm), numpy.inf)

def fdiis_getInt2eAsMat(mf):
    # Calculating the 'scf hessian' i. e. matrix of antisymmetrised 2e integrals, which correspond to second derivatives with regards to electronic coordinates in HF exactly

    # Full N**4 / N**4 matrix of integrals in chemist notation
    #integrals2e_irred = mf.mol.intor('int2e', aosym='s1')
    # Full N**4 / N**4 matrix of antisymmetised 2e integrals
    #integrals2e_asymm = numpy.transpose(integrals2e_irred, (0, 2, 1, 3)) -  numpy.transpose(integrals2e_irred, (0, 3, 1, 2))
    #return integrals2e_asymm

    #return pyscf.ao2mo.restore(1, pyscf.scf._vhf.int2e_sph(mf.mol._atm, mf.mol._bas, mf.mol._env), mf.mol.nao_nr())
    

    # Non-antisymmetric 2e integrals
    int2e_nasymm = pyscf.ao2mo.restore(1, mf._eri, mf.mol.nao_nr())

    return numpy.subtract(int2e_nasymm, numpy.transpose(int2e_nasymm, (0, 2, 1, 3)))


def fdiis_getGradient(mf, dm, hessian):
    # GRADIENT_(ab) = h_(ab) + \sum_ij DM_ji (ib || ja)
    
    g1 = numpy.transpose(mf.get_hcore(), (1, 0))
    g2 = numpy.einsum('ji,ibja->ab', dm, hessian)
    return g1 + g2

def fdiis_findMinimumByQuadraticApproximation(dm, mf, hessian):
    # Generation of an intermediate minimum guess by second order Taylor approximation
    # In HF: Hessian tensor (rank 4) is just antisymmetrised 2e-integrals.


    #integrals2e_red = mf.mol.intor('int2e', aosym='s8')
    #integrals2e_irred = addons.restore(1, integrals2e_red, mf.mol.nao_nr())
    
    #integrals2e_asymm = integrals2e_irred - numpy.transpose(integrals2e_irred, (0, 3, 2, 1))
    # (ij || kl) -> H_(Ni+j)(Nk+l)
    #integrals2e_asymm_matrix = numpy.reshape(integrals2e_asymm, (mf.mol.nao_nr()**2, mf.mol.nao_nr()**2))
    #integrals2e_asymm_eigvalues, integrals2e_asymm_eigvectors = numpy.linalg.eig(integrals2e_asymm_matrix)

    #print(integrals2e_asymm)

    dm_gradient = fdiis_getGradient(mf, dm, hessian)
    #print(dm_gradient)
    #print(numpy.linalg.norm(dm_gradient, numpy.inf))
    return numpy.linalg.tensorsolve(hessian, -dm_gradient)


def fdiis_verifyEnergyFunctional(mf, dm):
    core_ham = mf.get_hcore()
    e1 = numpy.einsum('ij,ji->', core_ham, dm)
    e2 = numpy.einsum('ji,lk,ijkl->', dm, dm, fdiis_getInt2eAsMat(mf))

    return e1 + e2 * 0.5

#def fdiis_convert_to_intrinsic_coordinates(dm):
#    dm_eigvals, dm_eigvecs = numpy.linalg.eig(dm)
#    #dm_eigvecs = numpy.transpose(dm_eigvecs)
#    dm_intrinsic_coordinates = numpy.insert(dm_eigvecs[:len(dm_eigvals)-1], 0, dm_eigvals, axis=1)
#    print(dm_intrinsic_coordinates)
#    return dm_intrinsic_coordinates

#def fdiis_convert_to_extrinsic_coordinates(intcoords):
#    dim = int((len(intcoords[0])))
#    dm_eigvals = intcoords[0]
#    dm_eigvecs_red = intcoords[1:]
#    print(dm_eigvals)
#    print(dm_eigvecs_red)



 #   print(len(dm_eigvecs_red))
    #dm_eig_matrix = numpy.zeros((dim-1, dim), dtype=float)
    #for i in range(dim-1):
    #    print(str((i+1)*dim-1) + " / " + str(len(dm_eigvecs_red)))
    #    dm_eig_matrix[i] = dm_eigvecs_red[i*dim:(i+1)*dim-1]    
    #dm_eig_matrix = numpy.transpose(dm_eig_matrix)
    #print(dm_eig_matrix)
    #print(len(dm_eig_matrix))
    #print(len(dm_eig_matrix[0]))
    #dm_eig_matrix = numpy.delete(dm_eig_matrix, 0, 0)
    #bvec = numpy.negative(dm_eigvecs_red[:dim-1])
    #zvec = numpy.linalg.solve(dm_eig_matrix, bvec)
    #last_eigvec = numpy.concatenate(([1.0], zvec))
    #print(last_eigvec)
    


def ediis_minimize(es, ds, fs):
    nx = es.size
    nao = ds.shape[-1]
    ds = ds.reshape(nx,-1,nao,nao)
    fs = fs.reshape(nx,-1,nao,nao)
    df = numpy.einsum('inpq,jnqp->ij', ds, fs).real
    diag = df.diagonal()
    df = diag[:,None] + diag - df - df.T

    def costf(x):
        c = x**2 / (x**2).sum()
        return numpy.einsum('i,i', c, es) - numpy.einsum('i,ij,j', c, df, c)

    def grad(x):
        x2sum = (x**2).sum()
        c = x**2 / x2sum
        fc = es - 2*numpy.einsum('i,ik->k', c, df)
        cx = numpy.diag(x*x2sum) - numpy.einsum('k,n->kn', x**2, x)
        cx *= 2/x2sum**2
        return numpy.einsum('k,kn->n', fc, cx)

    if DEBUG:
        x0 = numpy.random.random(nx)
        dfx0 = numpy.zeros_like(x0)
        for i in range(nx):
            x1 = x0.copy()
            x1[i] += 1e-4
            dfx0[i] = (costf(x1) - costf(x0))*1e4
        print((dfx0 - grad(x0)) / dfx0)

    res = scipy.optimize.minimize(costf, numpy.ones(nx), method='BFGS',
                                  jac=grad, tol=1e-9)
    return res.fun, (res.x**2)/(res.x**2).sum()


class ADIIS(lib.diis.DIIS):
    '''
    Ref: JCP 132, 054109 (2010); DOI:10.1063/1.3304922
    '''
    def update(self, s, d, f, mf, h1e, vhf):
        if self._head >= self.space:
            self._head = 0
        if not self._buffer:
            shape = (self.space,) + f.shape
            self._buffer['dm'  ] = numpy.zeros(shape, dtype=f.dtype)
            self._buffer['fock'] = numpy.zeros(shape, dtype=f.dtype)
        self._buffer['dm'  ][self._head] = d
        self._buffer['fock'][self._head] = f

        ds = self._buffer['dm'  ]
        fs = self._buffer['fock']
        fun, c = adiis_minimize(ds, fs, self._head)
        if self.verbose >= logger.DEBUG1:
            etot = mf.energy_elec(d, h1e, vhf)[0] + fun
            logger.debug1(self, 'E %s  diis-c %s ', etot, c)
        fock = numpy.einsum('i,i...pq->...pq', c, fs)
        self._head += 1
        return fock

def adiis_minimize(ds, fs, idnewest):
    nx = ds.shape[0]
    nao = ds.shape[-1]
    ds = ds.reshape(nx,-1,nao,nao)
    fs = fs.reshape(nx,-1,nao,nao)
    df = numpy.einsum('inpq,jnqp->ij', ds, fs).real
    d_fn = df[:,idnewest]
    dn_f = df[idnewest]
    dn_fn = df[idnewest,idnewest]
    dd_fn = d_fn - dn_fn
    df = df - d_fn[:,None] - dn_f + dn_fn

    def costf(x):
        c = x**2 / (x**2).sum()
        return (numpy.einsum('i,i', c, dd_fn) * 2 +
                numpy.einsum('i,ij,j', c, df, c))

    def grad(x):
        x2sum = (x**2).sum()
        c = x**2 / x2sum
        fc = 2*dd_fn
        fc+= numpy.einsum('j,kj->k', c, df)
        fc+= numpy.einsum('i,ik->k', c, df)
        cx = numpy.diag(x*x2sum) - numpy.einsum('k,n->kn', x**2, x)
        cx *= 2/x2sum**2
        return numpy.einsum('k,kn->n', fc, cx)

    if DEBUG:
        x0 = numpy.random.random(nx)
        dfx0 = numpy.zeros_like(x0)
        for i in range(nx):
            x1 = x0.copy()
            x1[i] += 1e-4
            dfx0[i] = (costf(x1) - costf(x0))*1e4
        print((dfx0 - grad(x0)) / dfx0)
        

    res = scipy.optimize.minimize(costf, numpy.ones(nx), method='BFGS',
                                  jac=grad, tol=1e-9)
    return res.fun, (res.x**2)/(res.x**2).sum()

