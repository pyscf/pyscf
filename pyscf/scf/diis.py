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
        t1 = time.time()
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
        t2 = time.time()
        print("EDIIS time: " + str(t2-t1))
        return fock


class FDIIS(lib.diis.DIIS):
    '''SCF-EDIIS
    Ref: JCP 116, 8255 (2002); DOI:10.1063/1.1470195
    '''
    # 2 electron integrals
    _int2e_as_mat = None
    
    # FDIIS latency
    _use_dynamic_latency = False
    _dynamic_latency_thresh = 5
    _use_dynamic_scaling = False
    _dynamic_scaling_power = 1.2
    _dynamic_scaling_max_factor = 10
    _low_dynamic_latency_thresh = 1
    _static_latency_thresh = 5
    _fdiis_octahedral_breadth = 0.8
    _slashing_coeff = 1.0
    _fdiis_mo_coeff_norm = 0
    _cycle = 0
    

    # parallel adjusted timing
    parallel_adjusted_timer = 0.0

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
        self._cycle += 1

        lowerIndex = self._head - 2
        if lowerIndex == -1:
            lowerIndex = self.space-1

        
        energy_improvement = self._buffer['etot'][self._head-1] - self._buffer['etot'][lowerIndex] 

        ds = self._buffer['dm'  ]
        fs = self._buffer['fock']
        es = self._buffer['etot']

        # Number of used matrices = self.space + number of artifical matrices
        numOfUsedMatrices = self.space + mf.mol.nao_nr() * 2 - 1

        chkTime2 = time.time()
        useFdiis = (self._use_dynamic_latency and abs(energy_improvement) <= mf.conv_tol * 10**self._dynamic_latency_thresh and abs(energy_improvement) >= mf.conv_tol * 10**self._low_dynamic_latency_thresh) or ((not self._use_dynamic_latency) and self._cycle >= self._static_latency_thresh)
        if useFdiis:
            print("FDIIS active, Cycle: " + str(self._cycle))
            #fdiis_verifyDms(mf, ds)
            #numOfUsedMatricesA = numpy.argwhere(abs(self._buffer['etot']) <= 10**-30).ravel()
            #numOfUsedMatrices = self.space
            #if len(numOfUsedMatricesA) > 0:
            #    numOfUsedMatrices = numOfUsedMatricesA[0]
            #if abs(self._buffer['etot'][self.space-1]) <= -1:
            #    numOfUsedMatrices = self.space
            pertubationStrength = self._fdiis_octahedral_breadth
            if self._use_dynamic_scaling:
                perScale = self._fdiis_octahedral_breadth * (1 - self._dynamic_scaling_max_factor) / (self._dynamic_scaling_power**(numpy.log(mf.conv_tol)) - 1)
                perShift = self._dynamic_scaling_max_factor * self._fdiis_octahedral_breadth - perScale
                pertubationStrength = perScale * self._dynamic_scaling_power**(numpy.log(abs(energy_improvement))) + perShift
                print("PS: " + str(pertubationStrength))
            pds, self.parallel_adjusted_timer = fdiis_genPertubationMatrices(mf, d, s, self._fdiis_octahedral_breadth, self._slashing_coeff, ds, numOfUsedMatrices, self.parallel_adjusted_timer)
            fs, es, ds, self.parallel_adjusted_timer = fdiis_getPertubedFocksAndEnergiesAndDms(mf, pds, fs, h1e, s, es, ds, self.parallel_adjusted_timer)
            #print(es)

            #fs, ds, es = fdiis_propagateFocksOnce(mf, fs, ds, es, h1e, s)
        else:
            ds = numpy.zeros((numOfUsedMatrices, mf.mol.nao_nr(), mf.mol.nao_nr()))
            ds[0:self.space] = self._buffer['dm']
            fs = numpy.zeros((numOfUsedMatrices, mf.mol.nao_nr(), mf.mol.nao_nr()))
            fs[0:self.space] = self._buffer['fock']
            es = numpy.zeros(numOfUsedMatrices)
            es[0:self.space] = self._buffer['etot']

        chkTime3 = time.time()
        

        if useFdiis:
            etot, c = fdiis_minimize(es, ds, fs)
        else:
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
        print("PAT: " + str(self.parallel_adjusted_timer))


        return fock

def fdiis_propagateFocksOnce(mf, fs, ds, es, h1e, s1e):
    fs_prop = numpy.zeros(fs.shape)
    ds_prop = numpy.zeros(ds.shape)
    es_prop = numpy.zeros(es.shape)
    for i in range(len(fs)):
        mo_energy, mo_coeff = mf.eig(fs[i], s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        ds_prop[i] = mf.make_rdm1(mo_coeff, mo_occ)
        vhf = mf.get_veff(mf.mol, ds_prop[i], None, None)
        fs_prop[i] = mf.get_fock(h1e, s1e, vhf, ds_prop[i])  # = h1e + vhf, no DIIS
        es_prop[i] = mf.energy_tot(ds_prop[i], h1e, vhf)

    return fs_prop, ds_prop, es_prop

def fdiis_verifyDms(mf, ds):
    for i in range(len(ds)):
        print("(" + str(i) + ") / Tr: " + str(numpy.trace(mf.get_ovlp() @ ds[i])))


def fdiis_getPertubedFocksAndEnergiesAndDms(mf1, pertubedMatrices, fs, h1e, s, es, ds, pat):
    pertubedFocks = numpy.zeros(pertubedMatrices.shape)
    pertubedEnergies = numpy.zeros((len(pertubedMatrices)))
    pertubedDms = numpy.zeros(pertubedMatrices.shape)

    alreadyTimed = False


    for i in range(len(pertubedMatrices)):
        if i < len(ds):
            pertubedFocks[i] = fs[i]
            pertubedEnergies[i] = es[i]
            pertubedDms[i] = ds[i]
        else:
            t1 = time.time()

            
            vhf = mf1.get_veff(mf1.mol, pertubedMatrices[i], None, None)
            pertubedFocks[i] = mf1.get_fock(h1e, s, vhf, pertubedMatrices[i], -1, None, None, None, None)
            pertubedEnergies[i] = mf1.energy_elec(pertubedMatrices[i], h1e, vhf)[0]
            

            mo_energy, mo_coeff = mf1.eig(pertubedFocks[i], s)
            mo_occ = mf1.get_occ(mo_energy, mo_coeff)
            pertubedDms[i] = mf1.make_rdm1(mo_coeff, mo_occ)
            
            if not alreadyTimed:
                alreadyTimed = True
            else:
                pat += time.time() - t1

    #pertubedEnergies[len(pertubedMatrices)-1] = etot

    return pertubedFocks[:len(pertubedFocks)-1], pertubedEnergies[:len(pertubedEnergies)-1], pertubedDms[:len(pertubedDms)-1], pat

def fdiis_genPertubationMatrices(mf, dm, ovlp, pertubative_strength, slashing_coeff, ds, numOfUsedMatrices, pat):

    ovlp_inv = numpy.linalg.inv(ovlp)

    pseudo_dm = 0.5 * ovlp @ dm


    s = numpy.sin(pertubative_strength)
    c = numpy.cos(pertubative_strength)
    bn = mf.mol.nao_nr()

    numberOfMatrices = int(slashing_coeff * (bn*2-2)) + 1
    matrixRemainder = numberOfMatrices


    pertubationMatricesArray = numpy.zeros((numberOfMatrices, bn, bn))

    #print(dm.shape)

    orientationCoeff = 1
    orientationDir = 0

    alreadyTimed = False

    for orientation in range(bn*2-2):
        
        t1 = time.time()

        useMatrix = ((numpy.random.random() <= slashing_coeff) or (orientation + matrixRemainder >= bn*2-2)) and matrixRemainder > 0

        if useMatrix:
            for icoord in range(bn):
                for jcoord in range(bn):
                    # 'OCTAHEDRAL' ansatz
                

                    # THE o-th MATRIX is a ROTATION MATRIX AROUND THE SPACE BUILT FROM ALL COORDS EXCEPT o AND o+1
                    if (icoord == orientationDir and jcoord == orientationDir) or (icoord == orientationDir+1 and jcoord == orientationDir+1):
                        pertubationMatricesArray[numberOfMatrices-matrixRemainder][icoord][jcoord] = c
                    elif (icoord == orientationDir and jcoord == orientationDir+1):
                        pertubationMatricesArray[numberOfMatrices-matrixRemainder][icoord][jcoord] = orientationCoeff * s
                    elif (icoord == orientationDir+1 and jcoord == orientationDir):
                        pertubationMatricesArray[numberOfMatrices-matrixRemainder][icoord][jcoord] = -orientationCoeff * s
                    elif icoord == jcoord:
                        pertubationMatricesArray[numberOfMatrices-matrixRemainder][icoord][jcoord] = 1.0

            matrixRemainder -= 1


        
        if orientationCoeff == -1:
            orientationDir += 1
        orientationCoeff *= -1
        
        if not alreadyTimed:
            alreadyTimed = True
        else:
            pat += time.time() - t1

    #print(pertubationMatricesArray)
    pertubedMatrices = numpy.zeros((numOfUsedMatrices, bn, bn))

    pertubedMatrices[0:len(ds)] = ds
    
    alreadyTimed = False

    for i in range(len(ds),numOfUsedMatrices-1):
        t1 = time.time()
        
        pertubedMatrices[i] = 2 * ovlp_inv @ numpy.linalg.inv(pertubationMatricesArray[i-len(ds)]) @ pseudo_dm @ pertubationMatricesArray[i-len(ds)]
        
        if not alreadyTimed:
            alreadyTimed = True
        else:
            pat += time.time() - t1


    return pertubedMatrices, pat

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


def fdiis_minimize(es, ds, fs):
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

    res = scipy.optimize.minimize(costf, 0.5 * numpy.ones(nx), method='BFGS', jac=grad, tol=1e-9)
    coeffs = (res.x**2)/(res.x**2).sum()
    print(coeffs)
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

