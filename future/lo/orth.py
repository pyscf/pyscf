#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>

from functools import reduce
import numpy
import scipy.linalg
import pyscf.lib.logger as log

def lowdin(s):
    ''' new basis is |mu> c^{lowdin}_{mu i} '''
    e, v = numpy.linalg.eigh(s)
    return numpy.dot(v/numpy.sqrt(e), v.T.conj())

def schmidt(s):
    c = numpy.linalg.cholesky(s)
    return scipy.linalg.solve_triangular(c, numpy.eye(c.shape[1]), lower=True,
                                         overwrite_b=False).T.conj()

def vec_lowdin(c, metric=1):
    ''' lowdin orth for the metric c.T*metric*c and get x, then c*x'''
    #u, w, vh = numpy.linalg.svd(c)
    #return numpy.dot(u, vh)
    # svd is slower than eigh
    return numpy.dot(c, lowdin(reduce(numpy.dot, (c.T,metric,c))))

def vec_schmidt(c, metric=1):
    ''' schmidt orth for the metric c.T*metric*c and get x, then c*x'''
    if isinstance(metric, numpy.ndarray):
        return numpy.dot(c, schmidt(reduce(numpy.dot, (c.T,metric,c))))
    else:
        return numpy.linalg.qr(c)[0]

def weight_orth(s, weight):
    ''' new basis is |mu> c_{mu i}, c = w[(wsw)^{-1/2}]'''
    s1 = weight[:,None] * s * weight
    c = lowdin(s1)
    return weight[:,None] * c


def pre_orth_ao(mol):
    return pre_orth_ao_atm_scf(mol)
def pre_orth_ao_atm_scf(mol):
    from pyscf.scf import atom_hf
    atm_scf = atom_hf.get_atm_nrhf(mol)
    nbf = mol.nao_nr()
    c = numpy.zeros((nbf, nbf))
    p0 = 0
    for ia in range(mol.natm):
        symb = mol.atom_symbol(ia)
        if symb in atm_scf:
            e_hf, mo_e, mo_occ, mo_c = atm_scf[symb]
        else:
            symb = mol.atom_pure_symbol(ia)
            e_hf, mo_e, mo_occ, mo_c = atm_scf[symb]
        p1 = p0 + mo_e.size
        c[p0:p1,p0:p1] = mo_c
        p0 = p1
    log.debug(mol, 'use SCF AO instead of input basis')
    return c


def orth_ao(mol, method='meta_lowdin', pre_orth_ao=None, scf_method=None):
    from pyscf.lo import nao
    s = mol.intor_symmetric('cint1e_ovlp_sph')

    if pre_orth_ao is None:
        nbf = mol.nao_nr()
        pre_orth_ao = numpy.eye(nbf)

    if method == 'lowdin':
        s1 = reduce(numpy.dot, (pre_orth_ao.T, s, pre_orth_ao))
        c_orth = numpy.dot(pre_orth_ao, lowdin(s1))
    elif method == 'nao':
        c_orth = nao.nao(mol, scf_method)
    else: # meta_lowdin: divide ao into core, valence and Rydberg sets,
          # orthogonalizing within each set
        weight = numpy.ones(pre_orth_ao.shape[0])
        c_orth = nao._nao_sub(mol, weight, pre_orth_ao)
    # adjust phase
    sc = numpy.dot(s, c_orth)
    for i in range(c_orth.shape[1]):
        if sc[i,i] < 0:
            c_orth[:,i] *= -1
    return c_orth

if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf.lo import nao
    mol = gto.Mole()
    mol.verbose = 1
    mol.output = 'out_orth'
    mol.atom.extend([
        ['O' , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)] ])
    mol.basis = {'H': '6-31g',
                 'O': '6-31g',}
    mol.build()

    mf = scf.RHF(mol)
    mf.scf()

    c0 = nao.prenao(mol, mf.make_rdm1())
    c = orth_ao(mol, 'meta_lowdin', c0)

    s = mol.intor_symmetric('cint1e_ovlp_sph')
    p = reduce(numpy.dot, (s, mf.make_rdm1(), s))
    print(reduce(numpy.dot, (c.T, p, c)).diagonal())
