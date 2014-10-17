#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>

import numpy

def lowdin_orth_coeff(s):
    ''' new basis is |mu> c^{lowdin}_{mu i} '''
    e, v = numpy.linalg.eigh(s)
    return numpy.dot(v/numpy.sqrt(e), v.T.conj())

def schmidt_orth_coeff(s):
    c = numpy.linalg.cholesky(s)
    return numpy.linalg.inv(c).T.conj()

def weight_orthogonal(s, weight):
    ''' new basis is |mu> c_{mu i}, c = w[(wsw)^{-1/2}]'''
    s1 = weight[:,None] * s * weight
    c = lowdin_orth_coeff(s1)
    return weight[:,None] * c


def orth_ao(mol, pre_orth_ao=None, method='meta_lowdin', scf_method=None):
    import nao
    s = mol.intor_symmetric('cint1e_ovlp_sph')

    if pre_orth_ao is None:
        nbf = mol.nao_nr()
        pre_orth_ao = numpy.eye(nbf)

    if method == 'lowdin':
        s1 = reduce(numpy.dot, (pre_orth_ao.T, s, pre_orth_ao))
        c_orth = numpy.dot(pre_orth_ao, lowdin_orth_coeff(s1))
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
    import nao
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

    c0 = nao.prenao(mol, mf.calc_den_mat())
    c = orth_ao(mol, c0, 'meta_lowdin')

    s = mol.intor_symmetric('cint1e_ovlp_sph')
    p = reduce(numpy.dot, (s, mf.calc_den_mat(), s))
    print reduce(numpy.dot, (c.T, p, c)).diagonal()
