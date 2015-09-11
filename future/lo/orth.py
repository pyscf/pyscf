#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>

from functools import reduce
import numpy
import scipy.linalg
import pyscf.lib.parameters as param
from pyscf import gto

def lowdin(s):
    ''' new basis is |mu> c^{lowdin}_{mu i} '''
    e, v = scipy.linalg.eigh(s)
    idx = e > 1e-15
    return numpy.dot(v[:,idx]/numpy.sqrt(e[idx]), v[:,idx].T.conj())

def schmidt(s):
    c = numpy.linalg.cholesky(s)
    return scipy.linalg.solve_triangular(c, numpy.eye(c.shape[1]), lower=True,
                                         overwrite_b=False).T.conj()

def vec_lowdin(c, s=1):
    ''' lowdin orth for the metric c.T*s*c and get x, then c*x'''
    #u, w, vh = numpy.linalg.svd(c)
    #return numpy.dot(u, vh)
    # svd is slower than eigh
    return numpy.dot(c, lowdin(reduce(numpy.dot, (c.T,s,c))))

def vec_schmidt(c, s=1):
    ''' schmidt orth for the metric c.T*s*c and get x, then c*x'''
    if isinstance(s, numpy.ndarray):
        return numpy.dot(c, schmidt(reduce(numpy.dot, (c.T,s,c))))
    else:
        return numpy.linalg.qr(c)[0]

def weight_orth(s, weight):
    ''' new basis is |mu> c_{mu i}, c = w[(wsw)^{-1/2}]'''
    s1 = weight[:,None] * s * weight
    c = lowdin(s1)
    return weight[:,None] * c


def pre_orth_ao(mol, method='ANO'):
    '''Localized GTOs for each atom.  Possible localized methods include
    the ANO/MINAO projected basis or fraction-averaged RHF'''
    if method.upper() in ('ANO', 'MINAO'):
# Use ANO/MINAO basis to define the strongly occupied set
        return pre_orth_project_ano(mol, method)
    else:
        return pre_orth_ao_atm_scf(mol)

def pre_orth_project_ano(mol, basname):
    import pyscf.scf
    def search_atm_l(atm, l):
        idx = []
        p0 = 0
        for ib, b in enumerate(atm._bas):
            l0 = atm.bas_angular(ib)
            nctr = atm.bas_nctr(ib)
            if l0 == l:
                idx.extend(range(p0,p0+(2*l+1)*nctr))
            p0 += (2*l0+1) * nctr
        return idx

    aos = {}
    for symb in mol._basis.keys():
        rawsymb = gto.mole._rm_digit(symb)
        stdsymb = param.ELEMENTS[gto.mole._ELEMENTDIC[rawsymb.upper()]][0]
        atm = gto.Mole()
        atm._atm, atm._bas, atm._env = \
                atm.make_env([[stdsymb,(0,0,0)]], {stdsymb:mol._basis[symb]}, [])

        if 'GHOST' in symb.upper():
            aos[symb] = numpy.eye(atm.nao_nr())
            continue

        s0 = atm.intor_symmetric('cint1e_ovlp_sph')

        basis_add = gto.basis.load(basname, stdsymb)
        atmp = gto.Mole()
        atmp._atm, atmp._bas, atmp._env = \
                atmp.make_env([[stdsymb,(0,0,0)]], {stdsymb:basis_add}, [])
        ano = pyscf.scf.addons.project_mo_nr2nr(atmp, 1, atm)
        rm_ano = numpy.eye(ano.shape[0]) - reduce(numpy.dot, (ano, ano.T, s0))
        c = rm_ano.copy()
        for l in range(4):
            idx  = numpy.array(search_atm_l(atm, l))
            idxp = numpy.array(search_atm_l(atmp, l))
            if len(idx) > len(idxp) > 0:
# For angular l, first place the projected ANO, then the rest AOs.
                sdiag = reduce(numpy.dot, (rm_ano[:,idx].T, s0, rm_ano[:,idx])).diagonal()
                c[:,idx[:len(idxp)]] = ano[:,idxp]
                c[:,idx[len(idxp):]] = rm_ano[:,numpy.argsort(sdiag)[len(idxp):]]
            elif len(idxp) >= len(idx) > 0:
                c[:,idx] = ano[:,idxp[:len(idx)]]
        aos[symb] = c

    nao = mol.nao_nr()
    c = numpy.zeros((nao,nao))
    p0 = 0
    for ia in range(mol.natm):
        symb = mol.atom_symbol(ia)
        if symb in mol._basis:
            ano = aos[symb]
        else:
            symb = mol.atom_pure_symbol(ia)
            ano = aos[symb]
        p1 = p0 + ano.shape[1]
        c[p0:p1,p0:p1] = ano
        p0 = p1
    return c

def pre_orth_ao_atm_scf(mol):
    from pyscf.scf import atom_hf
    atm_scf = atom_hf.get_atm_nrhf(mol)
    nbf = mol.nao_nr()
    c = numpy.zeros((nbf, nbf))
    p0 = 0
    for ia in range(mol.natm):
        symb = mol.atom_symbol(ia)
        if symb in atm_scf:
            e_hf, mo_e, mo_c, mo_occ = atm_scf[symb]
        else:
            symb = mol.atom_pure_symbol(ia)
            e_hf, mo_e, mo_c, mo_occ = atm_scf[symb]
        p1 = p0 + mo_e.size
        c[p0:p1,p0:p1] = mo_c
        p0 = p1
    return c


def orth_ao(mol, method='meta_lowdin', pre_orth_ao=None, scf_method=None,
            s=None):
    from pyscf.lo import nao
    if s is None:
        s = mol.intor_symmetric('cint1e_ovlp_sph')

    if pre_orth_ao is None:
#        pre_orth_ao = numpy.eye(mol.nao_nr())
        pre_orth_ao = pre_orth_project_ano(mol, 'ANO')

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
    for i in range(c_orth.shape[1]):
        if c_orth[i,i] < 0:
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
