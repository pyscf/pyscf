#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Intrinsic Atomic Orbitals
ref. JCTC, 9, 4834
'''

import numpy
import scipy.linalg
from pyscf import lib
from pyscf import gto
from pyscf import scf
import pyscf.lib.parameters
import pyscf.scf.addons
import pyscf.scf.atom_hf
import orth


# Alternately, use ANO for minao
# orthogonalize iao by orth.lowdin_orth_coeff(c.T*mol.intor(ovlp)*c)

# simply project free atom into the given space
def simple_preiao(mol, minao='minao'):
    atmlst = set([gto.mole._rm_digit(gto.mole._symbol(k)) \
                  for k in mol.basis.keys()])
    basis = {}
    for symb in atmlst:
        basis[symb] = minao_basis(symb)

    pmol = gto.Mole()
    pmol._atm, pmol._bas, pmol._env = pmol.make_env(mol.atom, basis, [])
    c = addons.project_mo_nr2nr(pmol, 1, mol)
    return c

def pre_atm_scf_ao(mol):
    atm_scf = scf.atom_hf.get_atm_nrhf_result(mol)
    cs = []
    for ia in range(mol.natm):
        symb = mol.symbol_of_atm(ia)
        if atm_scf.has_key(symb):
            cs.append(atm_scf[symb][3])
        else:
            symb = mol.pure_symbol_of_atm(ia)
            cs.append(atm_scf[symb][3])
    return scipy.linalg.block_diag(*cs)


def preiao(mol, mocc, minao='minao'):
    pmol = minao_mol(mol, minao)
    naomin = pmol.nao_nr()
    pmol._atm, pmol._bas, pmol._env = \
            gto.mole.conc_env(pmol._atm, pmol._bas, pmol._env, \
                              mol._atm, mol._bas, mol._env)
    s12 = pmol.intor_cross('cint1e_ovlp_sph', range(pmol.nbas),
                           range(pmol.nbas, pmol.nbas+mol.nbas))
    sblock_inv = numpy.linalg.inv(sblock_minao(mol, minao))
    s = mol.intor_symmetric('cint1e_ovlp_sph')
    sinv = numpy.linalg.inv(s)

    # C^T<u|a>S^{-1}<a|u>
    pc1 = reduce(numpy.dot, (mocc.T, s12.T, sblock_inv, s12))

    a = simple_iao(mol, minao)
    pa = reduce(numpy.dot, (pc1.T, pc1, a))
    c = a - reduce(numpy.dot, (mocc, mocc.T, s, a)) \
      - numpy.dot(sinv, pa) + reduce(numpy.dot, (mocc, mocc.T, pa)) * 2
    return c


def sblock_minao(mol, minao):
    s = [minao_atm(mol, ia, minao).intor_symmetric('cint1e_ovlp_sph') \
         for ia in range(mol.natm)]
    return scipy.linalg.block_diag(*s)

def minao_basis(symb, minao):
    basis_add = gto.basis.load(minao, symb)
    basis_new = []
    for l in range(4):
        nuc = gto.mole._charge(symb)
        ne = lib.parameters.ELEMENTS[nuc][2][l]
        nd = (l * 2 + 1) * 2
        nshell = int(numpy.ceil(float(ne)/nd))
        if nshell > 0:
            basis_new.append([l] + [b[:nshell+1] for b in basis_add[l][1:]])
    return basis_new

def minao_atm(mol, atm_id, minao='minao'):
    symb = mol.pure_symbol_of_atm(atm_id)
    atm = gto.Mole()
    atm._atm, atm._bas, atm._env = \
            atm.make_env([mol.atom[atm_id]],
                         {symb: minao_basis(symb, minao)}, [])
    atm.natm = len(atm._atm)
    atm.nbas = len(atm._bas)
    return atm

def minao_mol(mol, minao='minao'):
    atmlst = set([gto.mole._rm_digit(gto.mole._symbol(k)) \
                  for k in mol.basis.keys()])
    basis = {}
    for symb in atmlst:
        basis[symb] = minao_basis(symb)

    pmol = gto.Mole()
    pmol._atm, pmol._bas, pmol._env = pmol.make_env(mol.atom, basis, [])
    pmol.natm = len(pmol._atm)
    pmol.nbas = len(pmol._bas)
    return pmol

def simple_1iao(mol, atm_id, minao='minao'):
    atm = minao_atm(mol, atm_id, minao)
    c1 = scf.addons.project_mo_nr2nr(atm, 1, mol)
    s = reduce(numpy.dot, (c1.T, mol.intor_symmetric('cint1e_ovlp_sph'), c1))
    return orth.lowdin_orth_coeff(s)


if __name__ == "__main__":
    mol = gto.Mole()
    mol.verbose = 0
    mol.output = 'out_iao'
    mol.atom.extend([
        ['O' , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)] ])
    mol.basis = {'H': 'cc-pvtz',
                 'O': 'cc-pvtz',}
    mol.build()
    c = simple_preiao(mol, 0)
    #print(c)

#    mf = scf.RHF(mol)
#    mf.scf()


