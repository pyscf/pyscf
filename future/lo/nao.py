#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Natural atomic orbitals
Ref:
    F. Weinhold et al., J. Chem. Phys. 83(1985), 735-746
'''

from functools import reduce
import numpy
import scipy.linalg
import pyscf.lib.parameters
from pyscf.lo import param
from pyscf.lo import orth


def prenao(mol, dm):
    s = mol.intor_symmetric('cint1e_ovlp_sph')
    p = reduce(numpy.dot, (s, dm, s))
    return _prenao_sub(mol, p, s)[1]

def _prenao_sub(mol, p, s):
    idx = [[[]]*8 for i in range(mol.natm)]
    k = 0
    for ib in range(mol.nbas):
        ia = mol.atom_of_bas(ib)
        l = mol.angular_of_bas(ib)
        nc = mol.nctr_of_bas(ib)
        idx[ia][l] = idx[ia][l] + list(range(k, k+nc*(l*2+1)))
        k += nc * (l * 2 + 1)

    nao = mol.nao_nr()
    occ = numpy.zeros(nao)
    cao = numpy.zeros((nao,nao))

    for ia in range(mol.natm):
        for l, lst in enumerate(idx[ia]):
            if len(lst) < 1:
                continue
            degen = l * 2 + 1
            p_frag = _spheric_average_mat(p, l, lst)
            s_frag = _spheric_average_mat(s, l, lst)
            e, v = scipy.linalg.eigh(p_frag, s_frag)
            e = e[::-1]
            v = v[:,::-1]

            for k in range(degen):
                ilst = lst[k::degen]
                occ[ilst] = e
                for i,i0 in enumerate(ilst):
                    cao[i0,ilst] = v[i]
    return occ, cao

def _spheric_average_mat(mat, l, lst):
    degen = l * 2 + 1
    nd = len(lst) // degen
    t = scipy.linalg.block_diag(*([numpy.ones(degen)/numpy.sqrt(degen)]*nd))
    mat_frag = reduce(numpy.dot, (t,mat[lst,:][:,lst],t.T))
    return mat_frag


def nao(mol, mf, restore=True):
    dm = mf.make_rdm1()
    s = mol.intor_symmetric('cint1e_ovlp_sph')
    p = reduce(numpy.dot, (s, dm, s))
    pre_occ, pre_nao = _prenao_sub(mol, p, s)
    cnao = _nao_sub(mol, pre_occ, pre_nao)
    if restore:
        # restore natural character
        p_nao = reduce(numpy.dot, (cnao.T, p, cnao))
        s_nao = numpy.eye(p_nao.shape[0])
        cnao = numpy.dot(cnao, _prenao_sub(mol, p_nao, s_nao)[1])
    return cnao

def _nao_sub(mol, pre_occ, pre_nao):
    core_lst, val_lst, rydbg_lst = core_val_ryd_list(mol)
    s = mol.intor_symmetric('cint1e_ovlp_sph')
    nao = mol.nao_nr()
    cnao = numpy.zeros((nao,nao))

    if core_lst:
        c = pre_nao[:,core_lst]
        s1 = reduce(numpy.dot, (c.T, s, c))
        cnao[:,core_lst] = numpy.dot(c, orth.lowdin_orth_coeff(s1))

    c1 = cnao[:,core_lst]
    rm_core = numpy.eye(nao) - reduce(numpy.dot, (c1, c1.T, s))
    c = numpy.dot(rm_core, pre_nao[:,val_lst])
    s1 = reduce(numpy.dot, (c.T.conj(), s, c))
    wt = pre_occ[val_lst]
    cnao[:,val_lst] = numpy.dot(c, orth.weight_orthogonal(s1, wt))

    if rydbg_lst:
        c1 = cnao[:,val_lst]
        rm_val = numpy.eye(nao)-reduce(numpy.dot, (c1, c1.T, s))
        c = reduce(numpy.dot, (rm_core, rm_val, pre_nao[:,rydbg_lst]))
        s1 = reduce(numpy.dot, (c.T.conj(), s, c))
        cnao[:,rydbg_lst] = numpy.dot(c, orth.lowdin_orth_coeff(s1))
    return cnao

def core_val_ryd_list(mol):
    count = numpy.zeros((mol.natm, 9), dtype=int)
    core_lst = []
    val_lst = []
    rydbg_lst = []
    k = 0
    valenceof = lambda nuc, l: \
            int(numpy.ceil(pyscf.lib.parameters.ELEMENTS[nuc][2][l]/(4*l+2.)))
    for ib in range(mol.nbas):
        ia = mol.atom_of_bas(ib)
        nuc = mol.charge_of_atm(ia)
        l = mol.angular_of_bas(ib)
        nc = mol.nctr_of_bas(ib)
        for n in range(nc):
            if count[ia,l]+n < param.CORESHELL[nuc][l]:
                core_lst += list(range(k, k+(2*l+1)))
            elif count[ia,l]+n < valenceof(nuc, l):
                val_lst += list(range(k, k+(2*l+1)))
            else:
                rydbg_lst += list(range(k, k+(2*l+1)))
            k = k + 2*l+1
        count[ia,l] += nc
    return core_lst, val_lst, rydbg_lst


if __name__ == "__main__":
    from pyscf import gto
    from pyscf import scf
    mol = gto.Mole()
    mol.verbose = 1
    mol.output = 'out_nao'
    mol.atom.extend([
        ['O' , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)] ])
    mol.basis = {'H': '6-31g',
                 'O': '6-31g',}
    mol.build()

    mf = scf.RHF(mol)
    mf.scf()

    s = mol.intor_symmetric('cint1e_ovlp_sph')
    p = reduce(numpy.dot, (s, mf.make_rdm1(), s))
    o0, c0 = _prenao_sub(mol, p, s)
    print(o0)
    print(abs(c0).sum() - 21.848915907988854)

    c = nao(mol, mf)
    print(reduce(numpy.dot, (c.T, p, c)).diagonal())
    print(core_val_ryd_list(mol))
