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
from pyscf.lo import orth
from pyscf.lib import logger

# Note the valence space for Li, Be may need include 2p, Al..Cl may need 3d ...
AOSHELL = [
    # core       core+valence
# core+valence = lambda nuc, l: \
#            int(numpy.ceil(pyscf.lib.parameters.ELEMENTS[nuc][2][l]/(4*l+2.)))
    ['0s0p0d0f', '0s0p0d0f'],     #  0  GHOST
    ['0s0p0d0f', '1s0p0d0f'],     #  1  H
    ['0s0p0d0f', '1s0p0d0f'],     #  2  He
    ['1s0p0d0f', '2s0p0d0f'],     #  3  Li
    ['1s0p0d0f', '2s0p0d0f'],     #  4  Be
    ['1s0p0d0f', '2s1p0d0f'],     #  5  B
    ['1s0p0d0f', '2s1p0d0f'],     #  6  C
    ['1s0p0d0f', '2s1p0d0f'],     #  7  N
    ['1s0p0d0f', '2s1p0d0f'],     #  8  O
    ['1s0p0d0f', '2s1p0d0f'],     #  9  F
    ['1s0p0d0f', '2s1p0d0f'],     # 10  Ne
    ['2s1p0d0f', '3s1p0d0f'],     # 11  Na
    ['2s1p0d0f', '3s1p0d0f'],     # 12  Mg
    ['2s1p0d0f', '3s2p0d0f'],     # 13  Al
    ['2s1p0d0f', '3s2p0d0f'],     # 14  Si
    ['2s1p0d0f', '3s2p0d0f'],     # 15  P
    ['2s1p0d0f', '3s2p0d0f'],     # 16  S
    ['2s1p0d0f', '3s2p0d0f'],     # 17  Cl
    ['2s1p0d0f', '3s2p0d0f'],     # 18  Ar
    ['3s2p0d0f', '4s2p0d0f'],     # 19  K
    ['3s2p0d0f', '4s2p0d0f'],     # 20  Ca
    ['3s2p0d0f', '4s2p1d0f'],     # 21  Sc
    ['3s2p0d0f', '4s2p1d0f'],     # 22  Ti
    ['3s2p0d0f', '4s2p1d0f'],     # 23  V
    ['3s2p0d0f', '4s2p1d0f'],     # 24  Cr
    ['3s2p0d0f', '4s2p1d0f'],     # 25  Mn
    ['3s2p0d0f', '4s2p1d0f'],     # 26  Fe
    ['3s2p0d0f', '4s2p1d0f'],     # 27  Co
    ['3s2p0d0f', '4s2p1d0f'],     # 28  Ni
    ['3s2p0d0f', '4s2p1d0f'],     # 29  Cu
    ['3s2p0d0f', '4s2p1d0f'],     # 30  Zn
    ['3s2p1d0f', '4s3p1d0f'],     # 31  Ga
    ['3s2p1d0f', '4s3p1d0f'],     # 32  Ge
    ['3s2p1d0f', '4s3p1d0f'],     # 33  As
    ['3s2p1d0f', '4s3p1d0f'],     # 34  Se
    ['3s2p1d0f', '4s3p1d0f'],     # 35  Br
    ['3s2p1d0f', '4s3p1d0f'],     # 36  Kr
    ['4s3p1d0f', '5s3p1d0f'],     # 37  Rb
    ['4s3p1d0f', '5s3p1d0f'],     # 38  Sr
    ['4s3p1d0f', '5s3p2d0f'],     # 39  Y
    ['4s3p1d0f', '5s3p2d0f'],     # 40  Zr
    ['4s3p1d0f', '5s3p2d0f'],     # 41  Nb
    ['4s3p1d0f', '5s3p2d0f'],     # 42  Mo
    ['4s3p1d0f', '5s3p2d0f'],     # 43  Tc
    ['4s3p1d0f', '5s3p2d0f'],     # 44  Ru
    ['4s3p1d0f', '5s3p2d0f'],     # 45  Rh
    ['4s3p1d0f', '4s3p2d0f'],     # 46  Pd
    ['4s3p1d0f', '5s3p2d0f'],     # 47  Ag
    ['4s3p1d0f', '5s3p2d0f'],     # 48  Cd
    ['4s3p2d0f', '5s4p2d0f'],     # 49  In
    ['4s3p2d0f', '5s4p2d0f'],     # 50  Sn
    ['4s3p2d0f', '5s4p2d0f'],     # 51  Sb
    ['4s3p2d0f', '5s4p2d0f'],     # 52  Te
    ['4s3p2d0f', '5s4p2d0f'],     # 53  I
    ['4s3p2d0f', '5s4p2d0f'],     # 54  Xe
    ['5s4p2d0f', '6s4p2d0f'],     # 55  Cs
    ['5s4p2d0f', '6s4p2d0f'],     # 56  Ba
    ['5s4p2d0f', '6s4p3d0f'],     # 57  La
    ['5s4p2d0f', '6s4p3d1f'],     # 58  Ce
    ['5s4p2d0f', '6s4p2d1f'],     # 59  Pr
    ['5s4p2d0f', '6s4p2d1f'],     # 60  Nd
    ['5s4p2d0f', '6s4p2d1f'],     # 61  Pm
    ['5s4p2d0f', '6s4p2d1f'],     # 62  Sm
    ['5s4p2d0f', '6s4p2d1f'],     # 63  Eu
    ['5s4p2d0f', '6s4p3d1f'],     # 64  Gd
    ['5s4p2d0f', '6s4p3d1f'],     # 65  Tb
    ['5s4p2d0f', '6s4p2d1f'],     # 66  Dy
    ['5s4p2d0f', '6s4p2d1f'],     # 67  Ho
    ['5s4p2d0f', '6s4p2d1f'],     # 68  Er
    ['5s4p2d0f', '6s4p2d1f'],     # 69  Tm
    ['5s4p2d0f', '6s4p2d1f'],     # 70  Yb
    ['5s4p2d1f', '6s4p3d1f'],     # 71  Lu
    ['5s4p2d1f', '6s4p3d1f'],     # 72  Hf
    ['5s4p2d1f', '6s4p3d1f'],     # 73  Ta
    ['5s4p2d1f', '6s4p3d1f'],     # 74  W
    ['5s4p2d1f', '6s4p3d1f'],     # 75  Re
    ['5s4p2d1f', '6s4p3d1f'],     # 76  Os
    ['5s4p2d1f', '6s4p3d1f'],     # 77  Ir
    ['5s4p2d1f', '6s4p3d1f'],     # 78  Pt
    ['5s4p2d1f', '6s4p3d1f'],     # 79  Au
    ['5s4p2d1f', '6s4p3d1f'],     # 80  Hg
    ['5s4p3d1f', '6s5p3d1f'],     # 81  Tl
    ['5s4p3d1f', '6s5p3d1f'],     # 82  Pb
    ['5s4p3d1f', '6s5p3d1f'],     # 83  Bi
    ['5s4p3d1f', '6s5p3d1f'],     # 84  Po
    ['5s4p3d1f', '6s5p3d1f'],     # 85  At
    ['5s4p3d1f', '6s5p3d1f'],     # 86  Rn
    ['6s5p3d1f', '7s5p3d1f'],     # 87  Fr
    ['6s5p3d1f', '7s5p3d1f'],     # 88  Ra
    ['6s5p3d1f', '7s5p4d1f'],     # 89  Ac
    ['6s5p3d1f', '7s5p4d1f'],     # 90  Th
    ['6s5p3d1f', '7s5p4d2f'],     # 91  Pa
    ['6s5p3d1f', '7s5p4d2f'],     # 92  U
    ['6s5p3d1f', '7s5p4d2f'],     # 93  Np
    ['6s5p3d1f', '7s5p3d2f'],     # 94  Pu
    ['6s5p3d1f', '7s5p3d2f'],     # 95  Am
    ['6s5p3d1f', '7s5p4d2f'],     # 96  Cm
    ['6s5p3d1f', '7s5p4d2f'],     # 97  Bk
    ['6s5p3d1f', '7s5p3d2f'],     # 98  Cf
    ['6s5p3d1f', '7s5p3d2f'],     # 99  Es
    ['6s5p3d1f', '7s5p3d2f'],     #100  Fm
    ['6s5p3d1f', '7s5p3d2f'],     #101  Md
    ['6s5p3d1f', '7s5p3d2f'],     #102  No
    ['6s5p3d2f', '7s5p4d2f'],     #103  Lr
    ['6s5p3d2f', '7s5p4d2f'],     #104  Rf
    ['6s5p3d2f', '7s5p4d2f'],     #105  Db
    ['6s5p3d2f', '7s5p4d2f'],     #106  Sg
    ['6s5p3d2f', '7s5p4d2f'],     #107  Bh
    ['6s5p3d2f', '7s5p4d2f'],     #108  Hs
    ['6s5p3d2f', '7s5p4d2f'],     #109  Mt
    ['6s5p3d2f', '7s5p4d2f'],     #110  E110
    ['6s5p3d2f', '7s5p4d2f'],     #111  E111
    ['6s5p3d2f', '7s5p4d2f'],     #112  E112
    ['6s5p4d2f', '7s6p4d2f'],     #113  E113
    ['6s5p4d2f', '7s6p4d2f'],     #114  E114
    ['6s5p4d2f', '7s6p4d2f'],     #115  E115
    ['6s5p4d2f', '7s6p4d2f'],     #116  E116
    ['6s3p4d2f', '7s6p4d2f'],     #117  E117
    ['6s3p4d2f', '7s6p4d2f']      #118  E118
]

def prenao(mol, dm):
    s = mol.intor_symmetric('cint1e_ovlp_sph')
    p = reduce(numpy.dot, (s, dm, s))
    return _prenao_sub(mol, p, s)[1]

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


def _prenao_sub(mol, p, s):
    idx = [[[]]*8 for i in range(mol.natm)]
    k = 0
    for ib in range(mol.nbas):
        ia = mol.bas_atom(ib)
        l = mol.bas_angular(ib)
        nc = mol.bas_nctr(ib)
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

def _nao_sub(mol, pre_occ, pre_nao, s=None):
    if s is None:
        s = mol.intor_symmetric('cint1e_ovlp_sph')
    core_lst, val_lst, rydbg_lst = _core_val_ryd_list(mol)
    nbf = mol.nao_nr()
    cnao = numpy.empty((nbf,nbf))

    if core_lst:
        c = pre_nao[:,core_lst]
        s1 = reduce(numpy.dot, (c.T, s, c))
        cnao[:,core_lst] = c1 = numpy.dot(c, orth.lowdin(s1))
        c = pre_nao[:,val_lst]
        c -= reduce(numpy.dot, (c1, c1.T, s, c))
    else:
        c = pre_nao[:,val_lst]

    s1 = reduce(numpy.dot, (c.T, s, c))
    wt = pre_occ[val_lst]
    cnao[:,val_lst] = numpy.dot(c, orth.weight_orth(s1, wt))

    if rydbg_lst:
        cvlst = core_lst + val_lst
        c1 = cnao[:,cvlst]
        c = pre_nao[:,rydbg_lst]
        c -= reduce(numpy.dot, (c1, c1.T, s, c))
        s1 = reduce(numpy.dot, (c.T, s, c))
        cnao[:,rydbg_lst] = numpy.dot(c, orth.lowdin(s1))
    snorm = numpy.linalg.norm(reduce(numpy.dot, (cnao.T, s, cnao)) - numpy.eye(nbf))
    if snorm > 1e-9:
        logger.warn(mol, 'Weak orthogonality for localized orbitals %s', snorm)
    return cnao

def _core_val_ryd_list(mol):
    count = numpy.zeros((mol.natm, 9), dtype=int)
    core_lst = []
    val_lst = []
    rydbg_lst = []
    k = 0
    valenceof = lambda nuc, l: \
            int(numpy.ceil(pyscf.lib.parameters.ELEMENTS[nuc][2][l]/(4*l+2.)))
    for ib in range(mol.nbas):
        ia = mol.bas_atom(ib)
        nuc = mol.atom_charge(ia)
        l = mol.bas_angular(ib)
        nc = mol.bas_nctr(ib)
        for n in range(nc):
            coreshell = [int(x) for x in AOSHELL[nuc][0][::2]]
            cvshell = [int(x) for x in AOSHELL[nuc][1][::2]]
            if l > 3:
                rydbg_lst += list(range(k, k+(2*l+1)))
            elif count[ia,l]+n < coreshell[l]:
                core_lst += list(range(k, k+(2*l+1)))
            elif count[ia,l]+n < cvshell[l]:
                val_lst += list(range(k, k+(2*l+1)))
            else:
                rydbg_lst += list(range(k, k+(2*l+1)))
            k = k + 2*l+1
        count[ia,l] += nc
    return core_lst, val_lst, rydbg_lst

def _spheric_average_mat(mat, l, lst):
    degen = l * 2 + 1
    nd = len(lst) // degen
    t = scipy.linalg.block_diag(*([numpy.ones(degen)/numpy.sqrt(degen)]*nd))
    mat_frag = reduce(numpy.dot, (t,mat[lst,:][:,lst],t.T))
    return mat_frag


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
    print(_core_val_ryd_list(mol))
