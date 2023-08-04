#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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

'''
Natural atomic orbitals
Ref:
    F. Weinhold et al., J. Chem. Phys. 83(1985), 735-746
'''

import sys
from functools import reduce
import numpy
import scipy.linalg
from pyscf import lib
from pyscf import scf
from pyscf.gto import mole
from pyscf.lo import orth
from pyscf.lib import logger
from pyscf.data import elements
from pyscf import __config__

# Note the valence space for Li, Be may need include 2p, Al..Cl may need 3d ...
# This is No. of shells, not the atomic configuations
#     core       core+valence
# core+valence = lambda nuc, l: \
#            int(numpy.ceil(elements.CONFIGURATION[nuc][l]/(4*l+2.)))
AOSHELL = getattr(__config__, 'lo_nao_AOSHELL', None)
if AOSHELL is None:
    AOSHELL = list(zip(elements.N_CORE_SHELLS,
                       elements.N_CORE_VALENCE_SHELLS))

def prenao(mol, dm):
    if not (isinstance(dm, numpy.ndarray) and dm.ndim == 2):
        # UHF or ROHF
        dm = dm[0] + dm[1]

    if getattr(mol, 'pbc_intor', None):  # whether mol object is a cell
        s = mol.pbc_intor('int1e_ovlp', hermi=1)
    else:
        s = mol.intor_symmetric('int1e_ovlp')

    p = reduce(numpy.dot, (s, dm, s))
    return _prenao_sub(mol, p, s)[1]

def nao(mol, mf, s=None, restore=True):
    if s is None:
        if getattr(mol, 'pbc_intor', None):  # whether mol object is a cell
            s = mol.pbc_intor('int1e_ovlp', hermi=1)
        else:
            s = mol.intor_symmetric('int1e_ovlp')

    dm = mf.make_rdm1()
    if isinstance(mf, (scf.uhf.UHF, scf.rohf.ROHF)):
        dm = dm[0] + dm[1]

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
    ao_loc = mol.ao_loc_nr()
    nao = ao_loc[-1]
    occ = numpy.zeros(nao)
    cao = numpy.zeros((nao,nao), dtype=s.dtype)

    bas_ang = mol._bas[:,mole.ANG_OF]
    for ia, (b0,b1,p0,p1) in enumerate(mol.aoslice_by_atom(ao_loc)):
        l_max = bas_ang[b0:b1].max()
        for l in range(l_max+1):
            idx = []
            for ib in numpy.where(bas_ang[b0:b1] == l)[0]:
                idx.append(numpy.arange(ao_loc[b0+ib], ao_loc[b0+ib+1]))
            idx = numpy.hstack(idx)
            if idx.size < 1:
                continue

            if mol.cart:
                degen = (l + 1) * (l + 2) // 2
            else:
                degen = l * 2 + 1
            p_frag = _spheric_average_mat(p, l, idx, degen)
            s_frag = _spheric_average_mat(s, l, idx, degen)
            e, v = scipy.linalg.eigh(p_frag, s_frag)
            e = e[::-1]
            v = v[:,::-1]

            idx = idx.reshape(-1,degen)
            for k in range(degen):
                ilst = idx[:,k]
                occ[ilst] = e
                for i,i0 in enumerate(ilst):
                    cao[i0,ilst] = v[i]
    return occ, cao

def _nao_sub(mol, pre_occ, pre_nao, s=None):
    if s is None:
        if getattr(mol, 'pbc_intor', None):  # whether mol object is a cell
            s = mol.pbc_intor('int1e_ovlp', hermi=1)
        else:
            s = mol.intor_symmetric('int1e_ovlp')

    core_lst, val_lst, rydbg_lst = _core_val_ryd_list(mol)
    nao = mol.nao_nr()
    pre_nao = pre_nao.astype(s.dtype)
    cnao = numpy.empty((nao,nao), dtype=s.dtype)

    if core_lst:
        c = pre_nao[:,core_lst].copy()
        s1 = reduce(lib.dot, (c.conj().T, s, c))
        cnao[:,core_lst] = c1 = lib.dot(c, orth.lowdin(s1))
        c = pre_nao[:,val_lst].copy()
        c -= reduce(lib.dot, (c1, c1.conj().T, s, c))
    else:
        c = pre_nao[:,val_lst]

    if val_lst:
        s1 = reduce(lib.dot, (c.conj().T, s, c))
        wt = pre_occ[val_lst]
        cnao[:,val_lst] = lib.dot(c, orth.weight_orth(s1, wt))

    if rydbg_lst:
        cvlst = core_lst + val_lst
        c1 = cnao[:,cvlst].copy()
        c = pre_nao[:,rydbg_lst].copy()
        c -= reduce(lib.dot, (c1, c1.conj().T, s, c))
        s1 = reduce(lib.dot, (c.conj().T, s, c))
        cnao[:,rydbg_lst] = lib.dot(c, orth.lowdin(s1))
    snorm = numpy.linalg.norm(reduce(lib.dot, (cnao.conj().T, s, cnao)) - numpy.eye(nao))
    if snorm > 1e-9:
        logger.warn(mol, 'Weak orthogonality for localized orbitals %s', snorm)
    return cnao

def _core_val_ryd_list(mol):
    from pyscf.gto.ecp import core_configuration
    count = numpy.zeros((mol.natm, 9), dtype=int)
    core_lst = []
    val_lst = []
    rydbg_lst = []
    k = 0
    for ib in range(mol.nbas):
        ia = mol.bas_atom(ib)
        # Avoid calling mol.atom_charge because we should include ECP core electrons here
        nuc = mole.charge(mol.atom_symbol(ia))
        l = mol.bas_angular(ib)
        nc = mol.bas_nctr(ib)

        nelec_ecp = mol.atom_nelec_core(ia)
        ecpcore = core_configuration(nelec_ecp, atom_symbol=mol.atom_pure_symbol(ia))
        coreshell = [int(x) for x in AOSHELL[nuc][0][::2]]
        cvshell = [int(x) for x in AOSHELL[nuc][1][::2]]
        if mol.cart:
            deg = (l + 1) * (l + 2) // 2
        else:
            deg = 2 * l + 1
        for n in range(nc):
            if l > 3:
                rydbg_lst.extend(range(k, k+deg))
            elif ecpcore[l]+count[ia,l]+n < coreshell[l]:
                core_lst.extend(range(k, k+deg))
            elif ecpcore[l]+count[ia,l]+n < cvshell[l]:
                val_lst.extend(range(k, k+deg))
            else:
                rydbg_lst.extend(range(k, k+deg))
            k = k + deg
        count[ia,l] += nc
    return core_lst, val_lst, rydbg_lst

def _spheric_average_mat(mat, l, lst, degen=None):
    if degen is None:
        degen = l * 2 + 1
    nd = len(lst) // degen
    mat_frag = mat[lst][:,lst].reshape(nd,degen,nd,degen)
    return numpy.einsum('imjn->ij', mat_frag) / degen

def set_atom_conf(element, description):
    '''Change the default atomic core and valence configuration to the one
    given by "description".
    See data/elements.py for the default configuration.

    Args:
        element : str or int
            Element symbol or nuclear charge
        description : str or a list of str
            | "double p" : double p shell
            | "double d" : double d shell
            | "double f" : double f shell
            | "polarize" : add one polarized shell
            | "1s1d"     : keep core unchanged and set 1 s 1 d shells for valence
            | ("3s2p","1d") : 3 s, 2 p shells for core and 1 d shells for valence
    '''
    charge = mole.charge(element)

    def to_conf(desc):
        desc = desc.replace(' ','').replace('-','').replace('_','').lower()
        if "doublep" in desc:
            desc = '2p'
        elif "doubled" in desc:
            desc = '2d'
        elif "doublef" in desc:
            desc = '2f'
        elif "polarize" in desc:
            loc = AOSHELL[charge][1].find('0')
            desc = '1' + AOSHELL[charge][1][loc+1]
        return desc

    if isinstance(description, str):
        c_desc, v_desc = AOSHELL[charge][0], to_conf(description)
    else:
        c_desc, v_desc = to_conf(description[0]), to_conf(description[1])

    ncore = [int(x) for x in AOSHELL[charge][0][::2]]
    ncv = [int(x) for x in AOSHELL[charge][1][::2]]
    for i, s in enumerate(('s', 'p', 'd', 'f')):
        if s in c_desc:
            ncore[i] = int(c_desc.split(s)[0][-1])
        if s in v_desc:
            ncv[i] = ncore[i] + int(v_desc.split(s)[0][-1])
    c_conf  = '%ds%dp%dd%df' % tuple(ncore)
    cv_conf = '%ds%dp%dd%df' % tuple(ncv)
    AOSHELL[charge] = [c_conf, cv_conf]
    sys.stderr.write('Update %s conf: core %s core+valence %s\n' %
                     (element, c_conf, cv_conf))


if __name__ == "__main__":
    from pyscf import gto
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

    s = mol.intor_symmetric('int1e_ovlp_sph')
    p = reduce(numpy.dot, (s, mf.make_rdm1(), s))
    o0, c0 = _prenao_sub(mol, p, s)
    print(o0)
    print(abs(c0).sum() - 21.848915907988854)

    c = nao(mol, mf)
    print(reduce(numpy.dot, (c.T, p, c)).diagonal())
    print(_core_val_ryd_list(mol))

    set_atom_conf('Fe', '1s1d')      # core 3s2p0d0f core+valence 4s2p1d0f
    set_atom_conf('Fe', 'double d')  # core 3s2p0d0f core+valence 4s2p2d0f
    set_atom_conf('Fe', 'double p')  # core 3s2p0d0f core+valence 4s4p2d0f
    set_atom_conf('Fe', 'polarize')  # core 3s2p0d0f core+valence 4s4p2d1f
