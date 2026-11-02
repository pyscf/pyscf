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
#         Jose Luis Casals Sainz <jluiscasalssainz@gmail.com>
#

'''
GAMESS WFN File format
'''

from pyscf import gto
from pyscf import lib
import numpy

# types
# 1 S
# 2 PX
# 3 PY
# 4 PZ
# 5 DXX
# 6 DYY
# 7 DZZ
# 8 DXY
# 9 DXZ
# 10 DYZ
# 11 FXXX
# 12 FYYY
# 13 FZZZ
# 14 FXXY
# 15 FXXZ
# 16 FYYZ
# 17 FXYY
# 18 FXZZ
# 19 FYZZ
# 20 FXYZ
# 21 GXXXX
# 22 GYYYY
# 23 GZZZZ
# 24 GXXXY
# 25 GXXXZ
# 26 GXYYY
# 27 GYYYZ
# 28 GXZZZ
# 29 GYZZZ
# 30 GXXYY
# 31 GXXZZ
# 32 GYYZZ
# 33 GXXYZ
# 34 GXYYZ
# 35 GXYZZ
# 36 HZZZZZ
# 37 HYZZZZ
# 38 HYYZZZ
# 39 HYYYZZ
# 40 HYYYYZ
# 41 HYYYYY
# 42 HXZZZZ
# 43 HXYZZZ
# 44 HXYYZZ
# 45 HXYYYZ
# 46 HXYYYY
# 47 HXXZZZ
# 48 HXXYZZ
# 49 HXXYYZ
# 50 HXXYYY
# 51 HXXXZZ
# 52 HXXXYZ
# 53 HXXXYY
# 54 HXXXXZ
# 55 HXXXXY
# 56 HXXXXX
TYPE_MAP = [
    [1],  # S
    [2, 3, 4],  # P
    [5, 8, 9, 6, 10, 7],  # D
    [11,14,15,17,20,18,12,16,19,13],  # F
    [21,24,25,30,33,31,26,34,35,28,22,27,32,29,23],  # G
    [56,55,54,53,52,51,50,49,48,47,46,45,44,43,42,41,40,39,38,37,36],  # H
]


def write_mo(fout, mol, mo_coeff, mo_energy=None, mo_occ=None):
    if mol.cart:
        raise NotImplementedError('Cartesian basis not available')

    # FIXME: Duplicated primitives may lead to problems.  x2c._uncontract_mol
    # is the workaround at the moment to remove duplicated primitives.
    from pyscf.x2c import x2c
    mol, ctr = x2c._uncontract_mol(mol, True, 0.)
    mo_coeff = numpy.dot(ctr, mo_coeff)

    nmo = mo_coeff.shape[1]
    mo_cart = []
    centers = []
    types = []
    exps = []
    p0 = 0
    for ib in range(mol.nbas):
        ia = mol.bas_atom(ib)
        l = mol.bas_angular(ib)
        es = mol.bas_exp(ib)
        c = mol._libcint_ctr_coeff(ib)
        np, nc = c.shape
        nd = nc*(2*l+1)
        mosub = mo_coeff[p0:p0+nd].reshape(-1,nc,nmo)
        c2s = gto.cart2sph(l)
        mosub = numpy.einsum('yki,cy,pk->pci', mosub, c2s, c)
        mo_cart.append(mosub.transpose(1,0,2).reshape(-1,nmo))

        types.extend([t] * np for t in TYPE_MAP[l])
        ncart = mol.bas_len_cart(ib)
        exps.extend([es]*ncart)
        centers.extend([ia+1]*(np*ncart))
        p0 += nd
    mo_cart = numpy.vstack(mo_cart)
    centers = numpy.hstack(centers)
    types = numpy.hstack(types)
    exps = numpy.hstack(exps)
    nprim, nmo = mo_cart.shape

    fout.write('From PySCF\n')
    fout.write('GAUSSIAN            %3d MOL ORBITALS    %3d PRIMITIVES      %3d NUCLEI\n'
               % (mo_cart.shape[1], mo_cart.shape[0], mol.natm))
    for ia in range(mol.natm):
        x, y, z = mol.atom_coord(ia)
        fout.write('  %-4s %-4d (CENTRE%3d)  %11.8f %11.8f %11.8f  CHARGE = %.1f\n'
                   % (mol.atom_pure_symbol(ia), ia+1, ia+1, x, y, z,
                      mol.atom_charge(ia)))
    for i0, i1 in lib.prange(0, nprim, 20):
        fout.write('CENTRE ASSIGNMENTS  %s\n' % ''.join('%3d' % x for x in centers[i0:i1]))
    for i0, i1 in lib.prange(0, nprim, 20):
        fout.write('TYPE ASSIGNMENTS    %s\n' % ''.join('%3d' % x for x in types[i0:i1]))
    for i0, i1 in lib.prange(0, nprim, 5):
        fout.write('EXPONENTS  %s\n' % ' '.join('%13.7E' % x for x in exps[i0:i1]))

    for k in range(nmo):
        mo = mo_cart[:,k]
        if mo_energy is None and mo_occ is None:
            fout.write('CANMO %d\n' % (k+1))
        elif mo_energy is None:
            fout.write('MO  %-4d                  OCC NO = %12.8f ORB. ENERGY = 0.0000\n' %
                       (k+1, mo_occ[k]))
        else:
            fout.write('MO  %-4d                  OCC NO = %12.8f ORB. ENERGY = %12.8f\n' %
                       (k+1, mo_occ[k], mo_energy[k]))
        for i0, i1 in lib.prange(0, nprim, 5):
            fout.write(' %s\n' % ' '.join('%15.8E' % x for x in mo[i0:i1]))
    fout.write('END DATA\n')
    if mo_energy is None or mo_occ is None:
        fout.write('ALDET    ENERGY =        0.0000000000   VIRIAL(-V/T)  =   0.00000000\n')
    elif mo_energy is None and mo_occ is None:
        pass
    else:
        fout.write('RHF      ENERGY =        0.0000000000   VIRIAL(-V/T)  =   0.00000000\n')


def write_ci(fout, fcivec, norb, nelec, ncore=0):
    from pyscf import fci
    if isinstance(nelec, (int, numpy.number)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    fout.write('NELACTIVE, NDETS, NORBCORE, NORBACTIVE\n')
    fout.write(' %5d %5d %5d %5d\n' % (neleca+nelecb, fcivec.size, ncore, norb))
    fout.write('COEFFICIENT/ OCCUPIED ACTIVE SPIN ORBITALS\n')

    nb = fci.cistring.num_strings(norb, nelecb)
    stringsa = fci.cistring.gen_strings4orblist(range(norb), neleca)
    stringsb = fci.cistring.gen_strings4orblist(range(norb), nelecb)

    def str2orbidx(string, ncore):
        bstring = bin(string)
        return [i+1+ncore for i,s in enumerate(bstring[::-1]) if s == '1']

    addrs = numpy.argsort(abs(fcivec.ravel()))
    for iaddr in reversed(addrs):
        addra, addrb = divmod(iaddr, nb)
        idxa = ['%3d' % x for x in str2orbidx(stringsa[addra], ncore)]
        idxb = ['%3d' % (-x) for x in str2orbidx(stringsb[addrb], ncore)]
        # TODO:add a cuttoff and a counter for ndets
        fout.write('%18.10E %s %s\n' % (fcivec[addra,addrb], ' '.join(idxa), ' '.join(idxb)))
