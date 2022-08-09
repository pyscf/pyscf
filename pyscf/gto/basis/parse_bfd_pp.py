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
Parsers for basis set associated with Burkatzi-Filippi-Dolg  pseudo potential
'''

import os
from pyscf.lib.exceptions import BasisNotFoundError

MAPSPDF = {'S': 0,
           'P': 1,
           'D': 2,
           'F': 3,
           'G': 4,
           'H': 5,
           'I': 6,
           'K': 7}

# basistype is one of 'vdz', 'vtz', 'vqz', 'v5z'
def parse(symb, basistype):
    basisfile = os.path.join(os.path.dirname(__file__), 'Burkatzi-Filippi-Dolg-PP.dat')
    raw = ''.join(search_seg(basisfile, symb))
    bas_dat = raw.split('Fit-data:')[1]
    for bas_seg in bas_dat.split('Basis-name: ')[1:]:
        if bas_seg.startswith(basistype.lower()):
            break

    bas_seg = bas_seg.split('\n')[3:]
    bas_block = []
    for bas in bas_seg:
        bas = bas.strip(' ')
        if bas:
            if bas.isdigit():
                bas_block.append([])
            else:
                bas_block[-1].append(bas)

    basis = []
    for dat in bas_block:
        l = MAPSPDF[dat[0][0].upper()]
        bas_by_l = [l]
        basis.append(bas_by_l)
        for bas in reversed(dat):
            expnt, coeff = [float(x) for x in bas.split()[1:]]
            bas_by_l.append([expnt, coeff])
    return basis

def parse_ecp(symb):
    basisfile = os.path.join(os.path.dirname(__file__), 'Burkatzi-Filippi-Dolg-PP.dat')
    raw = ''.join(search_seg(basisfile, symb))
    pp_dat = raw.split('Fit-data:')[0]
    info, loc_dat, nloc_dat = pp_dat.split('component')

    for dat in info.split('\n'):
        if 'Number of replaced protons' in dat:
            ncore = int(dat.split(' ')[-1])
        elif 'Number of projectors' in dat:
            nloc_max = int(dat.split(' ')[-1])

    loc = [[]]  # r^0 is empty
    loc_dat = loc_dat.replace('\t', ' ').split('\n')[2:5]
    zeff, rorder, expnta = [float(x) for x in loc_dat[0].split(' ')]
    assert (rorder == -1)
    loc.append([[expnta, zeff]])
    gamma, rorder, expnt = [float(x) for x in loc_dat[2].split(' ')]
    assert (rorder == 0)
    loc.append([[expnt, gamma]])
    alpha, rorder, expnt = [float(x) for x in loc_dat[1].split(' ')]
    assert (rorder == 1)
    assert (abs(zeff*expnta-alpha) < 1e-7)
    loc.append([[expnt, alpha]])

    nloc = []
    for dat in nloc_dat.replace('\t', ' ').split('\n')[2:]:
        if dat:  # remove blank lines
            coeff, rorder, expnt = [float(x) for x in dat.split(' ')[:3]]
            assert (rorder == 0)
            nloc.append([[], [],  # r^0, r^1 are empty
                         [[expnt, coeff]]])
    assert (len(nloc) == nloc_max)

    ecp = [[-1, loc]]
    for l, dat in enumerate(nloc):
        ecp.append([l, dat])

    return [ncore, ecp]

# basisfile = Burkatzi-Filippi-Dolg-PP.dat
def search_seg(basisfile, symb):
    with open(basisfile, 'r') as fin:
        # ignore head
        dat = fin.readline().lstrip(' ')
        while dat.startswith('#'):
            dat = fin.readline().lstrip(' ')

        # searching
        while dat:
            if dat[:-1].rstrip(' ') == symb:
                seg = []
                fin.readline()  # pass the "back to top" mark
                while (dat and not dat.startswith('back to top')):
                    seg.append(dat)
                    dat = fin.readline().lstrip(' ')
                return seg[:-1]
            dat = fin.readline().lstrip(' ')
    raise BasisNotFoundError('Basis not found for  %s  in  %s' % (symb, basisfile))


if __name__ == '__main__':
    #print(parse('Li', 'vtz'))
    #print(parse_ecp('Ga'))
    from pyscf.data import elements
    from pyscf.gto.basis import parse_nwchem

#    for bastype in 'vdz', 'vtz', 'vqz', 'v5z':
#        dat = []
#        for atom in elements.ELEMENTS[1:]:
#            try:
#                bas = parse(atom, bastype)
#                dat.append(parse_nwchem.convert_basis_to_nwchem(atom[0], bas))
#            except RuntimeError:
#                pass
#        with open('bfd_%s.dat'%bastype, 'w') as f:
#            f.write('#ftp://ftp.aip.org/epaps/journ_chem_phys/E-JCPSA6-126-315722/epaps_material.html\n')
#            f.write('# M. Burkatzki, C. Filippi, M. Dolg in J. Chem. Phys. 126, 234105 (2007)\n\n')
#            f.write('BASIS "ao basis" PRINT\n')
#            f.write('\n'.join(dat))
#            f.write('END\n')

    dat = []
    for atom in elements.ELEMENTS[1:]:
        try:
            ecp = parse_ecp(atom)
            dat.append(parse_nwchem.convert_ecp_to_nwchem(atom[0], ecp))
        except RuntimeError:
            pass
    with open('bfd_pp.dat', 'w') as f:
        f.write('#ftp://ftp.aip.org/epaps/journ_chem_phys/E-JCPSA6-126-315722/epaps_material.html\n')
        f.write('# M. Burkatzki, C. Filippi, M. Dolg in J. Chem. Phys. 126, 234105 (2007)\n\n')
        f.write('ECP\n')
        f.write('\n'.join(dat))
        f.write('END\n')
