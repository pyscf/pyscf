#!/usr/bin/env python
# Copyright 2014-2023 The PySCF Developers. All Rights Reserved.
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
# Author: Timothy Berkelbach <tim.berkelbach@gmail.com>
#

'''
parse CP2K PP format, following parse_nwchem.py
'''

import sys
import re
from pyscf.lib.exceptions import BasisNotFoundError
import numpy as np

def parse(string, symb=None):
    '''Parse the pseudo text *string* which is in CP2K format, return an internal
    basis format which can be assigned to :attr:`Cell.pseudo`
    Lines started with # are ignored.

    Args:
        string : Blank linke and the lines of "PSEUDOPOTENTIAL" and "END" will be ignored

    Examples:

    >>> cell = gto.Cell()
    >>> cell.pseudo = {'C': pyscf.gto.basis.pseudo_cp2k.parse("""
    ... #PSEUDOPOTENTIAL
    ... C GTH-BLYP-q4
    ...     2    2
    ...      0.33806609    2    -9.13626871     1.42925956
    ...     2
    ...      0.30232223    1     9.66551228
    ...      0.28637912    0
    ... """)}
    '''
    blocks = _split_blocks(string)
    if symb is not None:
        raw_data = _search_gthpp_block(blocks, symb)
        if not raw_data:
            raise BasisNotFoundError(f'Pseudopotential not found for {symb}.')
    elif blocks:
        raw_data = blocks[0]
    else:
        raise BasisNotFoundError('Not pseudo potential data')
    return _parse(raw_data)

def load(pseudofile, symb, suffix=None):
    '''Parse the *pseudofile's entry* for atom 'symb', return an internal
    pseudo format which can be assigned to :attr:`Cell.pseudo`
    '''
    return _parse(search_seg(pseudofile, symb, suffix))

def _load_GTH_POTENTIALS(pp_name, symb, pp_dir, with_soc=False):
    if with_soc:
        pp_files = ('GTH_SOC_POTENTIALS',)
    else:
        pp_files = ('GTH_POTENTIALS', 'POTENTIAL_UZH')
    for pp_file in pp_files:
        with open(f'{pp_dir}/{pp_file}', 'r') as searchfile:
            blocks = _split_blocks(searchfile.read())
        for block in blocks:
            header = block[0].split()
            if header[0] == symb and pp_name in header[1:]:
                return _parse(block)
    raise BasisNotFoundError(
        f'{pp_name} for {symb} not found in files {",".join(pp_files)}.')


def _split_blocks(string):
    # CP2K native databases separate entries with element/name headers.
    # older PySCF database files contain #PSEUDOPOTENTIAL delimiters.
    blocks = []
    for line in string.splitlines():
        line = line.split('#', 1)[0].strip()
        if not line or line in ('END', 'PSEUDOPOTENTIAL'):
            continue
        if re.match(r'^[A-Za-z][A-Za-z]?(?=\s|$)', line): # match element
            blocks.append([])
        if blocks:
            blocks[-1].append(line)
    return blocks

def _unpack_triu(dat):
    '''
    i, j = np.triu_indices(n)
    a[i,j] = a[j,i] = dat
    return a.tolist()
    '''
    if len(dat) == 0:
        return []
    if len(dat) == 1:
        result = [dat]
    elif len(dat) == 3:
        result = [[dat[0], dat[1]], [dat[1], dat[2]]]
    elif len(dat) == 6:
        result = [[dat[0], dat[1], dat[2]],
                  [dat[1], dat[3], dat[4]],
                  [dat[2], dat[4], dat[5]]]
    else:
        raise ValueError(f'Incorrect number of GTH projector coefficients {len(dat)}')
    return result

def _parse(plines):
    line_iter = iter(plines)
    try:
        header_ln = next(line_iter)  # noqa: F841
        nelecs = [ int(nelec) for nelec in next(line_iter).split() ]
    except ValueError:
        raise BasisNotFoundError('Not pseudo potential data')

    rnc_ppl = next(line_iter).split()
    rloc = float(rnc_ppl[0])
    nexp = int(rnc_ppl[1])
    cexp = [ float(c) for c in rnc_ppl[2:] ]
    if len(cexp) != nexp:
        raise ValueError('Invalid GTH local potential')

    proj_types = next(line_iter).split()
    nproj_types = int(proj_types[0])
    has_soc = len(proj_types) == 2 and proj_types[1] == 'SOC'
    r = []
    nproj = []
    hproj = []
    kproj = []
    for p in range(nproj_types):
        rnh_ppnl = next(line_iter).split()
        rl = float(rnh_ppnl[0])
        r.append(rl)
        nl = int(rnh_ppnl[1])
        nproj.append(nl)
        hproj_p_ij = []
        for h in rnh_ppnl[2:]:
            hproj_p_ij.append(float(h))
        for i in range(1,nl):
            for h in next(line_iter).split():
                hproj_p_ij.append(float(h))
        hproj.append(_unpack_triu(hproj_p_ij))

        if has_soc:
            if p == 0:
                # kproj are only defined for r(2) and higher
                kproj.append([])
                continue
            kproj_p_ij = []
            for i in range(nl):
                for k in next(line_iter).split():
                    kproj_p_ij.append(float(k))
            kproj.append(_unpack_triu(kproj_p_ij))

    if has_soc:
        pseudo_params = [nelecs,
                         rloc, nexp, cexp,
                         (nproj_types, 'SOC')]
        pseudo_params.extend(zip(r, nproj, hproj, kproj))
    else:
        pseudo_params = [nelecs,
                         rloc, nexp, cexp,
                         nproj_types]
        pseudo_params.extend(zip(r, nproj, hproj))
    return pseudo_params

def search_seg(pseudofile, symb, suffix=None):
    '''
    Find the pseudopotential entry for atom 'symb' in file 'pseudofile'
    '''
    with open(pseudofile, 'r') as f:
        fdata = _split_blocks(f.read())
    dat = _search_gthpp_block(fdata, symb, suffix)
    if not dat:
        raise BasisNotFoundError(f'Pseudopotential for {symb} in {pseudofile}')
    return dat

def _search_gthpp_block(raw_data, symb, suffix=None):
    for dat in raw_data:
        if dat and dat[0].split()[0] == symb:
            if suffix is None:  # use default PP
                qsuffix = dat[0].split('-')[-1]
                if not (qsuffix.startswith('q') and qsuffix[1:].isdigit()):
                    return dat
            else:
                if any(suffix == x.split('-')[-1] for x in dat[0].split()):
                    return dat
    return None
