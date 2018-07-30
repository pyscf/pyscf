#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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

from functools import reduce
import numpy
from pyscf import __config__

DEFAULT_FLOAT_FORMAT = getattr(__config__, 'fcidump_float_format', ' %.16g')
TOL = getattr(__config__, 'fcidump_write_tol', 1e-15)

def write_head(fout, nmo, nelec, ms=0, orbsym=None):
    if not isinstance(nelec, (int, numpy.number)):
        ms = abs(nelec[0] - nelec[1])
        nelec = nelec[0] + nelec[1]
    fout.write(' &FCI NORB=%4d,NELEC=%2d,MS2=%d,\n' % (nmo, nelec, ms))
    if orbsym is not None and len(orbsym) > 0:
        fout.write('  ORBSYM=%s\n' % ','.join([str(x) for x in orbsym]))
    else:
        fout.write('  ORBSYM=%s\n' % ('1,' * nmo))
    fout.write('  ISYM=1,\n')
    fout.write(' &END\n')


def write_eri(fout, eri, nmo, tol=TOL, float_format=DEFAULT_FLOAT_FORMAT):
    from pyscf import ao2mo
    npair = nmo*(nmo+1)//2
    output_format = float_format + ' %4d %4d %4d %4d\n'
    if eri.size == nmo**4:
        eri = ao2mo.restore(8, eri, nmo)

    if eri.ndim == 2: # 4-fold symmetry
        assert(eri.size == npair**2)
        ij = 0
        for i in range(nmo):
            for j in range(0, i+1):
                kl = 0
                for k in range(0, nmo):
                    for l in range(0, k+1):
                        if abs(eri[ij,kl]) > tol:
                            fout.write(output_format % (eri[ij,kl], i+1, j+1, k+1, l+1))
                        kl += 1
                ij += 1
    else:  # 8-fold symmetry
        assert(eri.size == npair*(npair+1)//2)
        ij = 0
        ijkl = 0
        for i in range(nmo):
            for j in range(0, i+1):
                kl = 0
                for k in range(0, i+1):
                    for l in range(0, k+1):
                        if ij >= kl:
                            if abs(eri[ijkl]) > tol:
                                fout.write(output_format % (eri[ijkl], i+1, j+1, k+1, l+1))
                            ijkl += 1
                        kl += 1
                ij += 1

def write_hcore(fout, h, nmo, tol=TOL, float_format=DEFAULT_FLOAT_FORMAT):
    h = h.reshape(nmo,nmo)
    output_format = float_format + ' %4d %4d  0  0\n'
    for i in range(nmo):
        for j in range(0, i+1):
            if abs(h[i,j]) > tol:
                fout.write(output_format % (h[i,j], i+1, j+1))


def from_chkfile(output, chkfile, tol=TOL, float_format=DEFAULT_FLOAT_FORMAT):
    '''Read SCF results from PySCF chkfile and transform 1-electron,
    2-electron integrals using the SCF orbitals.  The transformed integrals is
    written to FCIDUMP'''
    from pyscf import scf, ao2mo, symm
    with open(output, 'w') as fout:
        mol, scf_rec = scf.chkfile.load_scf(chkfile)
        mo_coeff = numpy.array(scf_rec['mo_coeff'])
        nmo = mo_coeff.shape[1]
        if mol.symmetry:
            orbsym = symm.label_orb_symm(mol, mol.irrep_id,
                                         mol.symm_orb, mo_coeff, check=False)
            write_head(fout, nmo, mol.nelectron, mol.spin, orbsym)
        else:
            write_head(fout, nmo, mol.nelectron, mol.spin)

        eri = ao2mo.full(mol, mo_coeff, verbose=0)
        write_eri(fout, ao2mo.restore(8, eri, nmo), nmo, tol, float_format)

        t = mol.intor_symmetric('int1e_kin')
        v = mol.intor_symmetric('int1e_nuc')
        h = reduce(numpy.dot, (mo_coeff.T, t+v, mo_coeff))
        write_hcore(fout, h, nmo, tol, float_format)
        output_format = ' ' + float_format + '  0  0  0  0\n'
        fout.write(output_format % mol.energy_nuc())

def from_integrals(output, h1e, h2e, nmo, nelec, nuc=0, ms=0, orbsym=[],
                   tol=TOL, float_format=DEFAULT_FLOAT_FORMAT):
    '''Convert the given 1-electron and 2-electron integrals to FCIDUMP format'''
    with open(output, 'w') as fout:
        write_head(fout, nmo, nelec, ms, orbsym)
        write_eri(fout, h2e, nmo, tol=tol, float_format=float_format)
        write_hcore(fout, h1e, nmo, tol=tol, float_format=float_format)
        output_format = float_format + '  0  0  0  0\n'
        fout.write(output_format % nuc)

def read(filename):
    '''Parse FCIDUMP.  Return a dictionary to hold the integrals and
    parameters with keys:  H1, H2, ECORE, NORB, NELEC, MS, ORBSYM, ISYM
    '''
    import re
    dic = {}
    print('Parsing %s' % filename)
    finp = open(filename, 'r')
    dat = re.split('[=,]', finp.readline())
    while not 'FCI' in dat[0].upper():
        dat = re.split('[=,]', finp.readline())
    dic['NORB'] = int(dat[1])
    dic['NELEC'] = int(dat[3])
    dic['MS2'] = int(dat[5])
    norb = dic['NORB']

    sym = []
    dat = finp.readline().strip()
    while not 'END' in dat:
        sym.append(dat)
        dat = finp.readline().strip()

    isym = [x.split('=')[1] for x in sym if 'ISYM' in x]
    if len(isym) > 0:
        dic['ISYM'] = int(isym[0].replace(',','').strip())
    symorb = ','.join([x for x in sym if 'ISYM' not in x]).split('=')[1]
    dic['ORBSYM'] = [int(x.strip()) for x in symorb.replace(',', ' ').split()]

    norb_pair = norb * (norb+1) // 2
    h1e = numpy.zeros((norb,norb))
    h2e = numpy.zeros(norb_pair*(norb_pair+1)//2)
    dat = finp.readline().split()
    while dat:
        i, j, k, l = [int(x) for x in dat[1:5]]
        if k != 0:
            if i >= j:
                ij = i * (i-1) // 2 + j-1
            else:
                ij = j * (j-1) // 2 + i-1
            if k >= l:
                kl = k * (k-1) // 2 + l-1
            else:
                kl = l * (l-1) // 2 + k-1
            if ij >= kl:
                h2e[ij*(ij+1)//2+kl] = float(dat[0])
            else:
                h2e[kl*(kl+1)//2+ij] = float(dat[0])
        elif k == 0:
            if j != 0:
                h1e[i-1,j-1] = float(dat[0])
            else:
                dic['ECORE'] = float(dat[0])
        dat = finp.readline().split()

    idx, idy = numpy.tril_indices(norb, -1)
    if numpy.linalg.norm(h1e[idy,idx]) == 0:
        h1e[idy,idx] = h1e[idx,idy]
    elif numpy.linalg.norm(h1e[idx,idy]) == 0:
        h1e[idx,idy] = h1e[idy,idx]
    dic['H1'] = h1e
    dic['H2'] = h2e
    finp.close()
    return dic

if __name__ == '__main__':
    import sys
    # fcidump.py chkfile output
    from_chkfile(sys.argv[2], sys.argv[1])
