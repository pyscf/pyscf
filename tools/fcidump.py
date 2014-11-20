#!/usr/bin/env python

import os, sys
import tempfile
import numpy


def write_head(fout, nmo, nelec, ms=0, orbsym=[]):
    fout.write(' &FCI NORB=%4d,NELEC=%2d,MS2=%d,\n' % (nmo, nelec, ms))
    if orbsym:
        fout.write('  ORBSYM=%s\n' % ','.join(map(str, orbsym)))
    else:
        fout.write('  ORBSYM=%s\n' % ('1,' * nmo))
    fout.write('  ISYM=1,\n')
    fout.write(' &END\n')

# 4-fold symmetry
def write_eri(fout, eri, nmo):
    npair = nmo*(nmo+1)/2
    if eri.size == npair**2: # 4-fold symmetry
        ij = 0
        for i in range(nmo):
            for j in range(0, i+1):
                kl = 0
                for k in range(0, nmo):
                    for l in range(0, k+1):
                        fout.write(' %.16g %4d %4d %4d %4d\n' \
                                   % (eri[ij,kl], i+1, j+1, k+1, l+1))
                        kl += 1
                ij += 1
    else:
        ij = 0
        ijkl = 0
        for i in range(nmo):
            for j in range(0, i+1):
                kl = 0
                for k in range(0, i+1):
                    for l in range(0, k+1):
                        if ij >= kl:
                            fout.write(' %.16g %4d %4d %4d %4d\n' \
                                       % (eri[ijkl], i+1, j+1, k+1, l+1))
                            ijkl += 1
                        kl += 1
                ij += 1

def write_hcore(fout, h, nmo):
    h = h.reshape(nmo,nmo)
    for i in range(nmo):
        for j in range(0, i+1):
            fout.write(' %.16g %4d %4d  0  0\n' % (h[i,j], i+1, j+1))


def from_chkfile(output, chkfile):
    import pyscf.scf
    import pyscf.ao2mo
    import pyscf.symm
    with open(output, 'w') as fout:
        mol, scf_rec = pyscf.scf.chkfile.load_scf(chkfile)
        mo_coeff = numpy.array(scf_rec['mo_coeff'])
        nmo = mo_coeff.shape[1]
        if mol.symmetry:
            orbsym = pyscf.symm.label_orb_symm(mol, mol.irrep_name,
                                               mol.irrep_id, mo_coeff)
            write_head(fout, nmo, mol.nelectron, mol.spin, orbsym)
        else:
            write_head(fout, nmo, mol.nelectron, mol.spin)

        eri = pyscf.ao2mo.direct.full_iofree(mol, mo_coeff, verbose=0)
        write_eri(fout, pyscf.ao2mo.restore(8, eri, nmo), nmo)

        t = mol.intor_symmetric('cint1e_kin_sph')
        v = mol.intor_symmetric('cint1e_nuc_sph')
        h = reduce(numpy.dot, (mo_coeff.T, t+v, mo_coeff))
        write_hcore(fout, h, nmo)
        fout.write(' %.16g  0  0  0  0\n' % mol.nuclear_repulsion())

def from_integrals(output, h1e, h2e, nmo, nelec, nuc=0, ms=0, orbsym=[]):
    with open(output, 'w') as fout:
        write_head(fout, nmo, nelec, ms, orbsym)
        write_eri(fout, h2e, nmo)
        write_hcore(fout, h1e, nmo)
        fout.write(' %.16g  0  0  0  0\n' % nuc)


if __name__ == '__main__':
    # molpro_fcidump.py chkfile output
    fcidump(sys.argv[2], sys.argv[1])
