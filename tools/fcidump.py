#!/usr/bin/env python

from functools import reduce
import numpy

DEFAULT_FLOAT_FORMAT = ' %.16g'

def write_head(fout, nmo, nelec, ms=0, orbsym=[]):
    if not isinstance(nelec, (int, numpy.number)):
        ms = abs(nelec[0] - nelec[1])
        nelec = nelec[0] + nelec[1]
    fout.write(' &FCI NORB=%4d,NELEC=%2d,MS2=%d,\n' % (nmo, nelec, ms))
    if orbsym:
        fout.write('  ORBSYM=%s\n' % ','.join([str(x) for x in orbsym]))
    else:
        fout.write('  ORBSYM=%s\n' % ('1,' * nmo))
    fout.write('  ISYM=1,\n')
    fout.write(' &END\n')


def write_eri(fout, eri, nmo, tol=1e-15, float_format=DEFAULT_FLOAT_FORMAT):
    npair = nmo*(nmo+1)//2
    output_format = float_format + ' %4d %4d %4d %4d\n'
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

def write_hcore(fout, h, nmo, tol=1e-15, float_format=DEFAULT_FLOAT_FORMAT):
    h = h.reshape(nmo,nmo)
    output_format = float_format + ' %4d %4d  0  0\n'
    for i in range(nmo):
        for j in range(0, i+1):
            if abs(h[i,j]) > tol:
                fout.write(output_format % (h[i,j], i+1, j+1))


def from_chkfile(output, chkfile, tol=1e-15, float_format=DEFAULT_FLOAT_FORMAT):
    import pyscf.scf
    import pyscf.ao2mo
    import pyscf.symm
    with open(output, 'w') as fout:
        mol, scf_rec = pyscf.scf.chkfile.load_scf(chkfile)
        mo_coeff = numpy.array(scf_rec['mo_coeff'])
        nmo = mo_coeff.shape[1]
        if mol.symmetry:
            orbsym = pyscf.symm.label_orb_symm(mol, mol.irrep_id,
                                               mol.symm_orb, mo_coeff)
            write_head(fout, nmo, mol.nelectron, mol.spin, orbsym)
        else:
            write_head(fout, nmo, mol.nelectron, mol.spin)

        eri = pyscf.ao2mo.outcore.full_iofree(mol, mo_coeff, verbose=0)
        write_eri(fout, pyscf.ao2mo.restore(8, eri, nmo), nmo, tol, float_format)

        t = mol.intor_symmetric('cint1e_kin_sph')
        v = mol.intor_symmetric('cint1e_nuc_sph')
        h = reduce(numpy.dot, (mo_coeff.T, t+v, mo_coeff))
        write_hcore(fout, h, nmo, tol, float_format)
        output_format = ' ' + float_format + '  0  0  0  0\n'
        fout.write(output_format % mol.energy_nuc())

def from_integrals(output, h1e, h2e, nmo, nelec, nuc=0, ms=0, orbsym=[],
                   tol=1e-15, float_format=DEFAULT_FLOAT_FORMAT):
    with open(output, 'w') as fout:
        write_head(fout, nmo, nelec, ms, orbsym)
        write_eri(fout, h2e, nmo, tol=tol, float_format=float_format)
        write_hcore(fout, h1e, nmo, tol=tol, float_format=float_format)
        output_format = float_format + '  0  0  0  0\n'
        fout.write(output_format % nuc)

if __name__ == '__main__':
    import sys
    # fcidump.py chkfile output
    from_chkfile(sys.argv[2], sys.argv[1])
