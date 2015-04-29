#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

# MOLDEN format:
# http://www.cmbi.ru.nl/molden/molden_format.html

import numpy
import pyscf.lib.parameters as param


def header(mol, fout):
    fout.write('''[Molden Format]
[Atoms] (AU)\n''')
    for ia in range(mol.natm):
        symb = mol.atom_pure_symbol(ia)
        chg = mol.atom_charge(ia)
        fout.write('%s   %d   %d   ' % (symb, ia+1, chg))
        coord = mol.atom_coord(ia)
        fout.write('%18.14f   %18.14f   %18.14f\n' % tuple(coord))
    fout.write('[GTO]\n')
    for ia in range(mol.natm):
        fout.write('%d 0\n' %(ia+1))
        for b in mol._basis[mol.atom_symbol(ia)]:
            l = b[0]
            if isinstance(b[1], int):
                b_coeff = b[2:]
            else:
                b_coeff = b[1:]
            nprim = len(b_coeff)
            nctr = len(b_coeff[0]) - 1
            for ic in range(nctr):
                fout.write(' %s   %2d 1.00\n' % (param.ANGULAR[l], nprim))
                for ip in range(nprim):
                    fout.write('    %18.14g  %18.14g\n' %
                               (b_coeff[ip][0], b_coeff[ip][ic+1]))
        fout.write('\n')
    fout.write('[5d]\n[9g]\n\n')

def order_ao_index(mol):
# reorder d,f,g fucntion to
#  5D: D 0, D+1, D-1, D+2, D-2
#  6D: xx, yy, zz, xy, xz, yz
#
#  7F: F 0, F+1, F-1, F+2, F-2, F+3, F-3
# 10F: xxx, yyy, zzz, xyy, xxy, xxz, xzz, yzz, yyz, xyz
#
#  9G: G 0, G+1, G-1, G+2, G-2, G+3, G-3, G+4, G-4
# 15G: xxxx yyyy zzzz xxxy xxxz yyyx yyyz zzzx zzzy xxyy xxzz yyzz xxyz yyxz zzxy
    idx = []
    off = 0
    for ib in range(mol.nbas):
        l = mol.bas_angular(ib)
        for n in range(mol.bas_nctr(ib)):
            if l == 2:
                idx.extend([off+2,off+3,off+1,off+4,off+0])
            elif l == 3:
                idx.extend([off+3,off+4,off+2,off+5,off+1,off+6,off+0])
            elif l == 4:
                idx.extend([off+4,off+5,off+3,off+6,off+2,off+7,off+1,off+8,off+0])
            elif l > 4:
                raise RuntimeError('l=5 is not supported')
            else:
                idx.extend(range(off,off+l*2+1))
            off += l * 2 + 1
    return idx

def orbital_coeff(mol, fout, mo_coeff, spin='Alpha', symm=None, ene=None, \
                  occ=None):
    import pyscf.symm
    aoidx = order_ao_index(mol)
    nmo = mo_coeff.shape[1]
    if symm is None:
        if mol.symmetry:
            symm = pyscf.symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mo_coeff)
        else:
            symm = ['A']*nmo
    if ene is None:
        ene = numpy.arange(nmo)
    if occ is None:
        occ = numpy.zeros(nmo)
    assert(spin == 'Alpha' or spin == 'Beta')
    fout.write('[MO]\n')
    for imo in range(nmo):
        fout.write(' Sym= %s\n' % symm[imo])
        fout.write(' Ene= %15.10g\n' % ene[imo])
        fout.write(' Spin= %s\n' % spin)
        fout.write(' Occup= %10.5f\n' % occ[imo])
        for i,j in enumerate(aoidx):
            fout.write(' %3d    %18.14g\n' % (i+1, mo_coeff[j,imo]))

def from_scf(mf, filename):
    dump_scf(mf, filename)
def dump_scf(mf, filename):
    import pyscf.scf
    with open(filename, 'w') as f:
        header(mf.mol, f)
        if isinstance(mf, pyscf.scf.uhf.UHF) or 'UHF' == mf.__class__.__name__:
            orbital_coeff(mf.mol, f, mf.mo_coeff[0], spin='Alpha', \
                          ene=mf.mo_energy[0], occ=mf.mo_occ[0])
            orbital_coeff(mf.mol, f, mf.mo_coeff[1], spin='Beta', \
                          ene=mf.mo_energy[1], occ=mf.mo_occ[1])
        else:
            orbital_coeff(mf.mol, f, mf.mo_coeff, \
                          ene=mf.mo_energy, occ=mf.mo_occ)

def from_chkfile(outfile, chkfile, key='scf/mo_coeff'):
    import pyscf.scf
    with open(outfile, 'w') as f:
        mol, mf = pyscf.scf.chkfile.load_scf(chkfile)
        header(mol, f)
        ene = mf['mo_energy']
        occ = mf['mo_occ']
        if key == 'scf/mo_coeff':
            mo = mf['mo_coeff']
        else:
            mo = mf[key]
        if occ.ndim == 2:
            orbital_coeff(mol, f, mo[0], spin='Alpha', ene=ene[0], occ=occ[0])
            orbital_coeff(mol, f, mo[1], spin='Beta', ene=ene[1], occ=occ[1])
        else:
            orbital_coeff(mol, f, mo, ene=ene, occ=occ)


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    mol = gto.Mole()
    mol.verbose = 5
    mol.output = None#'out_gho'
    mol.atom = [['C', (0.,0.,0.)],
                ['H', ( 1, 1, 1)],
                ['H', (-1,-1, 1)],
                ['H', ( 1,-1,-1)],
                ['H', (-1, 1,-1)], ]
    mol.basis = {
        'C': 'sto-3g',
        'H': 'sto-3g'}
    mol.build(dump_input=False)
    m = scf.RHF(mol)
    m.scf()
    header(mol, mol.stdout)
    print(order_ao_index(mol))
    orbital_coeff(mol, mol.stdout, m.mo_coeff)
