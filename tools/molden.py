#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

# MOLDEN format:
# http://www.cmbi.ru.nl/molden/molden_format.html

import sys
import numpy
import pyscf.lib.parameters as param
from pyscf import gto


def orbital_coeff(mol, fout, mo_coeff, spin='Alpha', symm=None, ene=None,
                  occ=None, ignore_h=False):
    import pyscf.symm
    if ignore_h:
        mol, mo_coeff = remove_high_l(mol, mo_coeff)
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

def from_mo(mol, outfile, mo_coeff, spin='Alpha', symm=None, ene=None,
            occ=None, ignore_h=False):
    with open(outfile, 'w') as f:
        header(mol, f, ignore_h)
        orbital_coeff(mol, f, mo_coeff, spin, symm, ene, occ, ignore_h)


def from_scf(mf, filename, ignore_h=False):
    dump_scf(mf, filename)
def dump_scf(mf, filename, ignore_h=False):
    import pyscf.scf
    mol = mf.mol
    mo_coeff = mf.mo_coeff
    with open(filename, 'w') as f:
        header(mol, f, ignore_h)
        if isinstance(mf, pyscf.scf.uhf.UHF) or 'UHF' == mf.__class__.__name__:
            orbital_coeff(mol, f, mo_coeff[0], spin='Alpha',
                          ene=mf.mo_energy[0], occ=mf.mo_occ[0],
                          ignore_h=ignore_h)
            orbital_coeff(mol, f, mo_coeff[1], spin='Beta',
                          ene=mf.mo_energy[1], occ=mf.mo_occ[1],
                          ignore_h=ignore_h)
        else:
            orbital_coeff(mf.mol, f, mf.mo_coeff,
                          ene=mf.mo_energy, occ=mf.mo_occ, ignore_h=ignore_h)

def from_chkfile(outfile, chkfile, key='scf/mo_coeff', ignore_h=False):
    import pyscf.scf
    with open(outfile, 'w') as f:
        if key == 'scf/mo_coeff':
            mol, mf = pyscf.scf.chkfile.load_scf(chkfile)
            header(mol, f, ignore_h)
            ene = mf['mo_energy']
            occ = mf['mo_occ']
            mo = mf['mo_coeff']
        else:
            mol = pyscf.scf.chkfile.load_mol(chkfile)
            header(mol, f, ignore_h)
            dat = pyscf.scf.chkfile.load(chkfile, key.split('/')[0])
            if 'mo_energy' in dat:
                ene = dat['mo_energy']
            else:
                ene = None
            occ = dat['mo_occ']
            mo = dat['mo_coeff']
        if occ.ndim == 2:
            orbital_coeff(mol, f, mo[0], spin='Alpha', ene=ene[0], occ=occ[0],
                         ignore_h=ignore_h)
            orbital_coeff(mol, f, mo[1], spin='Beta', ene=ene[1], occ=occ[1],
                         ignore_h=ignore_h)
        else:
            orbital_coeff(mol, f, mo, ene=ene, occ=occ, ignore_h=ignore_h)

def parse(moldenfile):
    return load(moldenfile)
def load(moldenfile):
    '''Extract mol and orbitals from molden file
    '''
    mol = gto.Mole()
    with open(moldenfile, 'r') as f:
        line = first_token(f, '[Atoms]')
        if 'ANG' in line.upper():
            unit = 1
        else:
            unit = param.BOHR

        atoms = []
        line = f.readline()
        while line:
            if '[GTO]' in line:
                break
            dat = line.split()
            symb, atmid, chg = dat[:3]
            coord = numpy.array([float(x) for x in dat[3:]])*unit
            atoms.append((gto.mole._std_symbol(symb)+atmid, coord))
            line = f.readline()
        mol.atom = atoms

        def read_one_bas(lsym, nb, fac):
            fac = float(fac)
            bas = [param.ANGULARMAP[lsym],]
            for i in range(int(nb)):
                dat = _d2e(f.readline()).split()
                bas.append((float(dat[0]), float(dat[1])*fac))
            return bas
        basis = {}
        line = f.readline()

# Be careful with the atom sequence in [GTO] session, it does not correspond
# to the atom sequence in [Atoms] session.
        atom_seq = []
        while line:
            if '[' in line:
                break
            dat = line.split()
            if len(dat) == 0:
                pass
            elif dat[0].isdigit():
                atom_seq.append(int(dat[0])-1)
                symb = mol.atom[int(dat[0])-1][0]
                basis[symb] = []
            elif dat[0] in 'spdfghij':
                lsym, nb, fac = dat
                basis[symb].append(read_one_bas(lsym, nb, fac))
            line = f.readline()
        mol.basis = basis
        mol.atom = [mol.atom[i] for i in atom_seq]

        mol._cart_gto = True
        while line:
            if '[5d]' in line or '[9g]' in line:
                mol._cart_gto = False
            elif '[MO]' in line:
                break
            line = f.readline()
        try:
            mol.build(0, 0)
        except RuntimeError:
            mol.build(0, 0, spin=1)

        data = f.read()
        data = data.split('Sym')[1:]
        irrep_labels = []
        mo_energy = []
        spins = []
        mo_occ = []
        mo_coeff = []
        for rawd in data:
            lines = rawd.split('\n')
            irrep_labels.append(lines[0].split('=')[1].strip())
            orb = []
            for line in lines[1:]:
                if line.strip() == '':
                    continue
                elif 'Ene' in line:
                    mo_energy.append(float(_d2e(line).split('=')[1].strip()))
                elif 'Spin' in line:
                    spins.append(line.split('=')[1].strip())
                elif 'Occ' in line:
                    mo_occ.append(float(_d2e(line.split('=')[1].strip())))
                else:
                    orb.append(float(_d2e(line.split()[1])))
            mo_coeff.append(orb)
        mo_energy = numpy.array(mo_energy)
        mo_occ = numpy.array(mo_occ)
        if mol._cart_gto:
            aoidx = numpy.argsort(order_ao_index(mol, cart=True))
            mo_coeff = (numpy.array(mo_coeff).T)[aoidx]
# AO are assumed to be normalized in molpro molden file
            s = mol.intor('cint1e_ovlp_cart')
            mo_coeff = numpy.einsum('i,ij->ij', numpy.sqrt(1/s.diagonal()), mo_coeff)
        else:
            aoidx = numpy.argsort(order_ao_index(mol))
            mo_coeff = (numpy.array(mo_coeff).T)[aoidx]
        return mol, mo_energy, mo_coeff, mo_occ, irrep_labels, spins

def first_token(stream, key):
    line = stream.readline()
    while line:
        if key in line:
            return line
        line = stream.readline()

def _d2e(token):
    return token.replace('D', 'e').replace('d', 'e')

def header(mol, fout, ignore_h=False):
    if ignore_h:
        mol = remove_high_l(mol)
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

def order_ao_index(mol, cart=False):
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
    if cart:
        for ib in range(mol.nbas):
            l = mol.bas_angular(ib)
            for n in range(mol.bas_nctr(ib)):
                if l == 2:
                    idx.extend([off+0,off+3,off+5,off+1,off+2,off+4])
                elif l == 3:
                    idx.extend([off+0,off+6,off+9,off+3,off+1,
                                off+2,off+5,off+8,off+7,off+4])
                elif l == 4:
                    idx.extend([off+0 , off+10, off+14, off+1 , off+2 ,
                                off+6 , off+11, off+9 , off+13, off+3 ,
                                off+5 , off+12, off+4 , off+7 , off+8 ,])
                elif l > 4:
                    raise RuntimeError('l=5 is not supported')
                else:
                    idx.extend(range(off,off+(l+1)*(l+2)//2))
                off += (l+1)*(l+2)//2
    else:  # spheric orbitals
        for ib in range(mol.nbas):
            l = mol.bas_angular(ib)
            for n in range(mol.bas_nctr(ib)):
                if l == 2:
                    idx.extend([off+2,off+3,off+1,off+4,off+0])
                elif l == 3:
                    idx.extend([off+3,off+4,off+2,off+5,off+1,off+6,off+0])
                elif l == 4:
                    idx.extend([off+4,off+5,off+3,off+6,off+2,
                                off+7,off+1,off+8,off+0])
                elif l > 4:
                    raise RuntimeError('l=5 is not supported')
                else:
                    idx.extend(range(off,off+l*2+1))
                off += l * 2 + 1
    return idx

def remove_high_l(mol, mo_coeff=None):
    '''Remove high angular momentum (l >= 5) functions before dumping molden file.
    If molden function raised error message ``RuntimeError l=5 is not supported``,
    you can use this function to format orbitals.
    
    Note the formated orbitals may have normalization problem.  Some visualization
    tool will complain about the orbital normalization error.

    Examples:

    >>> mol1, orb1 = remove_high_l(mol, mf.mo_coeff)
    >>> molden.from_mo(mol1, outputfile, orb1)
    '''
    pmol = mol.copy()
    pmol.basis = {}
    for symb, bas in mol._basis.items():
        pmol.basis[symb] = [b for b in bas if b[0] <= 4]
    pmol.build(0, 0)
    if mo_coeff is None:
        return pmol
    else:
        k = 0
        idx = []
        for ib in range(mol.nbas):
            l = mol.bas_angular(ib)
            nc = mol.bas_nctr(ib)
            if l <= 4:
                idx.append(range(k, k+(l*2+1)*nc))
                k += (l*2+1) * nc
        idx = numpy.hstack(idx)
        return pmol, mo_coeff[idx]



if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    import tempfile
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

    ftmp = tempfile.NamedTemporaryFile()
    from_mo(mol, ftmp.name, m.mo_coeff)

    print(parse(ftmp.name))
