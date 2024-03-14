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

# MOLDEN format:
# http://www.cmbi.ru.nl/molden/molden_format.html

import sys
import re
import numpy
import pyscf
from pyscf import lib
from pyscf import gto
from pyscf.lib import logger
from pyscf import __config__

IGNORE_H = getattr(__config__, 'molden_ignore_h', True)


def orbital_coeff(mol, fout, mo_coeff, spin='Alpha', symm=None, ene=None,
                  occ=None, ignore_h=IGNORE_H):
    from pyscf.symm import label_orb_symm
    if mol.cart:
        # pyscf Cartesian GTOs are not normalized. This may not be consistent
        # with the requirements of molden format. Normalize Cartesian GTOs here
        norm = mol.intor('int1e_ovlp').diagonal() ** .5
        mo_coeff = numpy.einsum('i,ij->ij', norm, mo_coeff)

    if ignore_h:
        mol, mo_coeff = remove_high_l(mol, mo_coeff)

    aoidx = order_ao_index(mol)
    nmo = mo_coeff.shape[1]
    if symm is None:
        symm = ['A']*nmo
        if mol.symmetry:
            try:
                symm = label_orb_symm(mol, mol.irrep_name, mol.symm_orb,
                                      mo_coeff, tol=1e-5)
            except ValueError as e:
                logger.warn(mol, str(e))
    if ene is None or len(ene) != nmo:
        ene = numpy.arange(nmo)
    assert (spin == 'Alpha' or spin == 'Beta')
    if occ is None:
        occ = numpy.zeros(nmo)
        neleca, nelecb = mol.nelec
        if spin == 'Alpha':
            occ[:neleca] = 1
        else:
            occ[:nelecb] = 1

    if spin == 'Alpha':
        # Avoid duplicated [MO] session when dumping beta orbitals
        fout.write('[MO]\n')

    for imo in range(nmo):
        fout.write(' Sym= %s\n' % symm[imo])
        fout.write(' Ene= %15.10g\n' % ene[imo])
        fout.write(' Spin= %s\n' % spin)
        fout.write(' Occup= %10.5f\n' % occ[imo])
        for i,j in enumerate(aoidx):
            fout.write(' %3d    %18.14g\n' % (i+1, mo_coeff[j,imo]))

def from_mo(mol, filename, mo_coeff, spin='Alpha', symm=None, ene=None,
            occ=None, ignore_h=IGNORE_H):
    '''Dump the given MOs in Molden format'''
    with open(filename, 'w') as f:
        header(mol, f, ignore_h)
        orbital_coeff(mol, f, mo_coeff, spin, symm, ene, occ, ignore_h)


def from_scf(mf, filename, ignore_h=IGNORE_H):
    '''Dump the given SCF object in Molden format'''
    dump_scf(mf, filename, ignore_h)
def dump_scf(mf, filename, ignore_h=IGNORE_H):
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

def from_mcscf(mc, filename, ignore_h=IGNORE_H, cas_natorb=False):
    mol = mc.mol
    dm1 = mc.make_rdm1()
    if cas_natorb:
        mo_coeff, _, mo_energy = mc.canonicalize(sort=True, cas_natorb=cas_natorb)
    else:
        mo_coeff, mo_energy = mc.mo_coeff, mc.mo_energy

    mo_inv = numpy.dot(mc._scf.get_ovlp(), mo_coeff)
    occ = numpy.einsum('pi,pq,qi->i', mo_inv, dm1, mo_inv)
    with open(filename, 'w') as f:
        header(mol, f, ignore_h)
        orbital_coeff(mol, f, mo_coeff, ene=mo_energy, occ=occ, ignore_h=ignore_h)

def from_chkfile(filename, chkfile, key='scf/mo_coeff', ignore_h=IGNORE_H):
    import pyscf.scf
    with open(filename, 'w') as f:
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
            if 'mo_occ' in dat:
                occ = dat['mo_occ']
            else:
                occ = None
            mo = dat['mo_coeff']

        if isinstance(ene, str) and ene == 'None':
            ene = None
        if isinstance(ene, str) and occ == 'None':
            occ = None
        if occ is not None and occ.ndim == 2:
            orbital_coeff(mol, f, mo[0], spin='Alpha', ene=ene[0], occ=occ[0],
                          ignore_h=ignore_h)
            orbital_coeff(mol, f, mo[1], spin='Beta', ene=ene[1], occ=occ[1],
                          ignore_h=ignore_h)
        else:
            orbital_coeff(mol, f, mo, ene=ene, occ=occ, ignore_h=ignore_h)


_SEC_RE = re.compile(r'\[[^]]+\]')

def _read_one_section(molden_fp):
    sec = [None]
    last_pos = 0
    while True:
        line = molden_fp.readline()
        if not line:
            break

        line = line.strip()
        if line == '' or line[0] == '#':  # comment or blank line
            continue

        mo = _SEC_RE.match(line)
        if mo:
            if sec[0] is None:
                sec[0] = line
            else:
                # Next section? rewind the fp pointer
                molden_fp.seek(last_pos)
                break
        else:
            sec.append(line)

        last_pos = molden_fp.tell()

    return sec

def _parse_natoms(lines, envs):
    envs['natm'] = natm = int(lines[1])
    return natm

def _parse_atoms(lines, envs):
    if 'ANG' in lines[0].upper():
        envs['unit'] = 1
    unit = envs['unit']

    envs['atoms'] = atoms = []
    for line in lines[1:]:
        dat = line.split()
        symb, atmid, chg = dat[:3]
        coord = numpy.array([float(x) for x in dat[3:]])*unit
        atoms.append((gto.mole._std_symbol(symb)+atmid, coord))

    if envs['natm'] is not None and envs['natm'] != len(atoms):
        sys.stderr.write('Number of atoms in section ATOMS does not equal to N_ATOMS\n')
    return atoms

def _parse_charge(lines, envs):
    mulliken_charges = [float(_d2e(x)) for x in lines[1:]]
    return mulliken_charges

def _parse_gto(lines, envs):
    mol = envs['mol']
    atoms = envs['atoms']
    basis = {}
    lines_iter = iter(lines)
    next(lines_iter)  # skip section header

# * Do not use iter() here. Python 2 and 3 are different in iter()
    def read_one_bas(lsym, nb, fac=1):
        fac = float(fac)
        if fac == float(0):
            fac = float(1)
        bas = [lib.param.ANGULARMAP[lsym.lower()],]
        for i in range(int(nb)):
            dat = _d2e(next(lines_iter)).split()
            bas.append((float(dat[0]), float(dat[1])*fac))
        return bas

# * Be careful with the atom sequence in [GTO] session, it does not correspond
# to the atom sequence in [Atoms] session.
    atom_seq = []

    for line in lines_iter:
        dat = line.split()
        if dat[0].isdigit():
            atom_seq.append(int(dat[0])-1)
            symb = atoms[int(dat[0])-1][0]
            basis[symb] = []

        elif dat[0].upper() in 'SPDFGHIJ':
            basis[symb].append(read_one_bas(*dat))

    mol.atom = [atoms[i] for i in atom_seq]
    uniq_atoms = {a[0] for a in mol.atom}

    # To avoid the mol.build() sort the basis, disable mol.basis and set the
    # internal data _basis directly. It is a workaround to solve issue #1961.
    # Mole.decontract_basis function should be rewritten to support
    # discontinuous bases that have the same angular momentum.
    mol.basis = {}
    _basis = gto.mole._parse_default_basis(basis, uniq_atoms)
    mol._basis = envs['basis'] = gto.format_basis(_basis, sort_basis=False)
    return mol

def _parse_mo(lines, envs):
    mol = envs['mol']
    if not mol._built:
        try:
            mol.build(0, 0)
        except RuntimeError:
            mol.build(0, 0, spin=1)

    irrep_labels = []
    mo_energy = []
    spins = []
    mo_occ = []
    mo_coeff_prim = [] # primary data, will be reworked for missing values
    coeff_idx = []
    mo_id = 0
    for line in lines[1:]:
        line = line.upper()
        if 'SYM' in line:
            irrep_labels.append(line.split('=')[1].strip())
        elif 'ENE' in line:
            mo_energy.append(float(_d2e(line).split('=')[1].strip()))
            mo_id = len(mo_energy) - 1
        elif 'SPIN' in line:
            spins.append(line.split('=')[1].strip())
        elif 'OCC' in line:
            mo_occ.append(float(_d2e(line.split('=')[1].strip())))
        else:
            ao_id, c = line.split()[:2]
            coeff_idx.append([int(ao_id) - 1, mo_id])
            mo_coeff_prim.append(float(c))

    coeff_idx = numpy.array(coeff_idx)
    number_of_aos, number_of_mos = coeff_idx.max(axis=0) + 1
    mo_coeff = numpy.zeros([number_of_aos, number_of_mos])
    mo_coeff[coeff_idx[:,0], coeff_idx[:,1]] = mo_coeff_prim

    mo_energy = numpy.array(mo_energy)
    mo_occ = numpy.array(mo_occ)
    aoidx = numpy.argsort(order_ao_index(mol))
    mo_coeff = mo_coeff[aoidx]
    if mol.cart:
        # Cartesian GTOs are normalized in molden format but they are not in pyscf
        s = mol.intor('int1e_ovlp')
        mo_coeff = numpy.einsum('i,ij->ij', numpy.sqrt(1/s.diagonal()), mo_coeff)


    return mol, mo_energy, mo_coeff, mo_occ, irrep_labels, spins


def _parse_core(lines, envs):
    mol = envs['mol']
    atoms = envs['atoms']
    for line in lines[1:]:
        dat = line.split(':')
        if dat[0].strip().isdigit():
            atm_id = int(dat[0].strip()) - 1
            nelec_core = int(dat[1].strip())
            mol.ecp[atoms[atm_id][0]] = [nelec_core, []]

    if mol.ecp:
        sys.stderr.write('\nECP were detected in the molden file.\n'
                         'Note Molden format does not support ECP data. '
                         'ECP information was lost when saving to molden format.\n\n')
    return mol.ecp

_SEC_PARSER = {'N_ATOMS'  : _parse_natoms,
               'ATOMS'    : _parse_atoms,
               'GTO'      : _parse_gto,
               'CHARGE'   : _parse_charge,
               'MO'       : _parse_mo,
               'CORE'     : _parse_core,
               'MOLDEN FORMAT' : lambda *args: None,}

_SEC_ORDER = ['N_ATOMS', 'ATOMS', 'GTO', 'CHARGE', 'MO', 'CORE', 'MOLDEN FORMAT']

def load(moldenfile, verbose=0):
    '''Extract mol and orbitals from molden file
    '''
    sec_kinds = {} # found sections and their lines are stored in this dic
    with open(moldenfile, 'r') as f:
        mol = gto.Mole()
        mol.cart = True
        tokens = {'natm'  : None,
                  'unit'  : lib.param.BOHR,
                  'mol'   : mol,
                  'atoms' : None,
                  'basis' : None,}

        while True:
            lines = _read_one_section(f)
            sec_title = lines[0]
            if sec_title is None:
                break

            sec_title = sec_title[1:sec_title.index(']')].upper()
            if sec_title in _SEC_PARSER:
                if sec_title not in sec_kinds:
                    sec_kinds.update({sec_title : [lines]})
                else:
                    sec_kinds[sec_title].append(lines)

            elif sec_title[:2] in ('5D', '7F', '9G'):
                mol.cart = False

            elif sec_title[:2] == '6D' or sec_title[:3] in ('10F', '15G'):
                mol.cart = True

            else:
                sys.stderr.write('Unknown section %s\n' % sec_title)

    mo_energy, mo_coeff, mo_occ, irrep_labels, spins = None, None, None, None, None

    for sec_kind in _SEC_ORDER:
        if sec_kind == 'MO' and 'MO' in sec_kinds:
            if len(sec_kinds['MO']) == 1:
                mol, mo_energy, mo_coeff, mo_occ, irrep_labels, spins = \
                        _parse_mo(sec_kinds['MO'][0], tokens)
                # If found only one MO section while 'B' appears in the spins
                # labels, the MOs so obtained are spin orbitals, with beta
                # orbitals at the second half of the mo_coeff matrix.
                if any(s[0] == 'B' for s in spins):
                    if mo_coeff.shape[0] == mo_coeff.shape[1]:
                        # general spin orbitals which allows to mix spin alpha
                        # and spin beta components in the same orbitals
                        raise NotImplementedError
                    else:
                        # Regular spin orbitals, alpha and beta do not mix
                        beta_idx = numpy.array([s[0] == 'B' for s in spins])
                        alpha_idx = ~beta_idx
                        mo_energy = mo_energy[alpha_idx], mo_energy[beta_idx]
                        mo_coeff = mo_coeff[:,alpha_idx], mo_coeff[:,beta_idx]
                        mo_occ = mo_occ[alpha_idx], mo_occ[beta_idx]
                        irrep_labels = numpy.array(irrep_labels)
                        irrep_labels = irrep_labels[alpha_idx], irrep_labels[beta_idx]
                        spins = numpy.array(spins)
                        spins = spins[alpha_idx], spins[beta_idx]

            elif len(sec_kinds['MO']) == 2:
                res_a = _parse_mo(sec_kinds['MO'][0], tokens)
                res_b = _parse_mo(sec_kinds['MO'][1], tokens)
                mo_energy, mo_coeff, mo_occ, irrep_labels, spins = \
                        list(zip(res_a[1:], res_b[1:]))
                mol = res_b[0]

        if sec_kind in sec_kinds:
            for n, content in enumerate(sec_kinds[sec_kind]):
                _SEC_PARSER[sec_kind](content, tokens)

    if isinstance(mo_occ, tuple):
        mol.spin = int(mo_occ[0].sum() - mo_occ[1].sum())

    if not mol._built:
        mol.build(0, 0)
    return mol, mo_energy, mo_coeff, mo_occ, irrep_labels, spins

parse = read = load

def _d2e(token):
    return token.replace('D', 'e').replace('d', 'e')

def header(mol, fout, ignore_h=IGNORE_H):
    if ignore_h:
        mol = remove_high_l(mol)[0]
    fout.write('[Molden Format]\n')
    fout.write('made by pyscf v[%s]\n' % pyscf.__version__)
    fout.write('[Atoms] (AU)\n')
    for ia in range(mol.natm):
        symb = mol.atom_pure_symbol(ia)
        chg = mol.atom_charge(ia)
        fout.write('%s   %d   %d   ' % (symb, ia+1, chg))
        coord = mol.atom_coord(ia)
        fout.write('%18.14f   %18.14f   %18.14f\n' % tuple(coord))

    fout.write('[GTO]\n')
    for ia, (sh0, sh1, p0, p1) in enumerate(mol.offset_nr_by_atom()):
        fout.write('%d 0\n' %(ia+1))
        for ib in range(sh0, sh1):
            l = mol.bas_angular(ib)
            nprim = mol.bas_nprim(ib)
            nctr = mol.bas_nctr(ib)
            es = mol.bas_exp(ib)
            cs = mol.bas_ctr_coeff(ib)
            for ic in range(nctr):
                fout.write(' %s   %2d 1.00\n' % (lib.param.ANGULAR[l], nprim))
                for ip in range(nprim):
                    fout.write('    %18.14g  %18.14g\n' % (es[ip], cs[ip,ic]))
        fout.write('\n')

    if mol.cart:
        fout.write('[6d]\n[10f]\n[15g]\n')
    else:
        fout.write('[5d]\n[7f]\n[9g]\n')

    if mol.has_ecp():  # See https://github.com/zorkzou/Molden2AIM
        fout.write('[core]\n')
        for ia in range(mol.natm):
            nelec_ecp_core = mol.atom_nelec_core(ia)
            if nelec_ecp_core != 0:
                fout.write('%s : %d\n' % (ia+1, nelec_ecp_core))
    fout.write('\n')

def order_ao_index(mol):
    # reorder d,f,g function to
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
    if mol.cart:
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
    else:  # spherical orbitals
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
    pmol._basis = {}
    for symb, bas in mol._basis.items():
        pmol._basis[symb] = [b for b in bas if b[0] <= 4]
    pmol.build(0, 0)
    if mo_coeff is None:
        return pmol, None
    else:
        p1 = 0
        idx = []
        for ib in range(mol.nbas):
            l = mol.bas_angular(ib)
            nc = mol.bas_nctr(ib)
            if mol.cart:
                nd = (l + 1) * (l + 2) // 2
            else:
                nd = l * 2 + 1
            p0, p1 = p1, p1 + nd * nc
            if l <= 4:
                idx.append(range(p0, p1))

        idx = numpy.hstack(idx)
        return pmol, mo_coeff[idx]



if __name__ == '__main__':
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

    ftmp = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
    from_mo(mol, ftmp.name, m.mo_coeff)

    print(parse(ftmp.name))
