#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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
Non-relativistic nuclear spin-rotation tensors for RHF

Refs:
[1] J. Gauss, K. Ruud, T. Helgaker, J. Chem. Phys., 105, 2804 (1996)
'''


import time
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.prop.nmr import rhf as rhf_nmr
from pyscf.prop.ssc.rhf import _atom_gyro_list
from pyscf.prop.magnetizability import rhf as rhf_mag
from pyscf.prop.rotational_gtensor.rhf import inertia_tensor, _safe_solve
from pyscf.data import nist


nuc_magneton = .5 * (nist.E_MASS/nist.PROTON_MASS)  # e*hbar/2m
AU2KHZ = nist.HARTREE2J / nist.PLANCK * nuc_magneton ** 2 * 2/1e3
del nuc_magneton

def dia(nsrobj, gauge_orig=None, shielding_nuc=None, dm0=None):
    '''Diamagnetic part of NSR tensors.
    '''
    if shielding_nuc is None: shielding_nuc = nsrobj.shielding_nuc
    if dm0 is None: dm0 = nsrobj._scf.make_rdm1()

    mol = nsrobj.mol
    im, mass_center = inertia_tensor(mol)
    if gauge_orig is None:
        ao_coords = rhf_mag._get_ao_coords(mol)
        # Eq. (34) of JCP, 105, 2804
        nsr_dia = rhf_nmr.dia(nsrobj, gauge_orig, shielding_nuc, dm0)
        for n, atm_id in enumerate(shielding_nuc):
            coord = mol.atom_coord(atm_id)
            with mol.with_common_origin(coord):
                with mol.with_rinv_origin(coord):
                    # a11part = (B dot) -1/2 frac{\vec{r}_N}{r_N^3} r_N (dot mu)
                    h11 = mol.intor('int1e_cg_a11part', comp=9)
            e11 = numpy.einsum('xpq,qp->x', h11, dm0).reshape(3,3)
            nsr_dia[n] -= e11 - numpy.eye(3) * e11.trace()
            nsr_dia[n] *= 2
    else:
        nsr_dia = []
        for n, atm_id in enumerate(shielding_nuc):
            coord = mol.atom_coord(atm_id)
            with mol.with_rinv_origin(coord):
                with mol.with_common_origin(gauge_orig):
                    # a11part = (B dot) -1/2 frac{\vec{r}_N}{r_N^3} (r-R_c) (dot mu)
                    h11 = mol.intor('int1e_cg_a11part', comp=9)
                e11 = numpy.einsum('xpq,qp->x', h11, dm0).reshape(3,3)
                with mol.with_common_origin(coord):
                    # a11part = (B dot) -1/2 frac{\vec{r}_N}{r_N^3} r_N (dot mu)
                    h11 = mol.intor('int1e_cg_a11part', comp=9)
                # e11 ~ (B dot) -1/2 frac{\vec{r}_N}{r_N^3} (R_N-R_c) (dot mu)
                e11 -= numpy.einsum('xpq,qp->x', h11, dm0).reshape(3,3)
            e11 = e11 - numpy.eye(3) * e11.trace()
            nsr_dia.append(e11)

    nsr_dia = _safe_solve(im, numpy.asarray(nsr_dia))
    unit = _atom_gyro_list(mol)[shielding_nuc] * nist.ALPHA**2
    return numpy.einsum('ixy,i->ixy', nsr_dia, unit)


def para(nsrobj, mo10=None, mo_coeff=None, mo_occ=None, shielding_nuc=None):
    '''Paramagnetic part of NSR shielding tensors.
    '''
    if shielding_nuc is None: shielding_nuc = nsrobj.shielding_nuc

    # The first order Hamiltonian for rotation part is the same to the
    # first order Hamiltonian for magnetic field except a factor of 2.
    nsr_para = rhf_nmr.para(nsrobj, mo10, mo_coeff, mo_occ,
                            shielding_nuc)[0] * 2

    mol = nsrobj.mol
    im, mass_center = inertia_tensor(mol)
    nsr_para = _safe_solve(im, nsr_para)
    unit = _atom_gyro_list(mol)[shielding_nuc] * nist.ALPHA**2
    return numpy.einsum('ixy,i->ixy', nsr_para, unit)


def nuc(mol, shielding_nuc):
    '''Nuclear contributions'''
    im, mass_center = inertia_tensor(mol)
    charges = mol.atom_charges()
    coords = mol.atom_coords()

    nsr_nuc = []
    for n, atm_id in enumerate(shielding_nuc):
        rkl = coords - coords[atm_id]
        d = numpy.linalg.norm(rkl, axis=1)
        d[atm_id] = 1e100
        e11 = numpy.einsum('z,zx,zy->xy', charges/d**3, rkl, rkl)
        e11 = numpy.eye(3) * e11.trace() - e11
        nsr_nuc.append(e11)

    nsr_nuc = _safe_solve(im, numpy.asarray(nsr_nuc))
    unit = _atom_gyro_list(mol)[shielding_nuc] * nist.ALPHA**2
    return numpy.einsum('ixy,i->ixy', nsr_nuc, unit)


class NSR(rhf_nmr.NMR):
    '''Nuclear-spin rotation tensors'''

    def kernel(self):
        cput0 = (time.clock(), time.time())
        self.check_sanity()
        self.dump_flags()

        e_dia = self.dia(self.gauge_orig)
        self.mo10, self.mo_e10 = self.solve_mo1()
        e_para = self.para(self.mo10)
        e_nuc = nuc(self.mol, self.shielding_nuc)

        e_dia *= AU2KHZ
        e_para*= AU2KHZ
        e_nuc *= AU2KHZ
        e11 = e_para + e_dia + e_nuc

        logger.timer(self, 'NSR', *cput0)
        if self.verbose >= logger.NOTE:
            for i, atm_id in enumerate(self.shielding_nuc):
                _write(self.stdout, e11[i],
                       '\ntotal NSR of atom %d %s' \
                       % (atm_id, self.mol.atom_symbol(atm_id)))
                _write(self.stdout, e11[i], '\nNuclear spin rotation (kHz)')
                _write(self.stdout, e_dia[i], 'dia-magnetic contribution (kHz)')
                _write(self.stdout, e_para[i], 'para-magnetic contribution (kHz)')
        return e11

    dia = dia
    para = para

from pyscf import scf
scf.hf.RHF.NSR = lib.class_as_method(NSR)

def _write(stdout, nsr3x3, title):
    stdout.write('%s\n' % title)
    stdout.write('mu_x %s\n' % str(nsr3x3[0]))
    stdout.write('mu_y %s\n' % str(nsr3x3[1]))
    stdout.write('mu_z %s\n' % str(nsr3x3[2]))
    stdout.flush()


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    mol = gto.Mole()
    mol.verbose = 7
    mol.output = '/dev/null'
    mol.atom = '''h  ,  0.   0.   0.917
                  f  ,  0.   0.   0.
                  '''
    mol.basis = 'dzp'
    mol.build()

    mf = scf.RHF(mol).run()
    rotg = mf.NSR()
    m = rotg.kernel()
    print(m[1,0,0] - -274.44236312671563)
    print(lib.finger(m) - 26.68194604747653)

    rotg.gauge_orig = (0,0,.917/lib.param.BOHR)
    m = rotg.kernel()
    print(m[0,0,0] - 274.9496549281562)
    print(lib.finger(m) - 96.22450556074686)

    mol.atom = '''C  ,  0.   0.   0.
                  O  ,  0.   0.   1.1283
                  '''
    mol.basis = 'ccpvdz'
    mol.nucprop = {'C': {'mass': 13}}
    mol.build()
    mf = scf.RHF(mol).run()
    rotg = NSR(mf)
    m = rotg.kernel()
    print(m[0,0,0] - -34.6383402509185)
    print(lib.finger(m) - -12.115914020554674)

    mol.atom='''O      0.   0.       0.
                H      0.  -0.757    0.587
                H      0.   0.757    0.587'''
    mol.basis = 'ccpvdz'
    mol.build()
    mf = scf.RHF(mol).run()
    rotg = NSR(mf)
    m = rotg.kernel()
    print(lib.finger(m) - -66.68587704814556)

