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
This interface requires the cppe library

https://github.com/maxscheurer/cppe
arXiv:1804.03598

The CPPE library needs to be built from sources (according to the CPPE document):

mkdir build && cd build && cmake -DENABLE_PYTHON_INTERFACE=ON .. && make

If successfully built, find the directory where the file cppe.*.so locates
then put the directory in PYTHONPATH.

The potential file required by CPPE library needs to be generated from the
PyFraME library  https://gitlab.com/FraME-projects/PyFraME
'''

import sys
import numpy

try:
    import cppe
except:
    sys.stderr.write('cppe library was not found\n')
    sys.stderr.write(__doc__)
    raise

from pyscf import lib
from pyscf.lib import logger
from pyscf import gto
from pyscf import df
from pyscf.solvent import _attach_solvent

@lib.with_doc(_attach_solvent._for_scf.__doc__)
def pe_for_scf(mf, solvent_obj, dm=None):
    if not isinstance(solvent_obj, PolEmbed):
        solvent_obj = PolEmbed(mf.mol, solvent_obj)
    return _attach_solvent._for_scf(mf, solvent_obj, dm)

@lib.with_doc(_attach_solvent._for_casscf.__doc__)
def pe_for_casscf(mc, solvent_obj, dm=None):
    if not isinstance(solvent_obj, PolEmbed):
        solvent_obj = PolEmbed(mc.mol, solvent_obj)
    return _attach_solvent._for_casscf(mc, solvent_obj, dm)

@lib.with_doc(_attach_solvent._for_casci.__doc__)
def pe_for_casci(mc, solvent_obj, dm=None):
    if not isinstance(solvent_obj, PolEmbed):
        solvent_obj = PolEmbed(mc.mol, solvent_obj)
    return _attach_solvent._for_casci(mc, solvent_obj, dm)

@lib.with_doc(_attach_solvent._for_post_scf.__doc__)
def pe_for_post_scf(method, solvent_obj, dm=None):
    if not isinstance(solvent_obj, PolEmbed):
        solvent_obj = PolEmbed(method.mol, solvent_obj)
    return _attach_solvent._for_post_scf(method, solvent_obj, dm)

@lib.with_doc(_attach_solvent._for_tdscf.__doc__)
def pe_for_tdscf(method, solvent_obj, dm=None):
    if not isinstance(solvent_obj, PolEmbed):
        solvent_obj = PolEmbed(method.mol, solvent_obj)
    return _attach_solvent._for_tdscf(method, solvent_obj, dm)


class PolEmbed(lib.StreamObject):
    def __init__(self, mol, options_or_potfile):
        self.mol = mol
        self.stdout = mol.stdout
        self.verbose = mol.verbose
        self.max_memory = mol.max_memory

        self.frozen = False
        # FIXME: Should the solvent in PE model by default has the character
        # of rapid process?
        self.equilibrium_solvation = False

##################################################
# don't modify the following attributes, they are not input options
        if isinstance(options_or_potfile, str):
            options = cppe.PeOptions()
            options.potfile = options_or_potfile
        else:
            options = options_or_potfile

        if not isinstance(options, cppe.PeOptions):
            raise TypeError("Invalid type for options.")

        self.options = options
        self.cppe_state = self._create_cppe_state(mol)
        self.potentials = self.cppe_state.potentials
        self.V_es = None

        # e (the dielectric correction) and v (the additional potential) are
        # updated during the SCF iterations
        self.e = None
        self.v = None
        self._dm = None

        self._keys = set(self.__dict__.keys())

    def dump_flags(self, verbose=None):
        logger.info(self, '******** %s flags ********', self.__class__)
        options = self.options
        logger.info(self, 'frozen = %s'       , self.frozen)
        logger.info(self, 'equilibrium_solvation = %s', self.equilibrium_solvation)
        logger.info(self, "cppe.potfile                  = %s", options.potfile)
        logger.info(self, "cppe.iso_pol                  = %s", options.iso_pol)
        logger.info(self, "cppe.induced_thresh           = %s", options.induced_thresh)
        logger.info(self, "cppe.do_diis                  = %s", options.do_diis)
        logger.info(self, "cppe.diis_start_norm          = %s", options.diis_start_norm)
        logger.info(self, "cppe.maxiter                  = %s", options.maxiter)
        logger.info(self, "cppe.damp_induced             = %s", options.damp_induced)
        logger.info(self, "cppe.damping_factor_induced   = %s", options.damping_factor_induced)
        logger.info(self, "cppe.damp_multipole           = %s", options.damp_multipole)
        logger.info(self, "cppe.damping_factor_multipole = %s", options.damping_factor_multipole)
        logger.info(self, "cppe.pe_border                = %s", options.pe_border)
        return self

    def _create_cppe_state(self, mol):
        cppe_mol = cppe.Molecule()
        for z, coord in zip(mol.atom_charges(), mol.atom_coords()):
            cppe_mol.append(cppe.Atom(z, *coord))

        def callback(output):
            logger.info(self, output)
        cppe_state = cppe.CppeState(self.options, cppe_mol, callback)
        cppe_state.calculate_static_energies_and_fields()
        return cppe_state

    def reset(self, mol=None):
        '''Reset mol and clean up relevant attributes for scanner mode'''
        if mol is not None:
            self.mol = mol
        self.cppe_state = self._create_cppe_state(mol)
        self.potentials = self.cppe_state.potentials
        self.V_es = None
        return self

    def kernel(self, dm, elec_only=False):
        '''
        '''
        if not (isinstance(dm, numpy.ndarray) and dm.ndim == 2):
            # spin-traced DM for UHF or ROHF
            dm = dm[0] + dm[1]

        if self.V_es is None:
            V_es = numpy.zeros((self.mol.nao, self.mol.nao),
                               dtype=numpy.float64)
            for p in self.potentials:
                moments = []
                for m in p.multipoles:
                    m.remove_trace()
                    moments.append(m.values)
                V_es += self._compute_multipole_potential_integrals(p.position,
                                                                    m.k,
                                                                    moments)
            self.V_es = V_es

        self.cppe_state.energies["Electrostatic"]["Electronic"] = (
            numpy.einsum('ij,ij->', self.V_es, dm)
        )

        n_sitecoords = 3 * self.cppe_state.get_polarizable_site_number()
        V_ind = numpy.zeros((self.mol.nao, self.mol.nao),
                            dtype=numpy.float64)
        if n_sitecoords:
            # TODO: use list comprehensions
            current_polsite = 0
            elec_fields = numpy.zeros(n_sitecoords, dtype=numpy.float64)
            for p in self.potentials:
                if not p.is_polarizable:
                    continue
                elec_fields_s = self._compute_field(p.position, dm)
                elec_fields[3*current_polsite:3*current_polsite + 3] = elec_fields_s
                current_polsite += 1
            self.cppe_state.update_induced_moments(elec_fields, elec_only)
            induced_moments = numpy.array(self.cppe_state.get_induced_moments())
            current_polsite = 0
            for p in self.potentials:
                if not p.is_polarizable:
                    continue
                site = p.position
                V_ind += self._compute_field_integrals(site=site,
                                                       moment=induced_moments[3*current_polsite:3*current_polsite + 3])
                current_polsite += 1
        e = self.cppe_state.total_energy
        if not elec_only:
            vmat = self.V_es + V_ind
        else:
            vmat = V_ind
            e = self.cppe_state.energies["Polarization"]["Electronic"]
        logger.info(self, 'Polarizable embedding energy = %.15g', e)

        self.e = e
        self.v = vmat
        return e, vmat

    def _compute_multipole_potential_integrals(self, site, order, moments):
        if order > 2:
            raise NotImplementedError("""Multipole potential integrals not
                                      implemented for order > 2.""")
        self.mol.set_rinv_orig(site)
        # TODO: only calculate up to requested order!
        integral0 = self.mol.intor("int1e_rinv")
        integral1 = self.mol.intor("int1e_iprinv") + self.mol.intor("int1e_iprinv").transpose(0, 2, 1)
        integral2 = self.mol.intor("int1e_ipiprinv") + self.mol.intor("int1e_ipiprinv").transpose(0, 2, 1) + 2.0 * self.mol.intor("int1e_iprinvip")

        # k = 2: 0,1,2,4,5,8 = XX, XY, XZ, YY, YZ, ZZ
        # add the lower triangle to the upper triangle, i.e.,
        # XY += YX : 1 + 3
        # XZ += ZX : 2 + 6
        # YZ += ZY : 5 + 7
        # and divide by 2
        integral2[1] += integral2[3]
        integral2[2] += integral2[6]
        integral2[5] += integral2[7]
        integral2[1] *= 0.5
        integral2[2] *= 0.5
        integral2[5] *= 0.5

        op = integral0 * moments[0] * cppe.prefactors(0)
        if order > 0:
            op += numpy.einsum('aij,a->ij', integral1,
                               moments[1] * cppe.prefactors(1))
        if order > 1:
            op += numpy.einsum('aij,a->ij',
                               integral2[[0, 1, 2, 4, 5, 8], :, :],
                               moments[2] * cppe.prefactors(2))

        return op

    def _compute_field_integrals(self, site, moment):
        self.mol.set_rinv_orig(site)
        integral = self.mol.intor("int1e_iprinv") + self.mol.intor("int1e_iprinv").transpose(0, 2, 1)
        op = numpy.einsum('aij,a->ij', integral, -1.0*moment)
        return op

    def _compute_field(self, site, D):
        self.mol.set_rinv_orig(site)
        integral = self.mol.intor("int1e_iprinv") + self.mol.intor("int1e_iprinv").transpose(0, 2, 1)
        return numpy.einsum('ij,aij->a', D, integral)

    def _B_dot_x(self, dm):
        dms = numpy.asarray(dm)
        dm_shape = dms.shape
        nao = dm_shape[-1]
        dms = dms.reshape(-1,nao,nao)
        v_pe_ao = [self.kernel(x, elec_only=True)[1] for x in dms]
        return numpy.asarray(v_pe_ov).reshape(dm_shape)

    def nuc_grad_method(self, grad_method):
        raise NotImplementedError("Nuclear gradients not implemented for PE.")

if __name__ == '__main__':
    import tempfile
    from pyscf import gto
    from pyscf.solvent import PE
    from pyscf.solvent import pol_embed
    mol = gto.M(atom='''
           6        0.000000    0.000000   -0.542500
           8        0.000000    0.000000    0.677500
           1        0.000000    0.935307   -1.082500
           1        0.000000   -0.935307   -1.082500
                ''', basis='sto3g')
    mf = mol.RHF()
    with tempfile.NamedTemporaryFile() as f:
        f.write(b'''!
@COORDINATES
3
AA
O     3.53300000    2.99600000    0.88700000      1
H     4.11100000    3.13200000    1.63800000      2
H     4.10500000    2.64200000    0.20600000      3
@MULTIPOLES
ORDER 0
3
1     -0.67444000
2      0.33722000
3      0.33722000
@POLARIZABILITIES
ORDER 1 1
3
1      5.73935000     0.00000000     0.00000000     5.73935000     0.00000000     5.73935000
2      2.30839000     0.00000000     0.00000000     2.30839000     0.00000000     2.30839000
3      2.30839000     0.00000000     0.00000000     2.30839000     0.00000000     2.30839000
EXCLISTS
3 3
1   2  3
2   1  3
3   1  2''')
        f.flush()
        pe_options = cppe.PeOptions()
        pe_options.potfile = f.name
        #pe = pol_embed.PolEmbed(mol, pe_options)
        #mf = PE(mf, pe).run()
        mf = PE(mf, pe_options).run()
        print(mf.e_tot - -112.35232445743728)
        print(mf.with_solvent.e - 0.00020182314249546455)
