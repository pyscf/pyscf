#!/usr/bin/env python
# Copyright 2019-2020 The PySCF Developers. All Rights Reserved.
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

GitHub:      https://github.com/maxscheurer/cppe
Code:        10.5281/zenodo.3345696
Publication: https://doi.org/10.1021/acs.jctc.9b00758

The CPPE library can be installed via:
pip install git+https://github.com/maxscheurer/cppe.git

The potential file required by CPPE library needs to be generated from the
PyFraME library  https://gitlab.com/FraME-projects/PyFraME
'''

import sys
import numpy
from pkg_resources import parse_version

try:
    import cppe
except ModuleNotFoundError:
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
    scf_solvent = getattr(method._scf, 'with_solvent', None)
    assert scf_solvent is None or isinstance(scf_solvent, PolEmbed)

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
            options = {"potfile": options_or_potfile}
        else:
            options = options_or_potfile

        min_version = "0.2.0"
        if parse_version(cppe.__version__) < parse_version(min_version):
            raise ModuleNotFoundError("cppe version {} is required at least. "
                                      "Version {}"
                                      " was found.".format(min_version,
                                                           cppe.__version__))

        if not isinstance(options, dict):
            raise TypeError("Options should be a dictionary.")

        self.options = options
        self.cppe_state = self._create_cppe_state(mol)
        self.potentials = self.cppe_state.potentials
        self.V_es = None

        # e (the electrostatic and induction energy)
        # and v (the additional potential) are
        # updated during the SCF iterations
        self.e = None
        self.v = None
        self._dm = None

        self._keys = set(self.__dict__.keys())

    def dump_flags(self, verbose=None):
        logger.info(self, '******** %s flags ********', self.__class__)
        options = self.cppe_state.options
        option_keys = cppe.valid_option_keys
        logger.info(self, 'frozen = %s'       , self.frozen)
        logger.info(self, 'equilibrium_solvation = %s', self.equilibrium_solvation)
        for key in option_keys:
            logger.info(self, "cppe.%s = %s", key, options[key])
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

        e, v = self._exec_cppe(dm, elec_only)
        logger.info(self, 'Polarizable embedding energy = %.15g', e)

        self.e = e
        self.v = v
        return e, v

    def _exec_cppe(self, dm, elec_only=False):
        dms = numpy.asarray(dm)
        is_single_dm = dms.ndim == 2

        nao = dms.shape[-1]
        dms = dms.reshape(-1,nao,nao)
        n_dm = dms.shape[0]

        if self.V_es is None:
            positions = numpy.array([p.position for p in self.potentials])
            moments = []
            orders = []
            for p in self.potentials:
                p_moments = []
                for m in p.multipoles:
                    m.remove_trace()
                    p_moments.append(m.values)
                orders.append(m.k)
                moments.append(p_moments)
            self.V_es = self._compute_multipole_potential_integrals(positions, orders, moments)

        e_static = numpy.einsum('ij,xij->x', self.V_es, dms)
        self.cppe_state.energies["Electrostatic"]["Electronic"] = (
            e_static[0]
        )

        positions = numpy.array([p.position for p in self.potentials
                                 if p.is_polarizable])
        n_sites = positions.shape[0]
        V_ind = numpy.zeros((n_dm, nao, nao))

        e_tot = []
        e_pol = []
        if n_sites > 0:
            #:elec_fields = self._compute_field(positions, dms)
            fakemol = gto.fakemol_for_charges(positions)
            j3c = df.incore.aux_e2(self.mol, fakemol, intor='int3c2e_ip1')
            elec_fields = (numpy.einsum('aijg,nij->nga', j3c, dms) +
                           numpy.einsum('aijg,nji->nga', j3c, dms))

            induced_moments = numpy.empty((n_dm, n_sites * 3))
            for i_dm in range(n_dm):
                self.cppe_state.update_induced_moments(elec_fields[i_dm].ravel(), elec_only)
                induced_moments[i_dm] = numpy.array(self.cppe_state.get_induced_moments())

                e_tot.append(self.cppe_state.total_energy)
                e_pol.append(self.cppe_state.energies["Polarization"]["Electronic"])

            induced_moments = induced_moments.reshape(n_dm, n_sites, 3)
            #:V_ind = self._compute_field_integrals(positions, induced_moments)
            V_ind = numpy.einsum('aijg,nga->nij', j3c, -induced_moments)
            V_ind = V_ind + V_ind.transpose(0, 2, 1)

        if not elec_only:
            vmat = self.V_es + V_ind
            e = numpy.array(e_tot)
        else:
            vmat = V_ind
            e = numpy.array(e_pol)

        if is_single_dm:
            e = e[0]
            vmat = vmat[0]
        return e, vmat

    def _compute_multipole_potential_integrals(self, sites, orders, moments):
        orders = numpy.asarray(orders)
        if numpy.any(orders > 2):
            raise NotImplementedError("""Multipole potential integrals not
                                      implemented for order > 2.""")

        # order 0
        fakemol = gto.fakemol_for_charges(sites)
        integral0 = df.incore.aux_e2(self.mol, fakemol, intor='int3c2e')
        moments_0 = numpy.array([m[0] for m in moments])
        op = numpy.einsum('ijg,ga->ij', integral0, moments_0 * cppe.prefactors(0))

        # order 1
        if numpy.any(orders >= 1):
            idx = numpy.where(orders >= 1)[0]
            fakemol = gto.fakemol_for_charges(sites[idx])
            integral1 = df.incore.aux_e2(self.mol, fakemol, intor='int3c2e_ip1')
            moments_1 = numpy.array([moments[i][1] for i in idx])
            v = numpy.einsum('aijg,ga,a->ij', integral1, moments_1, cppe.prefactors(1))
            op += v + v.T

        if numpy.any(orders >= 2):
            idx = numpy.where(orders >= 2)[0]
            fakemol = gto.fakemol_for_charges(sites[idx])
            n_sites = idx.size
            # moments_2 is the lower triangler of
            # [[XX, XY, XZ], [YX, YY, YZ], [ZX, ZY, ZZ]] i.e.
            # XX, XY, XZ, YY, YZ, ZZ = 0,1,2,4,5,8
            # symmetrize it to the upper triangler part
            # XX, YX, ZX, YY, ZY, ZZ = 0,3,6,4,7,8
            m2 = numpy.einsum('ga,a->ga', [moments[i][2] for i in idx],
                              cppe.prefactors(2))
            moments_2 = numpy.zeros((n_sites, 9))
            moments_2[:, [0, 1, 2, 4, 5, 8]]  = m2
            moments_2[:, [0, 3, 6, 4, 7, 8]] += m2
            moments_2 *= .5

            integral2 = df.incore.aux_e2(self.mol, fakemol, intor='int3c2e_ipip1')
            v = numpy.einsum('aijg,ga->ij', integral2, moments_2)
            op += v + v.T
            integral2 = df.incore.aux_e2(self.mol, fakemol, intor='int3c2e_ipvip1')
            op += numpy.einsum('aijg,ga->ij', integral2, moments_2) * 2

        return op

    def _compute_field_integrals(self, sites, moments):
        mol = self.mol
        fakemol = gto.fakemol_for_charges(sites)
        j3c = df.incore.aux_e2(mol, fakemol, intor='int3c2e_ip1')
        op = numpy.einsum('aijg,nga->nij', j3c, -moments)
        op = op + op.transpose(0, 2, 1)
        return op

    def _compute_field(self, sites, Ds):
        mol = self.mol
        fakemol = gto.fakemol_for_charges(sites)
        j3c = df.incore.aux_e2(mol, fakemol, intor='int3c2e_ip1')
        field = (numpy.einsum('aijg,nij->nga', j3c, Ds) +
                 numpy.einsum('aijg,nji->nga', j3c, Ds))
        return field

    def _B_dot_x(self, dm):
        dms = numpy.asarray(dm)
        dm_shape = dms.shape
        nao = dm_shape[-1]
        dms = dms.reshape(-1,nao,nao)
        v_pe_ao = [self._exec_cppe(x, elec_only=True)[1] for x in dms]
        return numpy.asarray(v_pe_ao).reshape(dm_shape)

    def nuc_grad_method(self, grad_method):
        raise NotImplementedError("Nuclear gradients not implemented for PE.")

if __name__ == '__main__':
    import tempfile
    from pyscf.solvent import PE
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
        pe_options = {"potfile": f.name}
        # pe = pol_embed.PolEmbed(mol, pe_options)
        #mf = PE(mf, pe).run()
        mf = PE(mf, pe_options).run()
        print(mf.e_tot - -112.35232445743728)
        print(mf.with_solvent.e - 0.00020182314249546455)
