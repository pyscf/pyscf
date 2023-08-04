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
Interface to CPPE, a library of polarizable embedding solvent model.
This interface requires the cppe library

GitHub:      https://github.com/maxscheurer/cppe
Code:        10.5281/zenodo.3345696
Publication: https://doi.org/10.1021/acs.jctc.9b00758

The CPPE library can be installed via:
        pip install cppe
or
        pip install git+https://github.com/maxscheurer/cppe.git

The potential file required by CPPE library needs to be generated from the
PyFraME library  https://gitlab.com/FraME-projects/PyFraME

References:

  [1] Olsen, J. M., Aidas, K., & Kongsted, J. (2010). Excited States in Solution
  through Polarizable Embedding. J. Chem. Theory Comput., 6 (12), 3721-3734.
  https://doi.org/10.1021/ct1003803

  [2] Olsen, J. M. H., & Kongsted, J. (2011). Molecular Properties through
  Polarizable Embedding. Advances in Quantum Chemistry (Vol. 61).
  https://doi.org/10.1016/B978-0-12-386013-2.00003-6
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
from pyscf.data import elements


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


# data from https://doi.org/10.1021/acs.jctc.9b01162
_pe_ecps = [
    ("X1", gto.parse_ecp(
        """
        X1 nelec 0
        X1 ul
        2      1.000000000000      0.000000000000
        X1 S
        2      0.509800000000      2.420000000000
        X1 P
        2      0.491650000000     -0.435900000000
        """)),
    ("X2", gto.parse_ecp(
        """
        X2 nelec 0
        X2 ul
        2      1.000000000000      0.000000000000
        X2 S
        2      2.047500000000     54.510000000000
        X2 P
        2      0.448150000000      1.465000000000
        X2 D
        2      0.492050000000     -0.838000000000
        """)),
    ("X3", gto.parse_ecp(
        """
        X3 nelec 0
        X3 ul
        2      1.000000000000      0.000000000000
        X3 S
        2      1.641000000000    275.000000000000
        X3 P
        2      0.273300000000      1.900000000000
        X3 D
        2      0.440000000000     -3.400000000000
        """))
]


def _get_element_row(symbol):
    """
    Helper function to determine the row of an element
    for choosing the correct ECP for PE(ECP)
    """
    nucchg = elements.charge(symbol)
    if nucchg <= 2:
        element_row = 0
    elif nucchg <= 10:
        element_row = 1
    elif nucchg <= 18:
        element_row = 2
    else:
        raise NotImplementedError("PE(ECP) only implemented for first, "
                                  "second, and third row elements")
    return element_row


class PolEmbed(lib.StreamObject):
    def __init__(self, mol, options_or_potfile):
        self.mol = mol
        self.stdout = mol.stdout
        self.verbose = mol.verbose
        self.max_memory = mol.max_memory

        # The maximum iterations and convergence tolerance to update solvent
        # effects in CASCI, CC, MP, CI, ... methods
        self.max_cycle = 20
        self.conv_tol = 1e-7
        self.state_id = 0

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

        min_version = "0.3.1"
        if parse_version(cppe.__version__) < parse_version(min_version):
            raise ModuleNotFoundError("cppe version {} is required at least. "
                                      "Version {}"
                                      " was found.".format(min_version,
                                                           cppe.__version__))

        if not isinstance(options, dict):
            raise TypeError("Options should be a dictionary.")

        self.options = options
        # use PE(ECP) repulsive potentials
        self.do_ecp = self.options.pop("ecp", False)
        # use effective external field (EEF)
        self.eef = self.options.pop("eef", False)
        self.cppe_state = self._create_cppe_state(mol)
        self.potentials = self.cppe_state.potentials
        self.V_es = None

        if self.do_ecp:
            # Use ECPs in the environment as repulsive potentials
            # with parameters from https://doi.org/10.1021/acs.jctc.9b01162
            ecpatoms = []
            # one set of parameters for each row of elements (first 3 rows supported)
            for p in self.potentials:
                if p.element == "X":
                    continue
                element_row = _get_element_row(p.element)
                ecp_label, _ = _pe_ecps[element_row]
                ecpatoms.append([ecp_label, p.x, p.y, p.z])
            self.ecpmol = gto.M(atom=ecpatoms, ecp={l: k for (l, k) in _pe_ecps},
                                basis={}, unit="Bohr")
            # add the normal mol to compute integrals
            self.ecpmol += self.mol

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
        logger.info(self, 'pe(ecp) repulsive potentials = %s', self.do_ecp)
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
        # logger.info(self, "Static energies and fields computed")
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

    def effective_dipole_operator(self):
        """
        Compute the derivatives of induced moments wrt each coordinate
        and form integrals for effective dipole operator (EEF)
        """
        dips = self.mol.intor_symmetric('int1e_r', comp=3)
        if self.eef:
            logger.info(self, "Computing effective dipole operator for EEF.")
            positions = self.cppe_state.positions_polarizable
            n_sites = positions.shape[0]
            induced_moments = self.cppe_state.induced_moments_eef()
            induced_moments = induced_moments.reshape(n_sites, 3, 3)
            fakemol = gto.fakemol_for_charges(positions)
            j3c = df.incore.aux_e2(self.mol, fakemol, intor='int3c2e_ip1')
            V_ind = numpy.einsum('aijg,gax->xij', j3c, -induced_moments)
            dips += V_ind + V_ind.transpose(0, 2, 1)
        return list(dips)

    def _exec_cppe(self, dm, elec_only=False):
        dms = numpy.asarray(dm)
        is_single_dm = dms.ndim == 2

        nao = dms.shape[-1]
        dms = dms.reshape(-1,nao,nao)
        n_dm = dms.shape[0]

        max_memory = self.max_memory

        if self.V_es is None:
            # very conservative estimate (based on multipole potential integrals)
            # when all sites have a charge, dipole, and quadrupole moment
            max_memreq = 10 * len(self.potentials) * nao**2 * 8.0/1e6
            n_chunks_el = 1
            if max_memreq >= max_memory:
                n_chunks_el = int(max_memreq // max_memory + 1)
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
            self.V_es = self._compute_multipole_potential_integrals(
                positions, orders, moments, n_chunks_el
            )
            if self.do_ecp:
                self.V_ecp = self.ecpmol.intor("ECPscalar")

        e_static = numpy.einsum('ij,xij->x', self.V_es, dms)
        self.cppe_state.energies["Electrostatic"]["Electronic"] = (
            e_static[0]
        )

        e_ecp = 0.0
        if self.do_ecp:
            e_ecp = numpy.einsum('ij,xij->x', self.V_ecp, dms)[0]

        positions = self.cppe_state.positions_polarizable
        n_sites = positions.shape[0]
        V_ind = numpy.zeros((n_dm, nao, nao))

        e_tot = []
        e_pol = []
        if n_sites > 0:
            max_memreq = 6 * n_sites * nao**2 * 8.0/1e6
            n_chunks_ind = 1
            if max_memreq >= max_memory:
                n_chunks_ind = int(max_memreq // max_memory + 1)

            chunks = numpy.array_split(positions, n_chunks_ind)
            elec_fields_chunk = []
            for chunk in chunks:
                fakemol = gto.fakemol_for_charges(chunk)
                j3c = df.incore.aux_e2(self.mol, fakemol, intor='int3c2e_ip1')
                elf = (numpy.einsum('aijg,nij->nga', j3c, dms) +
                       numpy.einsum('aijg,nji->nga', j3c, dms))
                elec_fields_chunk.append(elf)
            elec_fields = numpy.concatenate(elec_fields_chunk, axis=1)

            induced_moments = numpy.empty((n_dm, n_sites * 3))
            for i_dm in range(n_dm):
                self.cppe_state.update_induced_moments(elec_fields[i_dm].ravel(), elec_only)
                induced_moments[i_dm] = numpy.array(self.cppe_state.get_induced_moments())

                e_tot.append(self.cppe_state.total_energy + e_ecp)
                e_pol.append(self.cppe_state.energies["Polarization"]["Electronic"])

            induced_moments = induced_moments.reshape(n_dm, n_sites, 3)
            induced_moments_chunked = numpy.array_split(induced_moments, n_chunks_ind, axis=1)
            for pos_chunk, ind_chunk in zip(chunks, induced_moments_chunked):
                fakemol = gto.fakemol_for_charges(pos_chunk)
                j3c = df.incore.aux_e2(self.mol, fakemol, intor='int3c2e_ip1')
                V_ind += numpy.einsum('aijg,nga->nij', j3c, -ind_chunk)
            V_ind = V_ind + V_ind.transpose(0, 2, 1)
        else:
            for i_dm in range(n_dm):
                e_tot.append(self.cppe_state.total_energy + e_ecp)
                e_pol.append(0.0)

        if not elec_only:
            vmat = self.V_es + V_ind
            if self.do_ecp:
                vmat += self.V_ecp
            e = numpy.array(e_tot)
        else:
            vmat = V_ind
            e = numpy.array(e_pol)

        if is_single_dm:
            e = e[0]
            vmat = vmat[0]
        return e, vmat

    def _compute_multipole_potential_integrals(self, all_sites, all_orders, all_moments, n_chunks=1):
        all_orders = numpy.asarray(all_orders)
        if numpy.any(all_orders > 2):
            raise NotImplementedError("""Multipole potential integrals not
                                      implemented for order > 2.""")

        op = 0
        for p0, p1 in lib.prange_split(all_sites.size, n_chunks):
            sites = all_sites[p0:p1]
            orders = all_orders[p0:p1]
            moments = all_moments[p0:p1]

            # order 0
            fakemol = gto.fakemol_for_charges(sites)
            integral0 = df.incore.aux_e2(self.mol, fakemol, intor='int3c2e')
            moments_0 = numpy.array([m[0] for m in moments])
            op += numpy.einsum('ijg,ga->ij', integral0, moments_0 * cppe.prefactors(0))

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
                # moments_2 is the lower triangle of
                # [[XX, XY, XZ], [YX, YY, YZ], [ZX, ZY, ZZ]] i.e.
                # XX, XY, XZ, YY, YZ, ZZ = 0,1,2,4,5,8
                # symmetrize it to the upper triangle part
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
