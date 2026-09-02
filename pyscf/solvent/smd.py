# Copyright 2023 The GPU4PySCF Authors. All Rights Reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

'''
SMD solvent model, copied from GPU4PYSCF with modification for CPU
'''

import numpy as np
from pyscf import lib, gto
from pyscf.data import radii
from pyscf.dft.gen_grid import LEBEDEV_ORDER
from pyscf.solvent import pcm
from pyscf.solvent import _solvent_data
from pyscf.solvent._attach_solvent import _for_scf
from pyscf.lib import logger

@lib.with_doc(_for_scf.__doc__)
def smd_for_scf(mf, solvent_obj=None, dm=None):
    if isinstance(solvent_obj, str):
        # Allows the solvent to be specified by its name, e.g. mf.SMD('toluene')
        solvent_obj = SMD(mf.mol, solvent_obj)
    if solvent_obj is None:
        solvent_obj = SMD(mf.mol)
    return _for_scf(mf, solvent_obj, dm)

# Inject PCM to SCF, TODO: add it to other methods later
from pyscf import scf
scf.hf.RHF.SMD = smd_for_scf
scf.uhf.UHF.SMD = smd_for_scf
hartree2kcal = 627.509451

# The solvent database was moved to the _solvent_data module. It is imported
# here for backward compatibility.
solvent_db = _solvent_data.solvent_db

def smd_radii(alpha):
    '''
    eq. (16)
    use smd radii if defined
    use Bondi radii if defined
    use 2.0 otherwise
    '''
    radii_table = radii.VDW.copy() * radii.BOHR
    radii_table[1] = 1.20
    radii_table[6] = 1.85
    radii_table[7] = 1.89
    if alpha >= 0.43:
        r = 1.52
    else:
        r = 1.52 + 1.8 * (0.43 - alpha)
    radii_table[8] = r
    radii_table[9] = 1.73
    radii_table[14] = 2.47
    radii_table[15] = 2.12
    radii_table[16] = 2.49
    radii_table[17] = 2.38
    #radii_table[35] = 3.06 # original SMD
    # following value from SMD18
    # https://chemistry-europe.onlinelibrary.wiley.com/doi/10.1002/chem.201803652
    radii_table[35] = 2.60
    radii_table[53] = 2.74
    return radii_table/radii.BOHR

import ctypes
from pyscf.lib import load_library
try:
    libsolvent = load_library('libsolvent')
except (IOError, NameError):
    libsolvent = None

def get_cds_legacy(smdobj):
    if libsolvent is None:
        raise RuntimeError(
            'SMD module is not available. '
            'You can compile this module with cmake option "-DENABLE_SMD=ON"')

    mol = smdobj.mol
    natm = mol.natm
    solvent_descriptors = smdobj.solvent_descriptors or solvent_db[smdobj.solvent]
    soln, _, sola, solb, solg, _, solc, solh = solvent_descriptors
    #symbols = [mol.atom_s(ia) for ia in range(mol.natm)]
    charges = np.asarray(mol.atom_charges(), dtype=np.int32, order='F')
    coords = np.asarray(mol.atom_coords(unit='B'), dtype=np.float64, order='C')
    icds = 1 if smdobj.solvent.upper() == 'WATER' else 2
    dcds = np.empty([natm,3])
    mnsol_interface =  libsolvent.mnsol_interface_

    double_ndptr = np.ctypeslib.ndpointer(dtype=np.float64)
    int_ndptr = np.ctypeslib.ndpointer(dtype=np.int32)
    double_ptr = ctypes.POINTER(ctypes.c_double)
    int_ptr = ctypes.POINTER(ctypes.c_int)

    mnsol_interface.argtypes = [
        double_ndptr, int_ndptr,
        int_ptr,
        double_ptr, double_ptr, double_ptr, double_ptr, double_ptr, double_ptr,
        int_ptr,
        double_ptr, double_ptr, double_ndptr]
    natm = ctypes.byref(ctypes.c_int(natm))
    icds = ctypes.byref(ctypes.c_int(icds))
    soln = ctypes.byref(ctypes.c_double(soln))
    sola = ctypes.byref(ctypes.c_double(sola))
    solb = ctypes.byref(ctypes.c_double(solb))
    solg = ctypes.byref(ctypes.c_double(solg))
    solc = ctypes.byref(ctypes.c_double(solc))
    solh = ctypes.byref(ctypes.c_double(solh))
    gcds = ctypes.c_double()
    areacds = ctypes.c_double()

    mnsol_interface(coords, charges,
                    natm,
                    sola, solb, solc, solg, solh, soln,
                    icds,
                    ctypes.byref(gcds), ctypes.byref(areacds), dcds)
    return gcds.value / hartree2kcal, dcds

# Note: in various places, SMD instance is not explictly tested. It is checked
# by the statement "isinstance(solvent, PCM)"
class SMD(pcm.PCM):
    '''
    SMD Solvent Model

    Attributes:
    ----------
    method : str
        No effects. It is set to 'SMD' as a placeholder

    vdw_scale : float
        A scaling factor for van der Waals radii. Default is 1.0.

    r_probe : float
        An additional radius (in Angstrom) added to the van der Waals radii.
        Default is 0.4 Angstrom.

    radii_table : dict
        Custom van der Waals radii for each element. By default, scaled van der Waals radii
        from `vdw_scale` and `r_probe` are used.

    sasa_ng : int
        The number of quadrature grids used for calculating the Solvent
        Accessible Surface Area (SASA). Default is 590.

    solvent : str
        The name of the solvent, which is used to determine the dielectric constant and other
        relevant parameters. Supported solvents can be accessed via the variable
        `pyscf.solvent.smd.solvent_db`. Their names are matched case-insensitively,
        ignoring spaces and hyphens, and common abbreviations (such as "DMSO") are
        recognized as well.

    frozen : bool
        Whether to freeze the potential produced by the solvent during SCF iterations or
        other convergence processes. When frozen=True is set, the solvent is
        assumed to respond slowly, while the electron density relaxes quickly.
        Default is False.

    max_cycle : int
        The maximum number of iterations to relax the solvent.

    conv_tol : float
        The convergence tolerance for total energy during solvent relaxation.

    equilibrium_solvation : bool
        Affects TDDFT and other excited state computations. Controls whether the solvent
        relaxes rapidly with respect to the electron density of the excited state.
        For vertical excitations, it is recommended to set this to False, as the solvent
        typically does not fully relax. The non-equilibrium solvation is then applied with
        the optical dielectric constant `eps_optical`. Default is False.

    eps_optical : float
        The optical (high-frequency) dielectric constant of the solvent, i.e. the square of
        its refractive index. It is only used by the non-equilibrium solvation of excited
        states (see `equilibrium_solvation`). If left unset, it is derived from the
        refractive index of `solvent` in the SMD solvent database. Default is None.

    state_id : int
        Specifies the target state in excited state calculations.
        `state_id=0` corresponds to the ground state, while `state_id=1` corresponds
        to the first excited state. Default is 0.

    Saved Results:
    --------------
    e_cds : float
        Cavitation, Dispersion and Solvent energy

    Intermediate Attributes:
    ------------------------
    These attributes are generated during calculations and should not be modified.
    Additionally, they may not be compatible between GPU and CPU implementations.

    - surface
    - _intermediates
    - v_grids_n
    - solvent_descriptors
    '''

    _keys = {
        'method', 'vdw_scale', 'sasa_ng',
        'mol', 'radii_table', 'lebedev_order', 'lmax', 'eta',
        'solvent', 'eps', 'eps_optical', 'max_cycle', 'conv_tol', 'state_id', 'frozen',
        'frozen_dm0_for_finite_difference_without_response',
        'equilibrium_solvation', 'solvent_descriptors',
        'surface', 'intopt', 'e', 'v', 'v_grids_n', 'e_cds',
        'surface_discretization_method',
    }

    def __init__(self, mol, solvent=''):
        self.mol = mol
        self.stdout = mol.stdout
        self.verbose = mol.verbose
        self.max_memory = mol.max_memory

        self.vdw_scale = 1.0
        self.sasa_ng = 590 # quadrature grids for calculating SASA
        self.method = 'SMD'
        self.solvent = solvent
        self.solvent_descriptors = None
        self.radii_table = None
        self.eps = None
        self.eps_optical = None
        self.surface_discretization_method = "SWIG"
        self.max_cycle = 20
        self.conv_tol = 1e-7
        self.state_id = 0
        self.frozen = False
        self.frozen_dm0_for_finite_difference_without_response = None
        self.equilibrium_solvation = False

        # Following are intermediates
        self.surface = {}
        self._intermediates = {}
        self.e = None
        self.v = None
        self.v_grids_n = None
        self.e_cds = None

    def _set_solvent(self, solvent):
        '''Unlike the PCM models, the SMD model reads all solvent parameters
        from the database at runtime (see .build). Only the name of the solvent
        is recorded here.
        '''
        name = _solvent_data.resolve_solvent_name(solvent)
        if getattr(self, '_intermediates', None):
            # The solvent was modified. The cached intermediates are outdated.
            self.reset()
        return name

    def build(self, ng=None):
        solvent_descriptors = self.solvent_descriptors or solvent_db[self.solvent]
        if self.radii_table is None:
            radii_table = smd_radii(solvent_descriptors[2])
        else:
            radii_table = self.radii_table
        logger.debug(self, 'radii_table %s', radii_table*radii.BOHR)
        mol = self.mol
        if ng is None:
            ng = self.sasa_ng

        self.surface = pcm.gen_surface(mol, rad=radii_table, ng=ng)
        self._intermediates = {}
        F, A = pcm.get_F_A(self.surface)
        D, S = pcm.get_D_S(self.surface, with_S=True, with_D=True)

        epsilon = self.eps or solvent_descriptors[5]
        f_epsilon = (epsilon - 1.0)/(epsilon + 1.0) if epsilon != float('inf') else 1.
        DA = D*A
        DAS = np.dot(DA, S)
        K = S - f_epsilon/(2.0*np.pi) * DAS
        R = -f_epsilon * (np.eye(K.shape[0]) - 1.0/(2.0*np.pi)*DA)
        intermediates = {
            'S': S,
            'D': D,
            'A': A,
            'K': K,
            'R': R,
            'f_epsilon': f_epsilon
        }
        self._intermediates.update(intermediates)

        charge_exp  = self.surface['charge_exp']
        grid_coords = self.surface['grid_coords']
        atom_coords = mol.atom_coords(unit='B')
        atom_charges = mol.atom_charges()

        int2c2e = mol._add_suffix('int2c2e')
        fakemol = gto.fakemol_for_charges(grid_coords, expnt=charge_exp**2)
        fakemol_nuc = gto.fakemol_for_charges(atom_coords)
        v_ng = gto.mole.intor_cross(int2c2e, fakemol_nuc, fakemol)
        self.v_grids_n = np.dot(atom_charges, v_ng)
        return self

    def get_eps_optical(self):
        '''The optical (high-frequency) dielectric constant of the solvent.

        Unless .eps_optical is set explicitly, it is evaluated as n**2 from the
        refractive index n of the solvent (see .sol_desc).
        '''
        if self.eps_optical is not None:
            return self.eps_optical
        n = (self.solvent_descriptors or solvent_db[self.solvent])[0]
        if not n:
            # Neither .solvent nor .sol_desc was specified. n is unknown.
            return pcm.PCM.get_eps_optical(self)
        return n**2

    @property
    def sol_desc(self):
        return self.solvent_descriptors

    @sol_desc.setter
    def sol_desc(self, values):
        '''
        format of sol desc
        [n, n25, alpha, beta, gamma, epsilon, phi, psi]
        '''
        assert len(values) == 8
        self.solvent_descriptors = values

    @property
    def lebedev_order(self):
        for key, val in LEBEDEV_ORDER.items():
            if val == self.sasa_ng:
                return key
        raise RuntimeError(f'sasa_ng={self.sasa_ng} does not have a corresponding lebedev_order')
    @lebedev_order.setter
    def lebedev_order(self, x):
        self.sasa_ng = LEBEDEV_ORDER[x]

    def dump_flags(self, verbose=None):
        solvent_descriptors = self.solvent_descriptors or solvent_db[self.solvent]
        n, _, alpha, beta, gamma, eps, phi, psi = solvent_descriptors
        logger.info(self, '******** %s ********', self.__class__)
        logger.info(self, 'sasa_ng = %s', self.sasa_ng)
        logger.info(self, 'eps = %s'   , self.eps or eps)
        logger.info(self, 'eps_optical = %s', self.eps_optical)
        logger.info(self, 'frozen = %s', self.frozen)
        logger.info(self, '---------- SMD solvent descriptors -------')
        logger.info(self, f'n     = {n}')
        logger.info(self, f'alpha = {alpha}')
        logger.info(self, f'beta  = {beta}')
        logger.info(self, f'gamma = {gamma}')
        logger.info(self, f'phi   = {phi}')
        logger.info(self, f'psi   = {psi}')
        logger.info(self, '--------------------- end ----------------')
        logger.info(self, 'equilibrium_solvation = %s', self.equilibrium_solvation)
        return self

    def get_cds(self):
        if self.e_cds is None:
            self.e_cds = get_cds_legacy(self)[0]
        return self.e_cds

    def nuc_grad_method(self, grad_method):
        raise DeprecationWarning('Use the make_grad_object function from '
                                 'pyscf.solvent.grad.smd instead.')

    def grad(self, dm, verbose=None):
        '''Computes the Jacobian for the energy associated with the solvent,
        including the derivatives of the solvent itsself and the interactions
        between the solvent and the charge density of the solute.
        '''
        from pyscf.solvent.grad.pcm import grad_qv, grad_nuc
        from pyscf.solvent.grad.smd import grad_solver, get_cds
        de_solvent = grad_qv(self, dm)
        de_solvent+= grad_solver(self, dm)
        de_solvent+= grad_nuc(self, dm)
        #de_cds     = get_cds(self.base.with_solvent)
        de_cds     = get_cds_legacy(self)[1]
        logger.info(self, 'Cavitation, Dispersion and Solvent structure contribution %s', de_cds)
        return de_solvent + de_cds

    def Hessian(self, hess_method):
        raise DeprecationWarning('Use the make_hess_object function from '
                                 'pyscf.solvent.hessian.smd instead.')

    def hess(self, dm):
        from pyscf.solvent.hessian.pcm import (
            analytical_hess_nuc, analytical_hess_qv, analytical_hess_solver)
        from pyscf.solvent.hessian.smd import get_cds
        de_solvent  =    analytical_hess_nuc(self, dm, verbose=self.verbose)
        de_solvent +=     analytical_hess_qv(self, dm, verbose=self.verbose)
        de_solvent += analytical_hess_solver(self, dm, verbose=self.verbose)
        de_cds = get_cds(self)
        logger.info(self, 'Cavitation, Dispersion and Solvent structure contribution %s', de_cds)
        return de_solvent + de_cds

    def reset(self, mol=None):
        super().reset(mol)
        self.e_cds = None
        return self
