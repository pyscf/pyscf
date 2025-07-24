#!/usr/bin/env python
# Copyright 2014-2022 The PySCF Developers. All Rights Reserved.
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
import numpy as np
from scipy import linalg
from pyscf import lib
from pyscf.mcscf.addons import StateAverageMCSCFSolver
from pyscf.mcscf.addons import StateAverageMixFCISolver
from pyscf.mcscf.df import _DFCASSCF
from pyscf.mcscf import mc1step
from pyscf.fci import direct_spin1
from pyscf import mcpdft
from pyscf import __config__

# API for general multi-state MC-PDFT method object
# In principle, various forms can be implemented: CMS, XMS, etc.

# API cleanup desiderata:
# 1. "sipdft", "state_interaction" -> "mspdft", "multi_state"
# 2. Canonicalize function to quickly generate mo_coeff, ci, mo_occ, mo_energy
#    for different choices of intermediate, reference, final states.
# 3. Probably "_finalize" stuff?
# 4. checkpoint stuff

MAX_CYC_DIABATIZE = getattr(__config__, 'mcpdft_mspdft_max_cyc_diabatize', 50)
CONV_TOL_DIABATIZE = getattr(__config__, 'mcpdft_mspdft_conv_tol_diabatize', 1e-8)
SING_TOL_DIABATIZE = getattr(__config__, 'mcpdft_mspdft_sing_tol_diabatize', 1e-8)
NUDGE_TOL_DIABATIZE = getattr(__config__, 'mcpdft_mspdft_nudge_tol_diabatize', 1e-3)

def make_heff_mcscf (mc, mo_coeff=None, ci=None):
    '''Build Hamiltonian matrix in basis of ci vector

    Args:
        mc : an instance of MCPDFT class

    Kwargs:
        mo_coeff : ndarray of shape (nao, nmo)
            MO coefficients
        ci : ndarray or list of len (nroots)
            CI vectors describing the model space, presumed to be in the
            optimized intermediate-state basis

    Returns:
        heff_mcscf : ndarray of shape (nroots, nroots)
            Effective MC-SCF hamiltonian matrix in the basis of the
            provided CI vectors
    '''
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if ci is None: ci = mc.ci

    h1, h0 = mc.get_h1eff (mo_coeff)
    h2 = mc.get_h2eff (mo_coeff)
    h2eff = direct_spin1.absorb_h1e (h1, h2, mc.ncas, mc.nelecas, 0.5)

    def construct_ham_slice(solver, slice, nelecas):
        ci_irrep = ci[slice]
        if hasattr(solver, "orbsym"):
            solver.orbsym = mc.fcisolver.orbsym

        hc_all_irrep = [solver.contract_2e(h2eff, c, mc.ncas, nelecas) for c in ci_irrep]
        heff_irrep = np.tensordot(ci_irrep, hc_all_irrep, axes=((1, 2), (1, 2)))
        diag_idx = np.diag_indices_from(heff_irrep)
        heff_irrep[diag_idx] += h0
        return heff_irrep

    if not isinstance(mc.fcisolver, StateAverageMixFCISolver):
        return construct_ham_slice(direct_spin1, slice(0, len(ci)), mc.nelecas)

    irrep_slices = []
    start = 0
    for solver in mc.fcisolver.fcisolvers:
        end = start+solver.nroots
        irrep_slices.append(slice(start, end))
        start = end

    return [construct_ham_slice(s, irrep, mc.fcisolver._get_nelec(s, mc.nelecas))
            for s, irrep in zip(mc.fcisolver.fcisolvers, irrep_slices)]


def si_newton (mc, ci=None, objfn=None, max_cyc=None, conv_tol=None,
        sing_tol=None, nudge_tol=None):
    '''Optimize the intermediate states describing the model space of
    an MS-PDFT calculation by maximizing the provided objective function
    using a gradient-ascent algorithm

    Args:
        mc : an instance of MSPDFT class

    Kwargs:
        ci : ndarray or list of len (nroots)
            CI vectors spanning the model space
        objfn : callable
            Takes CI vectors as a kwarg and returns the value, gradient,
            and Hessian of a chosen objective function wrt rotation
            between pairs of CI vectors
        max_cyc : integer
            Maximum number of cycles of the gradient-ascent algorithm
        conv_tol : float
            Maximum value of both gradient and step vectors at
            convergence
        sing_tol : float
            Tolerance for determining when normal coordinate belongs to
            the null space (df = d2f = 0) or when the Hessian is
            singular (df != 0, d2f = 0).
        nudge_tol : float
            Minimum step size along a normal coordinate when the surface
            is locally concave.

    Returns:
        conv : logical
            True if the optimization is converged
        ci : list of len (nroots)
            Optimized CI vectors describing intermediate states
    '''

    if ci is None: ci = mc.ci
    if objfn is None: objfn = mc.diabatizer
    if max_cyc is None: max_cyc = getattr (mc, 'max_cyc_diabatize', MAX_CYC_DIABATIZE)
    if conv_tol is None: conv_tol = getattr (mc, 'conv_tol_diabatize', CONV_TOL_DIABATIZE)
    if sing_tol is None: sing_tol = getattr (mc, 'sing_tol_diabatize', SING_TOL_DIABATIZE)
    if nudge_tol is None: nudge_tol = getattr (mc, 'nudge_tol_diabatize', NUDGE_TOL_DIABATIZE)
    ci = np.array (ci) # copy
    log = lib.logger.new_logger (mc, mc.verbose)
    nroots = mc.fcisolver.nroots
    rows,col = np.tril_indices(nroots,k=-1)
    u = np.eye (nroots)
    t = np.zeros((nroots,nroots))
    conv = False
    hdr = '{} intermediate-state'.format (mc.__class__.__name__)
    f, df, d2f, f_update = objfn (ci=ci)
    for it in range(max_cyc):
        log.info ("****iter {} ***********".format (it))
        log.info ("{} objective function value = {}".format (hdr, f))

        # Analyze Hessian
        d2f, evecs = linalg.eigh (d2f)
        evecs = np.array(evecs)
        df = np.dot (df, evecs)
        d2f_zero = np.abs (d2f) < sing_tol
        df_zero = np.abs (df) < sing_tol
        if np.any (d2f_zero & (~df_zero)):
            log.warn ("{} Hess is singular!".format (hdr))
        idx_null = d2f_zero & df_zero
        df[idx_null] = 0.0
        d2f[idx_null] = -1e-16
        pos_idx = d2f > 0
        neg_def = np.all (d2f < 0)
        log.info ("{} Hessian is negative-definite? {}".format (hdr, neg_def))

        # Analyze gradient
        grad_norm = np.linalg.norm(df)
        log.info ("{} grad norm = %f".format (hdr), grad_norm)
        log.info ("{} grad (normal modes) = {}".format (hdr, df))

        # Take step
        df[pos_idx & (np.abs (df/d2f) < nudge_tol)] = nudge_tol
        Dt = df/np.abs (d2f)
        step_norm = np.linalg.norm (Dt)
        log.info ("{} Hessian eigenvalues: {}".format (hdr, d2f))
        log.info ("{} step vector (normal modes): {}".format (hdr, Dt))
        t[:] = 0
        t[np.tril_indices(t.shape[0], k = -1)] = np.dot (Dt, evecs.T)
        t = t - t.T

        if grad_norm < conv_tol and step_norm < conv_tol and neg_def:
            conv = True
            break

        # I want the states we come from on the rows and the states we
        # are going to on the columns: |f> = |i>.Umat. However, the
        # antihermitian parameterization of a unitary operator always
        # puts it the other way around: |f> = Uop|i>, ~no matter how you
        # choose the generator indices~. So I have to transpose here.
        # Flipping the sign of t does the same thing, but don't get
        # confused: this isn't related to the choice of variables!
        u = np.dot (u, linalg.expm (t).T)
        f, df, d2f = f_update (u)

    try:
        ci = np.tensordot(u.T, ci, 1)
    except ValueError as e:
        print (u.shape, ci.shape)
        raise (e)
    if mc.verbose >= lib.logger.DEBUG:
        fmt_str = ' ' + ' '.join (['{:5.2f}',]*nroots)
        log.debug ("{} final overlap matrix:".format (hdr))
        for row in u: log.debug (fmt_str.format (*row))
    if conv:
        log.note ("{} optimization CONVERGED".format (hdr))
    else:
        log.note ("{} optimization did not converge after {} "
                   "cycles".format (hdr, it))

    return conv, list (ci)


class _MSPDFT (mcpdft.MultiStateMCPDFTSolver):
    '''MS-PDFT

    Extra attributes for MS-PDFT:

        diabatization: string
            The name describing the type of diabatization for I/O.
            Currently, only ``CMS'' is available.
        max_cyc_diabatize : integer
            Maximum cycles of the diabatization iteration. Default is 50.
        conv_tol_diabatize : float
            Convergence threshold of the diabatization algorithm. Default
            is 1e-8.
        sing_tol_diabatize : float
            Numerical tolerance for null state-rotation modes and
            singularities within the diabatization algorithm. Null modes
            (e.g., rotation between E1x and E1y states in a linear
            molecule) are ignored. Singularities (zero Hessian and
            non-zero gradient in the same mode) print a warning. Default
            is 1e-8.
        nudge_tol_diabatize : float
            Minimum step size along modes with positive curvature during
            the diabatization algorithm, so as to push away from saddle
            points and minima. Default is 1e-3.

    Saved results

        e_tot : float
            Weighted-average MS-PDFT final energy
        e_states : ndarray of shape (nroots)
            MS-PDFT final energies of the adiabatic states
        ci : list of length (nroots) of ndarrays
            CI vectors in the optimized diabatic basis. Related to the
            MC-SCF and MS-PDFT adiabat CI vectors by the expansion
            coefficients ``si_mcscf'' and ``si_pdft''. Either set of
            adiabat CI vectors can be obtained quickly via
            ``get_ci_adiabats''
        si : ndarray of shape (nroots, nroots)
            Expansion coefficients for the MS-PDFT adiabats in terms of
            the optimized diabatic states
        si_pdft : ndarray of shape (nroots, nroots)
            Synonym of si
        e_mcscf : ndarray of shape (nroots)
            Energies of the MC-SCF adiabatic states
        si_mcscf : ndarray of shape (nroots, nroots)
            Expansion coefficients for the MC-SCF adiabats in terms of
            the optimized diabatic states
        heff_mcscf : ndarray of shape (nroots, nroots)
            Molecular Hamiltonian in the diabatic basis
        hdiag_pdft : ndarray of shape (nroots)
            MC-PDFT total energies of the optimized diabatic states
    '''

    # Metaclass parent

    def __init__(self, mc, diabatizer, diabatize, diabatization):
        self.__dict__.update (mc.__dict__)
        keys = set (('diabatizer', 'diabatize', 'diabatization',
                     'heff_mcscf', 'hdiag_pdft',
                     'get_heff_offdiag', 'get_heff_pdft',
                     'si', 'si_mcscf', 'si_pdft',
                     'max_cyc_diabatize', 'conv_tol_diabatize',
                     'sing_tol_diabatize', 'nudge_tol_diabatize'))
        self._diabatizer = diabatizer
        self._diabatize = diabatize
        self._e_states = None
        self.max_cyc_diabatize = MAX_CYC_DIABATIZE
        self.conv_tol_diabatize = CONV_TOL_DIABATIZE
        self.sing_tol_diabatize = SING_TOL_DIABATIZE
        self.nudge_tol_diabatize = CONV_TOL_DIABATIZE
        self.diabatization = diabatization
        self.si_mcscf = None
        self.si_pdft = None
        self._keys = set (self.__dict__.keys ()).union (keys)

    @property
    def e_states (self):
        if self._in_mcscf_env:
            return self.fcisolver.e_states
        else:
            return self._e_states
    @e_states.setter
    def e_states (self, x):
        self._e_states = x
    # Unfixed to FCIsolver since MS-PDFT state energies are no longer
    # CI solutions

    @property
    def si (self):
        return self.si_pdft
    @si.setter
    def si (self, x):
        self.si_pdft = x

    def get_heff_offdiag (self):
        '''The off-diagonal elements of the effective Hamiltonian matrix

        = heff_mcscf - np.diag (heff_mcscf.diagonal ())
        = ( 0     H_10^* ... )
          ( H_10  0      ... )
          ( ...   ...    ... )

        Returns:
            heff_offdiag : ndarray of shape (nroots, nroots)
                Contains molecular Hamiltonian elements on the off-diagonals
                and zero on the diagonals
        '''
        idx = np.diag_indices_from (self.heff_mcscf)
        heff_offdiag = self.heff_mcscf.copy ()
        heff_offdiag[idx] = 0.0
        return heff_offdiag

    def get_heff_pdft (self):
        '''The MS-PDFT effective Hamiltonian matrix

        = get_heff_offdiag () + np.diag (hdiag_pdft)
        = ( EPDFT_0  H_10^*   ... )
          ( H_10     EPDFT_1  ... )
          ( ...      ...      ... )

        Returns:
            heff_pdft : ndarray of shape (nroots, nroots)
                Contains molecular Hamiltonian elements on the off-diagonals
                and PDFT energies on the diagonals
        '''
        idx = np.diag_indices_from (self.heff_mcscf)
        heff_pdft = self.heff_mcscf.copy ()
        heff_pdft[idx] = self.hdiag_pdft
        return heff_pdft

    def get_ci_adiabats (self, ci=None, uci='MSPDFT'):
        ''' Get the CI vectors in an alternate basis (usually one of the
            two adiabatic bases: MCSCF or MSPDFT)

            Kwargs:
                ci : list of length nroots
                    Diabatic ci vectors; defaults to self.ci
                uci : 'MSPDFT', 'MCSCF', or square array of length nroots
                    (String indicating) unitary matrix for transforming
                    ci vectors

            Returns:
                ci : list of length nroots
                    CI vectors for adiabats
        '''
        si_dict = {'MCSCF': self.si_mcscf,
                   'MSPDFT': self.si_pdft}
        if isinstance (uci, (str,np.bytes_)):
            if uci.upper () in si_dict:
                uci = si_dict[uci.upper ()]
            else:
                raise RuntimeError ("valid uci : 'MCSCF', 'MSPDFT', or ndarray")
        if ci is None: ci = self.ci
        return list (np.tensordot (uci.T, np.asarray (ci), axes=1))
    get_ci_basis=get_ci_adiabats

    def kernel (self, mo_coeff=None, ci0=None, otxc=None, grids_level=None,
                grids_attr=None, **kwargs):
        self.otfnal.reset (mol=self.mol) # scanner mode safety
        if ci0 is None and isinstance (getattr (self, 'ci', None), list):
            ci0 = [c.copy () for c in self.ci]
        self.optimize_mcscf_(mo_coeff=mo_coeff, ci0=ci0)
        diab_conv, self.ci = self.diabatize (ci=self.ci, ci0=ci0, **kwargs)
        self.converged = self.converged and diab_conv
        self.heff_mcscf = self.make_heff_mcscf ()
        e_mcscf, self.si_mcscf = self._eig_si (self.heff_mcscf)
        if abs (linalg.norm (self.e_mcscf-e_mcscf)) > 1e-9:
            raise RuntimeError (("Sanity fault: e_mcscf ({}) != "
                                "self.e_mcscf ({})").format (e_mcscf,
                                self.e_mcscf))
        self.hdiag_pdft = self.compute_pdft_energy_(
            otxc=otxc, grids_level=grids_level, grids_attr=grids_attr)[-1]
        self.e_states, self.si_pdft = self._eig_si (self.get_heff_pdft ())
        self.e_tot = np.dot (self.e_states, self.weights)
        self._log_diabats ()
        self._log_adiabats ()
        return (self.e_tot, self.e_ot, self.e_mcscf, self.e_cas, self.ci,
            self.mo_coeff, self.mo_energy)

    def optimize_mcscf_(self, mo_coeff=None, ci0=None, **kwargs):
        # Initialize in an adiabatic basis
        if ci0 is not None:
            if mo_coeff is None: mo_coeff = self.mo_coeff
            heff_mcscf = self.make_heff_mcscf (mo_coeff, ci0)
            e, self.si_mcscf = self._eig_si (heff_mcscf)
            ci1 = self.get_ci_adiabats (ci=ci0, uci='MCSCF')
        else:
            ci1 = None
        return super().optimize_mcscf_(mo_coeff=mo_coeff, ci0=ci1, **kwargs)

    # All of the below probably need to be wrapped over solvers in
    # multi-state-mix metaclass

    def diabatize (self, ci=None, ci0=None, **kwargs):
        '''Optimize the ``intermediate'' diabatic states of an MS-PDFT
        calculation in the space defined by the MC-SCF ``reference''
        adiabatic states.

        Kwargs
            ci : list of ndarrays of length nroots
                CI vectors defining the model space, usually from a
                or CASSCF kernel call
            ci0 : list of ndarrays of length nroots
                Initial guess for optimized diabatic CI vectors,
                possibly spanning a slightly different space, usually at
                a different geometry (i.e., during a geometry
                optimization or dynamics trajectory)

        Returns
            conv : logical
                Reports whether the diabatization algorithm converged
                successfully
            ci : list of ndarrays of length = nroots
                CI vectors of the optimized diabatic states
        '''
        if ci is None: ci = self.ci
        if ci0 is not None:
            ovlp = np.tensordot (np.asarray (ci).conj (), np.asarray (ci0),
                                 axes=((1,2),(1,2)))
            u, svals, vh = linalg.svd (ovlp)
            ci = self.get_ci_basis (ci=ci, uci=np.dot (u,vh))
        return self._diabatize (self, ci=ci, **kwargs)

    def diabatizer (self, mo_coeff=None, ci=None):
        '''Computes the value, gradient vector, and Hessian matrix with
        respect to pairwise state rotations of the objective function
        maximized by the optimized diabatic (``intermediate'') states.

        Kwargs
            mo_coeff : ndarray of shape (nao,nmo)
                MO coefficients
            ci : list of ndarrays of length nroots
                CI vectors

        Returns:
            f : float
                Objective function value for states
            df : ndarray of shape (npairs = nroots*(nroots-1)/2)
                Gradient vector of objective function with respect to
                pairwise rotations between states
            d2f : ndarray of shape (npairs,npairs)
                Hessian matrix of objective function with respect to
                pairwise rotations between states
            f_update : callable
                Takes a unitary matrix and returns f, df, and d2f as
                above. Some kinds of MS-PDFT can be sped up using
                intermediates. If so, the _diabatizer function can
                provide f_update. Otherwise, it defaults to running
                _diabatizer itself on a rotated set of CI vectors.
        '''
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if ci is None: ci = self.ci
        rets = self._diabatizer (self, mo_coeff=mo_coeff, ci=ci)
        f, df, d2f = rets[:3]
        if len (rets) > 3 and callable (rets[3]):
            f_update = rets[3]
        else:
            def f_update (u=1):
                ci1 = self.get_ci_basis(ci=ci, uci=u)
                f1, df1, d2f1 = self._diabatizer (
                    self, mo_coeff=mo_coeff, ci=ci1)
                return f1, df1, d2f1
        return f, df, d2f, f_update

    def _eig_si (self, heff):
        return linalg.eigh (heff)

    make_heff_mcscf = make_heff_mcscf

    def _log_diabats (self):
        # Information about the intermediate states
        hdiag_mcscf = self.heff_mcscf.diagonal ()
        hdiag_pdft = self.hdiag_pdft
        nroots = len (hdiag_pdft)
        log = lib.logger.new_logger (self, self.verbose)
        f, df, d2f = self.diabatizer ()[:3]
        hdr = '{} diabatic (intermediate)'.format (self.__class__.__name__)
        log.note ('%s objective function  value = %.15g |grad| = %.7g', hdr, f, linalg.norm (df))
        log.note ('%s average energy  EPDFT = %.15g  EMCSCF = %.15g', hdr,
                  np.dot (self.weights, hdiag_pdft), np.dot (self.weights, hdiag_mcscf))
        log.note ('%s states:', hdr)
        if getattr (self.fcisolver, 'spin_square', None):
            ss = self.fcisolver.states_spin_square (self.ci, self.ncas,
                                                    self.nelecas)[0]
            for i in range (nroots):
                log.note ('  State %d  EPDFT = %.15g  EMCSCF = %.15g'
                    '  S^2 = %.7f', i, hdiag_pdft[i],
                    hdiag_mcscf[i], ss[i])
        else:
            for i in range (nroots):
                log.note ('  State %d  EPDFT = %.15g  EMCSCF = '
                    '%.15g', i, hdiag_pdft[i], hdiag_mcscf[i])
        log.info ('MS-PDFT effective Hamiltonian matrix in diabatic basis:')
        fmt_str = ' '.join (['{:9.5f}',]*nroots)
        for row in self.get_heff_pdft(): log.info (fmt_str.format (*row))
        log.info ('Diabatic states (columns) in terms of reference states '
            '(rows):')
        for row in self.si_mcscf.T: log.info (fmt_str.format (*row))

    def _log_adiabats (self):
        # Information about the final states
        log = lib.logger.new_logger (self, self.verbose)
        nroots = len (self.e_states)
        log.note ('%s adiabatic (final) states:', self.__class__.__name__)
        if getattr (self.fcisolver, 'spin_square', None):
            ci = np.tensordot (self.si.T, np.asarray (self.ci), axes=1)
            ss = self.fcisolver.states_spin_square (ci, self.ncas,
                                                    self.nelecas)[0]
            for i in range (nroots):
                log.note ('  State %d weight %g  EMSPDFT = %.15g  S^2 = %.7f',
                          i, self.weights[i], self.e_states[i], ss[i])
        else:
            for i in range (nroots):
                log.note ('  State %d weight %g  EMSPDFT = %.15g', i,
                          self.weights[i], self.e_states[i])

    def nuc_grad_method (self):
        if not isinstance (self, mc1step.CASSCF):
            raise NotImplementedError ("CASCI-based PDFT nuclear gradients")
        elif getattr (self, 'frozen', None) is not None:
            raise NotImplementedError ("PDFT nuclear gradients with frozen orbitals")
        elif isinstance (self, _DFCASSCF):
            from pyscf.df.grad.mspdft import Gradients
        else:
            from pyscf.grad.mspdft import Gradients
        return Gradients (self)

    def nac_method(self):
        if not isinstance(self, mc1step.CASSCF):
            raise NotImplementedError("CASCI-based PDFT NACs")
        elif getattr(self, 'frozen', None) is not None:
            raise NotImplementedError("PDFT NACs with frozen orbitals")
        elif isinstance(self, _DFCASSCF):
            raise NotImplementedError("PDFT NACs with density fitting")
        else:
            from pyscf.nac.mspdft import NonAdiabaticCouplings

        return NonAdiabaticCouplings(self)

    def dip_moment (self, unit='Debye', origin='Coord_Center', state=None):
        if not isinstance (self, mc1step.CASSCF):
            raise NotImplementedError ("CASCI-based PDFT dipole moments")
        elif getattr (self, 'frozen', None) is not None:
            raise NotImplementedError ("PDFT dipole moments with frozen orbitals")
        elif isinstance (self, _DFCASSCF):
            raise NotImplementedError ("PDFT dipole moments with density-fitting ERIs")
        from pyscf.prop.dip_moment.mspdft import ElectricDipole
        if not lib.isinteger (state):
            raise RuntimeError ('Permanent dipole requires a single state')
        dip_obj =  ElectricDipole(self)
        mol_dipole = dip_obj.kernel (state=state, unit=unit, origin=origin)
        return mol_dipole

    def trans_moment (self, unit='Debye', origin='Coord_Center', state=None):
        if not isinstance (self, mc1step.CASSCF):
            raise NotImplementedError ("CASCI-based PDFT dipole moments")
        elif getattr (self, 'frozen', None) is not None:
            raise NotImplementedError ("PDFT dipole moments with frozen orbitals")
        elif isinstance (self, _DFCASSCF):
            raise NotImplementedError ("PDFT dipole moments with density-fitting ERIs")
        from pyscf.prop.trans_dip_moment.mspdft import TransitionDipole
        if not hasattr(state, '__len__') or len(state) !=2:
            raise RuntimeError ('Transition dipole requires two states')
        tran_dip_obj = TransitionDipole(self)
        mol_trans_dipole = tran_dip_obj.kernel (state=state, unit=unit, origin=origin)
        return mol_trans_dipole

def get_diabfns (obj):
    '''Interpret the name of the MS-PDFT method as a pair of functions
    which optimize the intermediate states and calculate the power
    series in the corresponding objective function to second order.

    Args:
        obj : string
            Specify particular MS-PDFT method. Currently, only "CMS" is
            supported. Not case-sensitive.

    Returns:
        diabatizer : callable
            Takes model-space CI vectors in a trial intermediate-state
            basis and returns the value and first and second derivatives
            of the objective function specified by obj
        diabatize : callable
            Takes model-space CI vectors and returns CI vectors in the
            optimized intermediate-state basis
    '''

    if obj.upper () == 'CMS':
        from pyscf.mcpdft.cmspdft import e_coul as diabatizer
        diabatize = si_newton

    elif obj.upper() == "XMS":
        from pyscf.mcpdft.xmspdft import safock_energy as diabatizer
        from pyscf.mcpdft.xmspdft import solve_safock as diabatize

    else:
        raise RuntimeError ('MS-PDFT type not supported')

    return diabatizer, diabatize

def multi_state (mc, weights=(0.5,0.5), diabatization='CMS', **kwargs):
    ''' Build multi-state MC-PDFT method object

    Args:
        mc : instance of class _PDFT

    Kwargs:
        weights : sequence of floats
        diabatization : objective-function type
            Currently supports only 'cms'

    Returns:
        si : instance of class _MSPDFT
    '''

    if isinstance (mc, mcpdft.MultiStateMCPDFTSolver):
        raise RuntimeError ('already a multi-state PDFT solver')
    if isinstance (mc.fcisolver, StateAverageMixFCISolver):
        raise RuntimeError ('state-average mix type')
    if not isinstance (mc, StateAverageMCSCFSolver):
        base_name = mc.__class__.__name__
        mc = mc.state_average (weights=weights, **kwargs)
    else:
        base_name = mc.__class__.__bases__[0].__name__
    mcbase_class = mc.__class__
    diabatizer, diabatize = get_diabfns (diabatization)

    class MSPDFT (_MSPDFT, mcbase_class):
        pass
    MSPDFT.__name__ = diabatization.upper () + base_name
    return MSPDFT (mc, diabatizer, diabatize, diabatization)


if __name__ == '__main__':
    # This ^ is a convenient way to debug code that you are working on. The
    # code in this block will only execute if you run this python script as the
    # input directly: "python mspdft.py".

    from pyscf import scf, gto
    from pyscf.tools import molden # My version is better for MC-SCF
    from pyscf.fci import csf_solver
    xyz = '''O  0.00000000   0.08111156   0.00000000
             H  0.78620605   0.66349738   0.00000000
             H -0.78620605   0.66349738   0.00000000'''
    mol = gto.M (atom=xyz, basis='sto-3g', symmetry=False, output='mspdft.log',
        verbose=lib.logger.DEBUG)
    mf = scf.RHF (mol).run ()
    mc = mcpdft.CASSCF (mf, 'tPBE', 4, 4).set (fcisolver = csf_solver (mol, 1))
    mc = mc.multi_state ([1.0/3,]*3, 'cms').run ()


