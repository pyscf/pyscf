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
import copy
from scipy import linalg
from pyscf import lib, dft
from pyscf.lib import logger
from pyscf.dft import libxc
from pyscf.dft.gen_grid import Grids
from pyscf.dft.numint import _NumInt, NumInt
from pyscf.mcpdft import pdft_veff, tfnal_derivs, _libxc, _dms, pdft_feff, pdft_eff
from pyscf.mcpdft.otpd import get_ontop_pair_density
from pyscf import __config__

FT_R0 = getattr(__config__, 'mcpdft_otfnal_ftransfnal_R0', 0.9)
FT_R1 = getattr(__config__, 'mcpdft_otfnal_ftransfnal_R1', 1.15)
FT_A = getattr(__config__, 'mcpdft_otfnal_ftransfnal_A', -475.60656009)
FT_B = getattr(__config__, 'mcpdft_otfnal_ftransfnal_B', -379.47331922)
FT_C = getattr(__config__, 'mcpdft_otfnal_ftransfnal_C', -85.38149682)

OT_ALIAS = {
    'MC23': 'tMC23',
    'MC25': 'tMC25',
}
OT_HYB_ALIAS = {'PBE0' : '0.25*HF + 0.75*PBE, 0.25*HF + 0.75*PBE',
                }

REG_OT_FUNCTIONALS={}

# ALIAS for the preset on-top functional
OT_PRESET={
    # Reparametrized-M06L: rep-M06L
    # XC_ID_MGGA_C_M06_L = 233
    # XC_ID_MGGA_X_M06_L = 203

    # MC23 = { '0.2952*HF + (1-0.2952)*rep-M06L, 0.2952*HF + (1-0.2952)*rep-M06L'}}
    # J. J. Bao, D. Zhang, S. Zhang, L. Gagliardi, and D. G. Truhlar
    # Proc. Natl. Acad. Sci. U.S.A. 122, e2419413121/1-8 (2025).
    # https://doi.org/10.1073/pnas.2419413121
    'MC23':{
        'xc_base':'M06L',
        'ext_params':{203: np.array([3.352197, 6.332929e-01, -9.469553e-01, 2.030835e-01,
                                     2.503819, 8.085354e-01, -3.619144, -5.572321e-01,
                                     -4.506606, 9.614774e-01, 6.977048, -1.309337, -2.426371,
                                     -7.896540e-03, 1.364510e-02, -1.714252e-06, -4.698672e-05, 0.0]),
                        233: np.array([0.06, 0.0031, 0.00515088, 0.00304966, 2.427648, 3.707473,
                                       -7.943377, -2.521466, 2.658691, 2.932276, -8.832841e-01,
                                       -1.895247, -2.899644, -5.068570e-01, -2.712838, 9.416102e-02,
                                       -3.485860e-03, -5.811240e-04, 6.668814e-04, 0.0, 2.669169e-01,
                                       -7.563289e-02, 7.036292e-02, 3.493904e-04, 6.360837e-04, 0.0, 1e-10])},
        'hyb':(0.2952,0.2952,0),
        'facs':(0.7048,0.7048)},

    # MC25 = { '0.28*HF + (1-0.28)*rep-M06L, 0.28*HF + (1-0.28)*rep-M06L'}}
    # D. Zhang, Y. Kim, M. Hennefarth, L. Gagliardi, and D. G. Truhlar
    # Under review
    'MC25': {
        'xc_base': 'M06L',
        'ext_params': {
            203: np.array([3.465030e+00, 5.375447e-01, -7.191629e-01, -9.915646e-01,
                           2.229138e+00, 5.404209e+00, -4.004898e+00, -5.983860e+00,
                           -2.086931e+00, -8.878196e-01, 4.888665e+00, 2.868958e+00,
                           -2.499939e+00, -9.852771e-03, 8.596984e-03, -1.220706e-05,
                           -1.336275e-05, 0.0]),
            233: np.array([0.06, 0.0031, 0.00515088, 0.00304966, 1.489435e+00,
                           2.942442e+00, -6.297330e+00, -2.062505e+00, 1.634904e+00,
                           1.608843e+00, -1.233955e+00, -1.964674e+00, -2.471985e+00,
                           -5.392796e-01, -1.509794e+00, 2.794569e-02, 1.061909e-01,
                           5.095118e-04, -2.927055e-03, 0.0, 9.690385e-01, -4.546714e-02,
                           4.151718e-02, 1.789189e-04, 1.024388e-03, 0.0, 1e-10])
        },
        'hyb': (0.28, 0.28, 0),
        'facs': (0.72, 0.72)
    },
}

def register_otfnal(xc_code, preset):
    '''
    This function registers the new on-top functional if it hasn't been
    registered previously.
    Args:
        xc_code: str
            The name of the on-top functional to be registered.
        preset: dict
            The dictionary containing the information about the on-top functional
            to be registered.
            xc_base: str
                The name of the underylying KS-functional in the libxc library.
            ext_params: dict, with LibXC exchange and correlation functional integer ID as key, and
                an array-like object containing the functional parameters as value.
            hyb: tuple
                The hybrid functional parameters.
            facs: tuple
                The mixing factors.
            kwargs: dict
                The additional keyword arguments.
    '''
    libxc_register_code = xc_code.upper ()
    libxc_base_code = preset['xc_base']
    ext_params = preset['ext_params']
    hyb = preset.get('hyb', None)
    facs = preset.get('facs', None)
    libxc.register_custom_functional_(libxc_register_code, libxc_base_code,
                                      ext_params=ext_params, hyb=hyb, facs=facs)
    REG_OT_FUNCTIONALS[xc_code.upper()] = {'hyb_x':preset.get('hyb',[0])[0],
                                           'hyb_c':preset.get('hyb',[0])[0]}

def unregister_otfnal(xc_code):
    '''
    This function unregisters the on-top functional if it has been registered
    previously.
    Args:
        xc_code: str
            The name of the on-top functional to be unregistered.
    '''
    try:
        if xc_code.upper() in REG_OT_FUNCTIONALS:
            libxc_unregister_code = xc_code.upper()
            libxc.unregister_custom_functional_(libxc_unregister_code)
            del REG_OT_FUNCTIONALS[xc_code.upper()]

    except Exception as e:
        raise RuntimeError(f"Failed to unregister functional '{xc_code}': {e}") from e

def _get_registered_ot_functional(xc_code, mol):
    '''
    This function returns the on-top functional if it has been registered
    previously.
    Args:
        xc_code: str
            The name of the on-top functional to be registered.
    '''
    if (xc_code.upper() not in REG_OT_FUNCTIONALS) and (xc_code.upper() in OT_PRESET):
        preset = OT_PRESET[xc_code.upper()]
        register_otfnal(xc_code, preset)
        logger.info(mol, 'Registered the on-top functional: %s', xc_code)
    return xc_code.upper()

def energy_ot (ot, casdm1s, casdm2, mo_coeff, ncore, max_memory=2000, hermi=1):
    '''Compute the on-top energy - the last term in

    E_MCPDFT = h_pq l_pq + 1/2 v_pqrs l_pq l_rs + E_ot[rho,Pi]

    Args:
        ot : an instance of otfnal class
        casdm1s : ndarray of shape (2, ncas, ncas)
            Contains spin-separated one-body density matrices in an
            active-orbital basis
        casdm2 : ndarray of shape (ncas, ncas, ncas, ncas)
            Contains spin-summed two-body density matrix in an active-
            orbital basis
        mo_coeff : ndarray of shape (nao, nmo)
            Contains molecular orbital coefficients for active-space
            orbitals. Columns ncore through ncore+ncas give the basis
            in which casdm1s and casdm2 are expressed.
        ncore : integer
            Number of doubly occupied inactive "core" orbitals not
            explicitly included in casdm1s and casdm2

    Kwargs:
        max_memory : int or float
            maximum cache size in MB
            default is 2000
        hermi : int
            1 if 1rdms are assumed hermitian, 0 otherwise

    Returns : float
        The MC-PDFT on-top (nonclassical) energy
    '''
    E_ot = 0.0
    ni, xctype = ot._numint, ot.xctype
    if xctype=='HF': return E_ot
    dens_deriv = ot.dens_deriv
    Pi_deriv = ot.Pi_deriv

    nao = mo_coeff.shape[0]
    ncas = casdm2.shape[0]
    cascm2 = _dms.dm2_cumulant (casdm2, casdm1s)
    dm1s = _dms.casdm1s_to_dm1s (ot, casdm1s, mo_coeff=mo_coeff, ncore=ncore,
                                 ncas=ncas)
    mo_cas = mo_coeff[:,ncore:][:,:ncas]

    t0 = (logger.process_clock (), logger.perf_counter ())
    make_rho = tuple (ni._gen_rho_evaluator (ot.mol, dm1s[i,:,:], hermi=hermi, with_lapl=False) for
        i in range(2))
    for ao, mask, weight, _ in ni.block_loop (ot.mol, ot.grids, nao,
            dens_deriv, max_memory):
        rho = np.asarray ([m[0] (0, ao, mask, xctype) for m in make_rho])
        t0 = logger.timer (ot, 'untransformed density', *t0)
        Pi = get_ontop_pair_density (ot, rho, ao, cascm2, mo_cas,
            Pi_deriv, mask)
        t0 = logger.timer (ot, 'on-top pair density calculation', *t0)
        if rho.ndim == 2:
            rho = np.expand_dims (rho, 1)
            Pi = np.expand_dims (Pi, 0)
        E_ot += ot.eval_ot (rho, Pi, dderiv=0, weights=weight)[0].dot (weight)
        t0 = logger.timer (ot, 'on-top energy calculation', *t0)

    return E_ot

class otfnal:
    r''' Parent class of on-top pair-density functional. The main
    callable is ``eval_ot,'' which is comparable to pyscf.dft.libxc
    ``eval_xc.'' A true ``kernel'' method, which would take arbitrary
    1- and 2-RDMs and return the total PDFT energy, awaits design
    decisions on how far I'm willing/able to generalize the otpd
    functions. For instance, in MP2 or CCSD, the 2-RDM spans the
    whole orbital space and it may not be possible to hold it in
    memory. At present, it's all designed around MC-SCF, which is
    why the ``kernel'' function that actually calculates the energy
    is in mcpdft.py instead of here.

    Attributes:
        mol : object of class pyscf.gto.mole
        grids : object of class pyscf.dft.gen_grid.Grids
        eval_ot : function with calling signature shown below
        _numint : object of class pyscf.dft.NumInt
            member functions "hybrid_coeff", "nlc_coeff, "rsh_coeff",
            and "_xc_type" (at least) must be overloaded; see below
        otxc : string
            name of on-top pair-density exchange-correlation functional
    '''

    def __init__ (self, mol):
        self.mol = mol
        self.verbose = mol.verbose
        self.stdout = mol.stdout

    Pi_deriv = 0

    def _init_info (self):
        logger.info (self, 'Building %s functional', self.otxc)
        hyb = self._numint.rsh_and_hybrid_coeff(self.otxc, spin=self.mol.spin)[2]
        if hyb[0] > 0:
            logger.info (self, 'Hybrid functional with %s CASSCF exchange',
                hyb)

    @property
    def xctype (self):
        return self._numint._xc_type (self.otxc)

    @property
    def dens_deriv (self):
        return ['LDA', 'GGA', 'MGGA'].index (self.xctype)

    def eval_ot (self, rho, Pi, dderiv=0, **kwargs):
        r''' Evaluate the on-dop energy and its functional derivatives
             on a grid

        Args:
            rho : ndarray of shape (2,*,ngrids)
                containing spin-density [and derivatives]
            Pi : ndarray with shape (*,ngrids)
                containing on-top pair density [and derivatives]

        Kwargs:
            dderiv : integer
                Order of derivatives to return

        Returns:
            eot : ndarray of shape (ngrids)
                integrand of the on-top exchange-correlation energy
            vot : (array_like (rho), array_like (Pi)) or None
                first functional derivative of Eot wrt (density, pair-
                density) and their derivatives
            fot : ndarray of shape (*,ngrids) or None
                second functional derivative of Eot wrt density, pair-
                density, and derivatives; first dimension is lower-
                triangular matrix elements corresponding to the basis
                (rho, Pi, |drho|^2, drho'.dPi, |dPi|) stopping at Pi (3
                elements) for t-LDA and |drho|^2 (6 elements) for t-GGA.
        '''
        raise NotImplementedError("on-top xc functional not defined")

    energy_ot = energy_ot
    get_eff_1body = pdft_eff.get_eff_1body
    get_eff_2body = pdft_eff.get_eff_2body
    get_eff_2body_kl = pdft_eff.get_eff_2body_kl

    get_veff_1body = pdft_veff.get_veff_1body
    get_veff_2body = pdft_veff.get_veff_2body
    get_veff_2body_kl = pdft_veff.get_veff_2body_kl

    get_feff_1body = pdft_feff.get_feff_1body
    get_feff_2body = pdft_feff.get_feff_2body

    def reset (self, mol=None):
        ''' Discard cached grid data and optionally update the mol '''
        if mol is not None:
            self.mol = mol
        self.grids.reset (mol=mol)

class transfnal (otfnal):
    __doc__ = otfnal.__doc__ + r'''

    ``translated functional'' of Li Manni et al., JCTC 10, 3669 (2014).
    The extra attributes are all callables; see their docstrings for
    more information.

    Args:
        ks : object of :class:`dft.RKS`
            ks.xc is the Kohn-Sham functional being ``translated''

    '''

    transl_prefix='t'

    def __init__ (self, ks, **kwargs):
        otfnal.__init__(self, ks.mol, **kwargs)
        self.otxc = 't' + ks.xc
        self._numint = copy.copy (ks._numint)
        self._numint.libxc = libxc
        self.grids = copy.copy (ks.grids)
        self._numint.hybrid_coeff = t_hybrid_coeff.__get__(self._numint)
        self._numint.nlc_coeff = t_nlc_coeff.__get__(self._numint)
        self._numint.rsh_coeff = t_rsh_coeff.__get__(self._numint)
        self._numint.eval_xc = t_eval_xc.__get__(self._numint)
        self._numint._xc_type = t_xc_type.__get__(self._numint)
        self._init_info ()

    def get_ratio (self, Pi, rho_avg):
        r''' R = Pi / [rho/2]^2 = Pi / rho_avg^2
            An intermediate quantity when computing the translated spin
            densities

            Note this function returns 1 for values and 0 for
            derivatives for every point where the charge density is
            close to zero (i.e., convention: 0/0 = 1)

            Args:
                Pi : ndarray of shape (*,ngrids)
                    Contains on-top pair density on a grid
                rho_avg : ndarray of shape (*,ngrids)
                    Contains the average of the spin-up and spin-down
                    charge densities on a grid, (rho[0]+rho[1])/2

            Returns:
                R : ndarray of shape (*,ngrids)
                    on-top ratio
        '''
        nderiv = min (rho_avg.shape[0], Pi.shape[0])
        ngrids = rho_avg.shape[1]
        assert (Pi.shape[1] == ngrids)
        if nderiv > 4:
            raise NotImplementedError("derivatives above order 1")

        R = np.zeros ((nderiv,ngrids), dtype=Pi.dtype)
        R[0,:] = 1
        R[0, Pi[0] == 0] = 0.0
        idx = rho_avg[0] >= (1e-15 / 2)
        # Chain rule!
        for ideriv in range(nderiv):
            R[ideriv,idx] = Pi[ideriv,idx] / rho_avg[0,idx] / rho_avg[0,idx]

        # Product rule!
        for ideriv in range (1,nderiv):
            R[ideriv,idx] -= (2 * rho_avg[ideriv,idx] * R[0,idx]
                / rho_avg[0,idx])
        return R

    def get_rho_translated (self, Pi, rho, _fn_deriv=0):
        r''' Compute the "translated" alpha and beta densities:
        For the unrestricted case,
        rho = [rho^a, rho^b]
        Here:
            rho^a will have dim of 1,4 or 6 depends on the functional. For MGGA,
            rho^a = [rho_u,grad_xu, grad_yu, grad_zu, laplacian_u, tau_u]
            Similar for rho_b.

        The translation is done as follows:

        rho_t^a = (rho/2) * (1 + zeta)
        rho_t^b = (rho/2) * (1 - zeta)
        rho'_t^a = (rho'/2) * (1 + zeta)
        rho'_t^b = (rho'/2) * (1 - zeta)
        tau_t^a = (tau/2) * (1 + zeta)
        tau_t^b = (tau/2) * (1 - zeta)

        See "get_zeta" for the meaning of "zeta"

        Args:
            Pi : ndarray of shape (*, ngrids)
                containing on-top pair density [and derivatives]
            rho : ndarray of shape (2, *, ngrids)
                containing spin density [and derivatives]

        Kwargs:
            _fn_deriv : integer
                Order of functional derivatives of zeta to compute.
                In "translated" functionals, no functional derivatives
                of zeta are used. This kwarg is used for convenience
                when calling from children classes. It changes the
                return signature and should not normally be touched by
                users.

        Returns:
            rho_t : ndarray of shape (2,*,ngrids)
                Translated spin density (and derivatives) in case of LDA or GGAs
                Translated spin density, derivatives, and kinetic energy density in case of MGGA
        '''

        # For nonzero charge & pair density, set alpha dens = beta dens
        # = 1/2 charge dens
        rho_avg = (rho[0,:,:] + rho[1,:,:]) / 2
        rho_t = rho.copy ()
        rho_t[0] = rho_t[1] = rho_avg

        # For 0 <= ratio < 1 and 0 <= rho, correct spin density using on-top
        # density
        nderiv_R = Pi.shape[0] if _fn_deriv else 1
        R = self.get_ratio (Pi[0:nderiv_R,:], rho_avg[0:nderiv_R,:])
        zeta = self.get_zeta (R, fn_deriv=_fn_deriv)

        # Chain rule!
        w = rho_avg * zeta[0:1]
        rho_t[0] += w
        rho_t[1] -= w

        if _fn_deriv > 0: return rho_t, R, zeta
        return rho_t

    def get_zeta (self, R, fn_deriv=0, _Rmax=1):
        r''' Compute the intermediate zeta used to compute the
        translated spin densities and its functional derivatives

        From the original translation [Li Manni et al., JCTC 10, 3669
        (2014)]:

        zeta = (1-ratio)^(1/2) ; ratio < 1
             = 0               ; otherwise

        Args:
            R : ndarray of shape (*,ngrids)
                Ratio (4Pi/rho^2) and possibly its spatial derivatives
                Only the first row is used in this function

        Kwargs:
            fn_deriv : integer
                order of functional derivative (d^n z / dR^n) to return
                along with the value of zeta
            _Rmax : float
                maximum value of R for which to compute zeta or its
                derivatives; columns of zeta with R[0]>_Rmax are zero.
                This is a hook for the ``fully-translated'' child class
                and should not be touched normally.

        Returns:
            zeta : ndarray of shape (fn_deriv+1, ngrids)
        '''
        if R.ndim == 2: R = R[0]
        ngrids = R.size
        zeta = np.zeros ((fn_deriv+1, ngrids), dtype=R.dtype)
        idx = R < _Rmax
        zeta[0,idx] = np.sqrt (1.0 - R[idx])
        if fn_deriv:
            zeta[1,idx] = -0.5 / zeta[0,idx]
        if fn_deriv > 1: fac = 0.5 / (1.0-R[idx])
        for n in range (1,fn_deriv):
            zeta[n+1,idx] = zeta[n,idx] * (2*n-1) * fac
        return zeta

    def split_x_c (self):
        ''' Get one translated functional for just the exchange and one
            for just the correlation part of the energy.

            Returns:
                xfnal : object of :class:`transfnal`
                    this functional, but only the exchange part
                cfnal : object of :class:`transfnal`
                    this functional, but only the correlation part
        '''
        xc_base = self.otxc[len (self.transl_prefix):]
        x_code, c_code = _libxc.split_x_c_comma (xc_base)
        x_code = self.transl_prefix + x_code + ','
        c_code = self.transl_prefix + ',' + c_code
        xfnal = copy.copy (self)
        xfnal._numint = copy.copy (self._numint)
        xfnal.grids = copy.copy (self.grids)
        xfnal.verbose = self.verbose
        xfnal.stdout = self.stdout
        xfnal.otxc = x_code
        cfnal = copy.copy (self)
        cfnal._numint = copy.copy (self._numint)
        cfnal.grids = copy.copy (self.grids)
        cfnal.verbose = self.verbose
        cfnal.stdout = self.stdout
        cfnal.otxc = c_code
        return xfnal, cfnal

    def jT_op (self, x, rho, Pi):
        r''' Evaluate jTx = (x.j)T where j is the Jacobian of the
        translated densities in terms of the untranslated density and
        pair density

        Args:
            x : ndarray of shape (2,*,ngrids)
                Usually, a functional derivative of the on-top xc energy
                wrt translated densities
            rho : ndarray of shape (2,*,ngrids)
                containing spin-density [and derivatives]
            Pi : ndarray with shape (*,ngrids)
                containing on-top pair density [and derivatives]

        Returns: ndarray of shape (*,ngrids)
            Usually, a functional derivative of the on-top pair density
            exchange-correlation energy wrt to total density and its
            derivatives. The potential must be spin-symmetric in
            pair-density functional theory.
            2 rows for tLDA, 3 rows for tGGA, and 4 rows for meta-GGA
        '''
        # ordering: rho, Pi, |rho'|^2, tau
        ncol = (2, 3, 4)[self.dens_deriv]
        ngrid = rho.shape[-1]
        jTx = np.zeros ((ncol,ngrid), dtype=x[0].dtype)
        rho = rho.sum (0)
        R = self.get_ratio (Pi, rho/2)
        zeta = self.get_zeta (R, fn_deriv=1)
        jTx[:2] = tfnal_derivs._gentLDA_jT_op (x, rho, Pi, R, zeta)
        if self.dens_deriv > 0:
            jTx[:3] += tfnal_derivs._tGGA_jT_op (x, rho, Pi, R, zeta)
        if self.dens_deriv > 1:
            jTx[:4] += tfnal_derivs._tmetaGGA_jT_op(x, rho, Pi, R, zeta)

        return jTx

    def d_jT_op (self, x, rho, Pi):
        r''' Evaluate the x.(nabla j) contribution to the second density
        derivatives of the on-top energy in terms of the untranslated
        density and pair density

        Args:
            x : ndarray of shape (2,*,ngrids)
                Usually, a functional derivative of the on-top xc energy
                wrt translated densities
            rho : ndarray of shape (2,*,ngrids)
                containing spin-density [and derivatives]
            Pi : ndarray with shape (*,ngrids)
                containing on-top pair density [and derivatives]

        Returns: ndarray of shape (*,ngrids)
            second derivative of the translation dotted with x
            3 rows for tLDA and 5 rows for tGGA
        '''
        nrow = 3 + 2*int(self.dens_deriv>0)
        f = np.zeros ((nrow, x[0].shape[-1]), dtype=x[0].dtype)

        rho = rho.sum (0)
        R = self.get_ratio (Pi, rho/2)
        zeta = self.get_zeta (R, fn_deriv=2)

        f[:3] = tfnal_derivs._gentLDA_d_jT_op (x, rho, Pi, R, zeta)
        if self.dens_deriv:
            f[:] += tfnal_derivs._tGGA_d_jT_op (x, rho, Pi, R, zeta)

        if self.verbose >= logger.DEBUG:
            idx = zeta[0] == 0
            logger.debug (self, 'MC-PDFT fot zeta check: %d zeta=0 columns',
                np.count_nonzero (idx))
            if np.count_nonzero (idx):
                for ix, frow in enumerate (f):
                    logger.debug (self, 'MC-PDFT fot zeta check: f[%d] norm '
                        'over zeta=0 columns: %e', ix, linalg.norm (frow[idx]))

        return f

    def eval_ot (self, rho, Pi, dderiv=1, weights=None, _unpack_vot=True):

        eot, vot, fot = tfnal_derivs.eval_ot (self, rho, Pi, dderiv=dderiv,
            weights=weights, _unpack_vot=_unpack_vot)
        if (self.verbose <= logger.DEBUG) or (dderiv<1) or (weights is None):
            return eot, vot, fot
        if rho.ndim == 2: rho = rho[:,None,:]
        if Pi.ndim == 1: Pi = Pi[None,:]
        rho_tot = rho.sum (0)
        nvr = rho_tot.shape[0]
        ngrids = rho_tot.shape[-1]

        r0 = 2*(np.random.rand (ngrids)-1)

        for p in range (20):
            # ~~~ eval_xc reference ~~~
            rho_t0 = self.get_rho_translated (Pi, rho)
            exc, vxc_p, fxc = self._numint.eval_xc (self.otxc, (rho_t0[0,:,:],
                rho_t0[1,:,:]), spin=1, relativity=0, deriv=dderiv,
                verbose=self.verbose)[:3]
            exc *= rho_t0[:,0,:].sum (0)
            vxc_p = tfnal_derivs._reshape_vxc_sigma (vxc_p, self.dens_deriv)
            vxc = tfnal_derivs._unpack_sigma_vector (vxc_p, rho_t0[0,1:4], rho_t0[1,1:4])
            if dderiv>1: fxc = tfnal_derivs._pack_fxc_ltri (fxc, self.dens_deriv)
            # ~~~ shift translated rho directly ~~~
            r = rho_t0 * r0 / 2**p
            drho_t = np.zeros_like (rho_t0, dtype=rho_t0.dtype)
            ndf = 2 * (1 + int (nvr>1))
            drho_t[0,0,0::ndf] = r[0,0,0::ndf]
            drho_t[1,0,1::ndf] = r[1,0,1::ndf]
            if ndf > 2:
                drho_t[0,1:4,2::ndf] = r[0,1:4,2::ndf]
                drho_t[1,1:4,3::ndf] = r[1,1:4,3::ndf]
            # ~~~ eval_xc @ rho_t1 = rho_t0 + drho_t ~~~
            rho_t1 = rho_t0 + drho_t
            exc1, vxc1 = self._numint.eval_xc (self.otxc, (rho_t1[0,:,:],
                rho_t1[1,:,:]), spin=1, relativity=0, deriv=dderiv,
                verbose=self.verbose)[:2]
            exc1 *= rho_t1[:,0,:].sum (0)
            vxc1 =  tfnal_derivs._unpack_vxc_sigma (vxc1, rho_t1, self.dens_deriv)
            df_lbl = ('rhoa', 'rhob', "rhoa'", "rhob'")[:2*(1+int(nvr>1))]
            _v_err_report (self, 'eval_xc {}'.format (p), df_lbl, rho_t0[0], rho_t0[1], exc, vxc,
                vxc_p, fxc, exc1, vxc1, drho_t, weights)

            # ~~~ eval_ot compare ~~~
            nvP = vot[1].shape[0]
            d1 = rho_tot[1:4] if nvr > 1 else None
            d2 = Pi[1:4] if nvP > 1 else None
            if _unpack_vot:
                vot_u = vot
                vot_p = tfnal_derivs.eval_ot (self, rho, Pi, dderiv=dderiv,
                                              weights=weights, _unpack_vot=False)[1]
            else:
                vot_p = vot
                vot_u = tfnal_derivs._unpack_sigma_vector (vot, d1, d2)
            drho = rho_tot * r0 / 2**p
            dPi = Pi * r0 / 2**p
            r, P = drho.copy (), dPi.copy ()
            drho[:] = dPi[:] = 0.0
            ndf = 2 + int(nvr>1) + int(nvP>1)
            drho[0,0::ndf] = r[0,0::ndf]
            dPi[0,1::ndf]  = P[0,1::ndf]
            if ndf > 2: drho[1:4,2::ndf] = r[1:4,2::ndf]
            if ndf > 3:  dPi[1:4,3::ndf] = P[1:4,3::ndf]
            rho1 = rho+(drho/2) # /2 because rho has one more dimension of size = 2
            # that gets summed later
            Pi1 = Pi + dPi
            # ~~~ ignore numerical instability of unfully-translated fnals ~~~
            if self.otxc[0].lower () == 't':
                z0 = self.get_zeta (self.get_ratio (Pi, rho_tot/2)[0],
                    fn_deriv=0)[0]
                z1 = self.get_zeta (self.get_ratio (Pi1, rho1.sum(0)/2)[0],
                    fn_deriv=0)[0]
                idx = (z0==0) |(z1==0)
                drho[:,idx] = dPi[:,idx] = 0
                rho1[:,:,idx] = rho[:,:,idx]
                Pi1[:,idx] = Pi[:,idx]
            # ~~~ eval_ot @ rho1 = rho + drho ~~~
            eot1, vot1 = tfnal_derivs.eval_ot (self, rho1, Pi1,
                dderiv=dderiv, weights=weights, _unpack_vot=True)[:2]
            #vot1 = tfnal_derivs._unpack_sigma_vector (vot1, d1, d2)
            df_lbl = ('rho', 'Pi', "rho'", "Pi'")[:ndf]
            _v_err_report (self, 'eval_ot {}'.format (p), df_lbl, rho_tot, Pi, eot, vot_u, vot_p, fot,
                eot1, vot1, (drho, dPi), weights)

        return eot, vot, fot

    eval_ot.__doc__ = otfnal.eval_ot.__doc__

# TODO: test continuity of smoothing function and warn at initialization?
class ftransfnal (transfnal):
    __doc__ = transfnal.__doc__ + r'''

    Extra attributes for ``fully-translated'' extension of Carlson
    et al., JCTC 11, 4077 (2015):

        R0 : float
            connecting point to polynomial smoothing function;
            R0 <= 1.0. Default is 0.9.
        R1 : float
            endpoint of polynomial smoothing function, zeta(R1) =
            zeta'(R1) = zeta''(R1) = 0.0; R1 >= 1.0. Default is 1.15.
        A : float
            Quintic coefficient of polynomial smoothing function.
            Default = -475.60656009 is chosen to make zeta continuous
            through its second derivative at given the default R0 and R1.
        B : float
            Quartic coefficient of polynomial smoothing function.
            Default = -379.47331922 is chosen to make zeta continuous
            through its second derivative given the default R0 and R1.
        C : float
            Cubic coefficient of polynomial smoothing function.
            Default = -85.38149682 chosen to make zeta continuous
            through its second derivative given the default R0 and R1.
    '''

    transl_prefix='ft'

    def __init__ (self, ks, **kwargs):
        otfnal.__init__(self, ks.mol, **kwargs)
        self.R0=FT_R0
        self.R1=FT_R1
        self.A=FT_A
        self.B=FT_B
        self.C=FT_C
        self.otxc = 'ft' + ks.xc
        self._numint = copy.copy (ks._numint)
        self._numint.libxc = libxc
        self.grids = copy.copy (ks.grids)
        self._numint.hybrid_coeff = ft_hybrid_coeff.__get__(self._numint)
        self._numint.nlc_coeff = ft_nlc_coeff.__get__(self._numint)
        self._numint.rsh_coeff = ft_rsh_coeff.__get__(self._numint)
        self._numint.eval_xc = ft_eval_xc.__get__(self._numint)
        self._numint._xc_type = ft_xc_type.__get__(self._numint)
        self._init_info ()

    Pi_deriv = transfnal.dens_deriv

    def get_rho_translated (self, Pi, rho):
        r''' Compute the "fully-translated" alpha and beta densities
        and their derivatives. This is the same as "translated" except

        rho'_t^a += zeta' * rho / 2
        rho'_t^b -= zeta' * rho / 2

        And the functional form of "zeta" is changed (see "get_zeta")

        Args:
            Pi : ndarray of shape (*, ngrids)
                containing on-top pair density [and derivatives]
            rho : ndarray of shape (2, *, ngrids)
                containing spin density [and derivatives]

        Returns:
            rho_ft : ndarray of shape (2,*,ngrids)
                Fully-translated spin density (and derivatives)
        '''
        nderiv_R = max (rho.shape[1], Pi.shape[0])
        if nderiv_R == 1: return transfnal.get_rho_translated (self, Pi, rho)

        # Spin density and first term of spin gradient in common with transfnal
        rho_avg = (rho[0,:,:] + rho[1,:,:]) / 2
        rho_ft, R, zeta = transfnal.get_rho_translated (self, Pi, rho,
            _fn_deriv=1)

        # Add propagation of chain rule through zeta
        w = (rho_avg[0] * zeta[1])[None,:] * R[1:4]
        rho_ft[0][1:4] += w
        rho_ft[1][1:4] -= w

        return rho_ft

    def get_zeta (self, R, fn_deriv=1):
        r''' Compute the intermediate zeta used to compute the translated spin
        densities and its functional derivatives

        From the "full" translation [Carlson et al., JCTC 11, 4077 (2015)]:
        zeta = (1-R)^(1/2)                          ; R < R0
             = A*(R-R1)^5 + B*(R-R1)^4 + C*(R-R1)^3 ; R0 <= R < R1
             = 0                                    ; otherwise

        Args:
            R : ndarray of shape (*,ngrids)
                Ratio (4Pi/rho^2) and possibly its spatial derivatives
                Only the first row is used in this function

        Kwargs:
            fn_deriv : integer
                order of functional derivative (d^n z / dR^n) to return
                along with the value of zeta

        Returns:
            zeta : ndarray of shape (fn_deriv+1, ngrids)
        '''
        # Rmax unused here. It only needs to be passed in the transfnal version
        if R.ndim == 2: R = R[0]
        R0, R1, A, B, C = self.R0, self.R1, self.A, self.B, self.C
        zeta = transfnal.get_zeta (self, R, fn_deriv=fn_deriv, _Rmax=R0)
        idx = (R >= R0) & (R < R1)
        if not np.count_nonzero (idx): return zeta
        zeta[:,idx] = 0.0
        dR = np.stack ([np.power (R[idx] - R1, n)
            for n in range (1,6)], axis=0)
        def _derivs ():
            yield A*dR[4] +    B*dR[3] +   C*dR[2]
            yield 5*A*dR[3] +  4*B*dR[2] + 3*C*dR[1]
            yield 20*A*dR[2] + 12*B*dR[1] + 6*C*dR[0]
            yield 60*A*dR[1] + 24*B*dR[0] + 6*C
            yield 120*A*dR[0] + 24*B
            yield 120*A
        for n, row in enumerate (_derivs ()):
            zeta[n,idx] = row
            if n == fn_deriv: break

        return zeta

    def jT_op (self, x, rho, Pi, **kwargs):
        r''' Evaluate jTx = (x.j)T where j is the Jacobian of the
        translated densities in terms of the untranslated density and
        pair density

        Args:
            x : ndarray of shape (2,*,ngrids)
                Usually, a functional derivative of the on-top xc energy
                wrt translated densities
            rho : ndarray of shape (2,*,ngrids)
                containing spin-density [and derivatives]
            Pi : ndarray with shape (*,ngrids)
                containing on-top pair density [and derivatives]

        Returns: ndarray of shape (*,ngrids)
            Usually, a functional derivative of the on-top pair density
            exchange-correlation energy wrt to total density and its
            derivatives. The potential must be spin-symmetric in
            pair-density functional theory.
        '''
        ntc = 2 + int(self.dens_deriv>0)
        ncol = 2 + 3*int(self.dens_deriv>0)
        ngrid = rho.shape[-1]
        jTx = np.zeros ((ncol,ngrid), dtype=x[0].dtype)
        jTx[:ntc,:] = transfnal.jT_op (self, x, rho, Pi, **kwargs)
        rho = rho.sum (0)
        R = self.get_ratio (Pi[0:4,:], rho[0:4,:]/2)
        zeta = self.get_zeta (R[0], fn_deriv=2)
        if self.dens_deriv > 0:
            jTx[:] += tfnal_derivs._ftGGA_jT_op (x, rho, Pi, R, zeta)
        return jTx

    def d_jT_op (self, x, rho, Pi, **kwargs):
        r''' Evaluate the x.(nabla j) contribution to the second density
        derivatives of the on-top energy in terms of the untranslated
        density and pair density

        Args:
            x : ndarray of shape (2,*,ngrids)
                Usually, a functional derivative of the on-top xc energy
                wrt translated densities
            rho : ndarray of shape (2,*,ngrids)
                containing spin-density [and derivatives]
            Pi : ndarray with shape (*,ngrids)
                containing on-top pair density [and derivatives]

        Returns: ndarray of shape (*,ngrids)
            second derivative of the translation dotted with x
            3 rows for tLDA and 5 rows for tGGA
        '''
        nrow_t = 3 + 2*int(self.dens_deriv>0)
        nrow = 3 + 12*int(self.dens_deriv>0)
        f = np.zeros ((nrow, x[0].shape[-1]), dtype=x[0].dtype)
        f[:nrow_t] = transfnal.d_jT_op (self, x, rho, Pi, **kwargs)
        if self.dens_deriv:
            rho = rho.sum (0)
            R = self.get_ratio (Pi[0:4,:], rho[0:4,:]/2)
            zeta = self.get_zeta (R[0], fn_deriv=3)
            f[:] += tfnal_derivs._ftGGA_d_jT_op (x, rho, Pi, R, zeta)
        return f


_CS_a_DEFAULT = 0.04918
_CS_b_DEFAULT = 0.132
_CS_c_DEFAULT = 0.2533
_CS_d_DEFAULT = 0.349

def _sanity_check_ftot(xc_code):
    '''
    This function will check the functional type and will
    raise the warning for fully-translated MGGAs or custom functionals.
    '''
    xc_type = libxc.xc_type(xc_code)
    if xc_type not in ['LDA', 'GGA']:
        msg = f"fully-translated {xc_type} on-top functionals are not defined"
        raise NotImplementedError(msg)

def get_transfnal (mol, otxc):
    if otxc.upper () in OT_ALIAS:
        otxc = OT_ALIAS[otxc.upper ()]
    if otxc.upper ().startswith ('T'):
        xc_base = otxc[1:]
        fnal_class = transfnal
    elif otxc.upper ().startswith ('FT'):
        xc_base = otxc[2:]
        _sanity_check_ftot(xc_base)
        fnal_class = ftransfnal
    else:
        raise NotImplementedError (
            'On-top pair-density functional names other than "translated" (t) or '
            '"fully-translated (ft).'
        )
    # Try to register the functional with libxc, if not already done
    xc_base = _get_registered_ot_functional (xc_base, mol)

    xc_base = OT_HYB_ALIAS.get (xc_base.upper (), xc_base)

    if ',' not in xc_base and \
        (xc_base.upper() not in REG_OT_FUNCTIONALS) and \
        (_libxc.is_hybrid_or_rsh (xc_base)):
        raise NotImplementedError (
            'Aliased or built-in translated hybrid or range-separated '
            'functionals\nother than those listed in otfnal.OT_HYB_ALIAS. '
            'Build a compound functional\nstring with a comma separating the '
            'exchange and correlation parts, or use\notfnal.make_hybrid_fnal '
            'instead.'
        )
    ks = dft.RKS (mol)
    ks.xc = xc_base
    return fnal_class (ks)

class colle_salvetti_corr (otfnal):


    def __init__(self, mol, **kwargs):
        super().__init__(mol, **kwargs)
        self.otxc = 'Colle_Salvetti'
        self._numint = NumInt ()
        self.grids = Grids (mol)
        self._numint.hybrid_coeff = lambda * args : 0
        self._numint.nlc_coeff = lambda * args : [0, 0]
        self._numint.rsh_coeff = lambda * args : [0, 0, 0]
        self._numint._xc_type = lambda * args : 'MGGA'
        self.CS_a =_CS_a_DEFAULT
        self.CS_b =_CS_b_DEFAULT
        self.CS_c =_CS_c_DEFAULT
        self.CS_d =_CS_d_DEFAULT
        self._init_info ()

    def get_E_ot (self, rho, Pi, weights):
        r''' Colle & Salvetti, Theor. Chim. Acta 37, 329 (1975)
        see also Lee, Yang, Parr, Phys. Rev. B 37, 785 (1988)
        [Eq. (3)]'''

        a, b, c, d = self.CS_a, self.CS_b, self.CS_c, self.CS_d
        rho_tot = rho[0,0] + rho[1,0]
        idx = rho_tot > 1e-15

        num  = -c * np.power (rho_tot[idx], -1/3)
        num  = np.exp (num, num)
        num *= Pi[4,idx]
        num *= b * np.power (rho_tot[idx], -8/3)
        num += 1

        denom  = d * np.power (rho_tot[idx], -1/3)
        denom += 1

        num /= denom
        num *= Pi[0,idx]
        num /= rho_tot[idx]
        num *= weights[idx]

        E_ot  = np.sum (num)
        E_ot *= -4 * a
        return E_ot

def _hybrid_2c_coeff (ni, xc_code, spin=0):
    ''' Wrapper to the xc_code hybrid coefficient parser to return the
    exchange and correlation components of the hybrid coefficient
    separately '''

    if xc_code.upper() in REG_OT_FUNCTIONALS:
        hyb_x = REG_OT_FUNCTIONALS[xc_code.upper ()].get('hyb_x', 0)
        hyb_c = REG_OT_FUNCTIONALS[xc_code.upper ()].get('hyb_c', 0)
        return [hyb_x, hyb_c]
    else:
        hyb_tot = _NumInt.hybrid_coeff (ni, xc_code, spin=spin)
        if hyb_tot == 0: return [0, 0]

        # For exchange-only functionals, hyb_c = hyb_x
        x_code, c_code = _libxc.split_x_c_comma (xc_code)
        x_code = x_code + ','
        c_code = ',' + c_code

        # All factors of 'HF' are summed by default. Therefore just run the same
        # code for the exchange and correlation parts of the string separately
        hyb_x = _NumInt.hybrid_coeff(ni, x_code, spin=spin) if len (x_code) else 0
        hyb_c = _NumInt.hybrid_coeff(ni, c_code, spin=spin) if len (c_code) else 0
        return [hyb_x, hyb_c]

def make_scaled_fnal (xc_code, hyb_x = 0, hyb_c = 0, fnal_x = None,
        fnal_c = None):
    ''' Convenience function to write the xc_code corresponding to a
        functional of the type

        Exc = hyb_x*E_x[Psi] + fnal_x*E_x[rho] + hyb_c*E_c[Psi]
              + fnal_c*E_c[rho]

        where E[Psi] is an energy from a wave function, and E[rho] is a
        density functional from libxc. The decomposition of E[Psi] into
        exchange (E_x) and correlation (E_c) components is arbitrary.

        Args:
            xc_code : string
                As used in pyscf.dft.libxc. An exception is raised if it
                is already a hybrid or contains a kinetic-energy
                functional component.

        Kwargs:
            hyb_x : float
                fraction of wave function exchange to be included
            hyb_c : float
                fraction of wave function correlation to be included
            fnal_x : float
                fraction of density functional exchange to be included.
                Defaults to 1 - hyb_x.
            fnal_c : float
                fraction of density functional correlation to be
                included. Defaults to 1 - hyb_c.

        returns:
            xc_code : string
                If xc_code has exchange part x_code and correlation part
                c_code, the return value is
                'fnal_x * x_code + hyb_x * HF,
                fnal_c * c_code + hyb_c * HF'
                You STILL HAVE TO PREPEND 't' OR 'ft'!!!
    '''
    if fnal_x is None: fnal_x = 1 - hyb_x
    if fnal_c is None: fnal_c = 1 - hyb_c

    if _libxc.is_hybrid_xc (xc_code):
        raise RuntimeError ('Functional {} is already a hybrid!'.format (
            xc_code))
    x_code, c_code = _libxc.split_x_c_comma (xc_code)

    x_facs, x_terms = _libxc.parse_xc_formula (x_code)
    if fnal_x != 1: x_facs = list (np.asarray (x_facs)*fnal_x)
    if hyb_x != 0:
        x_facs.append (hyb_x)
        x_terms.append ('HF')
    x_code = _libxc.assemble_xc_formula (x_facs, x_terms)

    c_facs, c_terms = _libxc.parse_xc_formula (c_code)
    if fnal_c != 1: c_facs = list (np.asarray (c_facs)*fnal_c)
    if hyb_c != 0:
        c_facs.append (hyb_c)
        c_terms.append ('HF')
    c_code = _libxc.assemble_xc_formula (c_facs, c_terms)

    return x_code + ',' + c_code

def make_hybrid_fnal (xc_code, hyb, hyb_type = 1):
    ''' Convenience function to write "hybrid" xc functional in terms of
        only one parameter

        Args:
            xc_code : string
                As used in pyscf.dft.libxc. An exception is raised if it
                is already a hybrid or contains a kinetic-energy
                functional component.
            hyb : float
                Parameter(s) defining the "hybridization" which is
                handled in various ways according to hyb_type

        Kwargs:
            hyb_type : int or string
                The type of hybrid functional. Current options are:
                - 0 or 'translation': Hybrid fnal is
                    'hyb*HF + (1-hyb)*x_code, hyb*HF + c_code'.
                    Based on the idea that 'exact exchange' of the
                    translated functional corresponds to exchange plus
                    correlation energy of the underlying wave function.
                    Requires len (hyb) == 1.
                - 1 or 'average': Hybrid fnal is
                    'hyb*HF + (1-hyb)*x_code, hyb*HF + (1-hyb)*c_code'.
                    Based on the idea that hyb = 1 recovers the wave
                    function energy itself. Requires len (hyb) == 1.
                - 2 or 'diagram': Hybrid fnal is
                    'hyb*HF + (1-hyb)*x_code, c_code'.
                    Based on the idea that the exchange energy of the
                    wave function somehow can be meaningfully separated
                    from the correlation energy. Requires len (hyb) == 1.
                - 3 or 'lambda': as in arXiv:1911.11162v1. Based on
                    existing 'double-hybrid' functionals. Requires
                    len (hyb) == 1.
                - 4 or 'scaling': Hybrid fnal is
                    'a*HF + (1-a)*x_code, a*HF + (1-a**b)*c_code'
                    where a = hyb[0] and b = 1 + hyb[1]. Based on the
                    scaling inequalities proven by Levy and Perdew in
                    PRA 32, 2010 (1985):
                    E_c[rho_a] < a*E_c[rho] if a < 1 and
                    E_c[rho_a] > a*E_c[rho] if a > 1;
                    BUT
                    E_c[rho_a] ~/~ a^2 E_c[rho], implying that
                    E_c[rho_a] ~ a^b E_c[rho] with b > 1 unknown.
                    Requires len (hyb) == 2.
    '''

    if not hasattr (hyb, '__len__'): hyb = [hyb]
    HYB_TYPE_CODE = {'translation': 0,
                     'average':     1,
                     'diagram':     2,
                     'lambda':      3,
                     'scaling':     4}
    if isinstance (hyb_type, str): hyb_type = HYB_TYPE_CODE[hyb_type]

    if hyb_type == 0:
        assert (len (hyb) == 1)
        return make_scaled_fnal (xc_code, hyb_x=hyb[0], hyb_c=hyb[0],
            fnal_x=(1-hyb[0]), fnal_c=1)
    elif hyb_type == 1:
        assert (len (hyb) == 1)
        return make_scaled_fnal (xc_code, hyb_x=hyb[0], hyb_c=hyb[0],
            fnal_x=(1-hyb[0]), fnal_c=(1-hyb[0]))
    elif hyb_type == 2:
        assert (len (hyb) == 1)
        return make_scaled_fnal (xc_code, hyb_x=hyb[0], hyb_c=0,
            fnal_x=(1-hyb[0]), fnal_c=1)
    elif hyb_type == 3:
        assert (len (hyb) == 1)
        return make_scaled_fnal (xc_code, hyb_x=hyb[0], hyb_c=hyb[0],
            fnal_x=(1-hyb[0]), fnal_c=(1-(hyb[0]*hyb[0])))
    elif hyb_type == 4:
        assert (len (hyb) == 2)
        a = hyb[0]
        b = hyb[0]**(1+hyb[1])
        return make_scaled_fnal (xc_code, hyb_x=a, hyb_c=a, fnal_x=(1-a),
            fnal_c=(1-b))
    else:
        raise RuntimeError ('hybrid type undefined')


# TODO: reconsider this goofy API...
__t_doc__="For 'translated' functionals, otxc string = 't'+xc string\n"
__ft_doc__="For 'fully translated' functionals, otxc string = 'ft'+xc string\n"

def t_hybrid_coeff(ni, xc_code, spin=0):
    #return _NumInt.hybrid_coeff(ni, xc_code[1:], spin=0)
    return _hybrid_2c_coeff (ni, xc_code[1:], spin=0)
t_hybrid_coeff.__doc__ = __t_doc__ + str(_NumInt.hybrid_coeff.__doc__)

def t_nlc_coeff(ni, xc_code):
    return _NumInt.nlc_coeff(ni, xc_code[1:])
t_nlc_coeff.__doc__ = __t_doc__ + str(_NumInt.nlc_coeff.__doc__)

def t_rsh_coeff(ni, xc_code):
    return _NumInt.rsh_coeff(ni, xc_code[1:])
t_rsh_coeff.__doc__ = __t_doc__ + str(_NumInt.rsh_coeff.__doc__)

def t_eval_xc(ni, xc_code, rho, spin=0, relativity=0, deriv=1, verbose=None):
    return _NumInt.eval_xc(ni, xc_code[1:], rho, spin=spin,
        relativity=relativity, deriv=deriv, verbose=verbose)
t_eval_xc.__doc__ = __t_doc__ + str(_NumInt.eval_xc.__doc__)

def t_xc_type(ni, xc_code):
    return _NumInt._xc_type(ni, xc_code[1:])
t_xc_type.__doc__ = __t_doc__ + str(_NumInt._xc_type.__doc__)

def t_rsh_and_hybrid_coeff(ni, xc_code, spin=0):
    return _NumInt.rsh_and_hybrid_coeff (ni, xc_code[1:], spin=spin)
t_rsh_and_hybrid_coeff.__doc__ = (__t_doc__
    + str(_NumInt.rsh_and_hybrid_coeff.__doc__))

def ft_hybrid_coeff(ni, xc_code, spin=0):
    #return _NumInt.hybrid_coeff(ni, xc_code[2:], spin=0)
    return _hybrid_2c_coeff(ni, xc_code[2:], spin=0)
ft_hybrid_coeff.__doc__ = __ft_doc__ + str(_NumInt.hybrid_coeff.__doc__)

def ft_nlc_coeff(ni, xc_code):
    return _NumInt.nlc_coeff(ni, xc_code[2:])
ft_nlc_coeff.__doc__ = __ft_doc__ + str(_NumInt.nlc_coeff.__doc__)

def ft_rsh_coeff(ni, xc_code):
    return _NumInt.rsh_coeff(ni, xc_code[2:])
ft_rsh_coeff.__doc__ = __ft_doc__ + str(_NumInt.rsh_coeff.__doc__)

def ft_eval_xc(ni, xc_code, rho, spin=0, relativity=0, deriv=1, verbose=None):
    return _NumInt.eval_xc(ni, xc_code[2:], rho, spin=spin,
        relativity=relativity, deriv=deriv, verbose=verbose)
ft_eval_xc.__doc__ = __ft_doc__ + str(_NumInt.eval_xc.__doc__)

def ft_xc_type(ni, xc_code):
    return _NumInt._xc_type(ni, xc_code[2:])
ft_xc_type.__doc__ = __ft_doc__ + str(_NumInt._xc_type.__doc__)

def ft_rsh_and_hybrid_coeff(ni, xc_code, spin=0):
    return _NumInt.rsh_and_hybrid_coeff (ni, xc_code[2:], spin=spin)
ft_rsh_and_hybrid_coeff.__doc__ = (__ft_doc__
    + str(_NumInt.rsh_and_hybrid_coeff.__doc__))

def _v_err_report (otfnal, tag, lbls, rho_tot, Pi, e0, v0, v0_packed, f, e1, v1, x, w):
    # Examine the error of the first and second functional derivatives in the
    # debugging block under transfnal.eval_ot below
    logger.debug (otfnal, '--- v_err_report (%s) ---', tag)
    ndf = len (lbls)
    nvP = v0[1].shape[0]
    de = (e1-e0) * w
    vx = ((v0[0]*x[0]).sum (0) + (v0[1]*x[1][:nvP]).sum (0)) * w
    if f is None:
        xfx = np.zeros_like (de)
    else:
        xf = tfnal_derivs.contract_fot (otfnal, f, rho_tot, Pi, x[0], x[1], vot_packed=v0_packed)
        for row in xf: row[:] *= w
        xfx = ((xf[0]*x[0]).sum (0) + (xf[1]*x[1][:nvP]).sum (0)) / 2
        xf_df = [xf[0][0], xf[1][0]]
        dv_df = [(v1[0][0]-v0[0][0])*w, (v1[1][0]-v0[1][0])*w]
        # The lesson of the debug experience from the commented-out block below is:
        # the largest errors (fractional or absolute) in the ftLDA fnal gradient
        # appear to be for R just under 1.0!
        #if 'LDA' in otfnal.otxc:
        #    print ("bigtab", otfnal.otxc, (np.sum (vx) - np.sum (de))/np.sum (de))
        #    tab = np.empty ((xf[0][0].size, 6), dtype=xf[0].dtype)
        #    tab[:,0] = otfnal.get_ratio (Pi, rho_tot/2)[0] - 1.0
        #    tab[:,1] = rho_tot
        #    tab[:,2] = Pi
        #    tab[:,3] = vx
        #    tab[:,4] = tab[:,3] - de
        #    tab[:,5] = tab[:,4] / de
        #    tab[(de==0)&(vx==0),3] = 0.0
        #    tab[(de==0)&(vx!=0),3] = 1.0
        #    tab = tab[np.argsort (-np.abs (tab[:,4])),:]
        #    for row in tab:
        #        print ("{:20.12e} {:9.2e} {:9.2e} {:9.2e} {:9.2e} {:9.2e}".format
        #           (*row))
        if ndf > 2:
            xf_df += [xf[0][1:4].T,]
            dv_df += [((v1[0][1:4]-v0[0][1:4])*w).T,]
        if ndf > 3:
            xf_df += [xf[1][1:4].T,]
            dv_df += [((v1[1][1:4]-v0[1][1:4])*w).T,]
    de_err1 = de - vx
    de_err2 = de_err1 - xfx
    for ix, lbl in enumerate (lbls):
        lib.logger.debug (otfnal, "%s gradient debug %s: %e - %e (- %e) -> %e "
            "(%e)", tag, lbl, np.sum  (de[ix::ndf]), np.sum (vx[ix::ndf]),
            np.sum (xfx[ix::ndf]), np.sum (de_err1[ix::ndf]),
            np.sum (de_err2[ix::ndf]))
    if f is not None:
        for lbl_row, xf_row, dv_row in zip (lbls, xf_df, dv_df):
            err_row = dv_row-xf_row
            for ix_col, lbl_col in enumerate (lbls):
                lib.logger.debug (otfnal, ("%s Hessian debug (H.x_%s)_%s: "
                    "%e - %e -> %e"), tag, lbl_col, lbl_row,
                    linalg.norm (dv_row[ix_col::ndf]),
                    linalg.norm (xf_row[ix_col::ndf]),
                    linalg.norm (err_row[ix_col::ndf]))
                # I am not doing rho'.rho'->sigma right for x.f.x/2
                # However I am somehow doing it right for f.x vs. delta v?
        lib.logger.debug (otfnal, "%s dE - v.x - x.f.x: %e - %e - %e = %e",
            tag, de.sum (), vx.sum (), xfx.sum (), de_err2.sum ())
