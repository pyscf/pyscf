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
#         Hong-Zhou Ye <hzyechem@gmail.com>
#

'''
Pipek-Mezey localization

ref. JCTC 10, 642 (2014); DOI:10.1021/ct401016x
'''

import numpy
from functools import reduce

from pyscf import lib
from pyscf.lib import logger
from pyscf.lo import orth
from pyscf.lo import boys
from pyscf.lo import iao
from pyscf.lo.stability import pipek_stability_jacobi, stability_newton
from pyscf import __config__


def atomic_pops(mol, mo_coeff, method='meta_lowdin', kpt=None, proj_data=None, mode=None,
                verbose=None):
    r'''
    Kwargs:
        method : string
            The atomic population projection scheme. It can be mulliken,
            lowdin, meta_lowdin, iao, or becke
        mode : str or None
            Specifies which matrix elements to compute.

            If ``mode == 'pop'``, only diagonal elements
                ``< i | \hat{P}_{\mathrm{atm}} | i >``
            are evaluated. This mode is optimized for efficient computation of
            atomic populations and the PM metric function.

            For any other value of ``mode`` (including ``None``), the full matrix
                ``< i | \hat{P}_{\mathrm{atm}} | j >``
            is computed.

            Default is ``None``.

    Returns:
        A 3-index tensor [A,i,j] indicates the population of any orbital-pair
        density |i><j| for each species (atom in this case).  This tensor is
        used to construct the population and gradients etc.

        You can customize the PM localization wrt other population metric,
        such as the charge of a site, the charge of a fragment (a group of
        atoms) by overwriting this tensor.  See also the example
        pyscf/examples/loc_orb/40-hubbard_model_PM_localization.py for the PM
        localization of site-based population for hubbard model.
    '''
    method = method.lower().replace('_', '-')
    nmo = mo_coeff.shape[1]
    if mode is None: mode = 'full'

    def proj_orth(mo_coeff, proj_coeff, offset_nr_by_atom):
        csc = lib.dot(proj_coeff.conj().T, mo_coeff)

        if mode == 'pop':
            proj = numpy.empty((mol.natm,nmo), dtype=numpy.float64)
            for i, (b0, b1, p0, p1) in enumerate(offset_nr_by_atom):
                proj[i] = (abs(csc[p0:p1])**2).sum(axis=0)
        else:
            proj = numpy.empty((mol.natm,nmo,nmo), dtype=csc.dtype)
            for i, (b0, b1, p0, p1) in enumerate(offset_nr_by_atom):
                lib.dot(csc[p0:p1].conj().T, csc[p0:p1], c=proj[i])
        return proj

    def proj_biorth(mo_coeff, proj_coeff, projtild_coeff, offset_nr_by_atom):
        csc = lib.dot(proj_coeff.conj().T, mo_coeff)
        csctild = lib.dot(projtild_coeff.conj().T, mo_coeff)

        if mode == 'pop':
            proj = numpy.empty((mol.natm,nmo), dtype=numpy.float64)
            for i, (b0, b1, p0, p1) in enumerate(offset_nr_by_atom):
                proj[i] = (csc[p0:p1].conj()*csctild[p0:p1]).sum(axis=0).real
        else:
            proj = numpy.empty((mol.natm,nmo,nmo), dtype=csc.dtype)
            for i, (b0, b1, p0, p1) in enumerate(offset_nr_by_atom):
                lib.dot(csc[p0:p1].conj().T, csctild[p0:p1], c=proj[i], alpha=0.5)
                proj[i] += proj[i].conj().T
        return proj

    if proj_data is None:
        proj_data = get_proj_data(mol, mo_coeff, method, kpt)

    if method == 'becke':
        charge_matrices = proj_data

        if mode == 'pop':
            proj = numpy.empty((mol.natm,nmo), dtype=numpy.float64)
            for i in range(mol.natm):
                proj[i] = lib.einsum('mi,mn,ni->i', mo_coeff.conj(), charge_matrices[i],
                                     mo_coeff).real
        else:
            proj = numpy.empty((mol.natm,nmo,nmo), dtype=mo_coeff.dtype)
            for i in range(mol.natm):
                proj[i] = reduce(lib.dot, (mo_coeff.conj().T, charge_matrices[i], mo_coeff))

    elif method == 'mulliken':
        s = get_ovlp(mol, kpt)
        csc = mo_coeff
        csctild = lib.dot(s, mo_coeff)
        if mode == 'pop':
            proj = numpy.empty((mol.natm,nmo), dtype=numpy.float64)
            for i, (b0, b1, p0, p1) in enumerate(mol.offset_nr_by_atom()):
                proj[i] = (csc[p0:p1].conj()*csctild[p0:p1]).sum(axis=0).real
        else:
            proj = numpy.empty((mol.natm,nmo,nmo), dtype=mo_coeff.dtype)
            for i, (b0, b1, p0, p1) in enumerate(mol.offset_nr_by_atom()):
                proj[i] = lib.dot(csc[p0:p1].conj().T, csctild[p0:p1], c=proj[i], alpha=0.5)
                proj[i] += proj[i].conj().T

    elif method in ('lowdin', 'meta-lowdin'):
        proj_coeff, offset_nr_by_atom = proj_data
        proj = proj_orth(mo_coeff, proj_coeff, offset_nr_by_atom)

    elif method == 'iao-biorth':
        proj_coeff, projtild_coeff, offset_nr_by_atom = proj_data
        proj = proj_biorth(mo_coeff, proj_coeff, projtild_coeff, offset_nr_by_atom)

    elif method in ('iao', 'ibo'):  # Why is 'ibo' the same as 'iao'...?
        proj_coeff, offset_nr_by_atom = proj_data
        proj = proj_orth(mo_coeff, proj_coeff, offset_nr_by_atom)

    else:
        raise KeyError('method = %s' % method)

    return proj


def gen_proj_op(mol, mo_coeff, method='meta_lowdin', kpt=None, proj_data=None, verbose=None):
    ''' Calculate projection matrix (same as the return of :func:`atomic_pops`) and
        its action on a given vector (for efficient Hvp).
    '''
    method = method.lower().replace('_', '-')
    nmo = mo_coeff.shape[1]

    def get_proj_op_orth(mo_coeff, proj_coeff, offset_nr_by_atom):
        natm = len(offset_nr_by_atom)

        csc = lib.dot(proj_coeff.conj().T, mo_coeff)    # nproj,nmo
        proj = numpy.empty((natm,nmo,nmo), dtype=csc.dtype)
        for i, (b0, b1, p0, p1) in enumerate(offset_nr_by_atom):
            lib.dot(csc[p0:p1].conj().T, csc[p0:p1], c=proj[i])

        def proj_op(x):
            cscx = lib.dot(csc, x)
            projx = numpy.empty((natm,nmo,nmo), dtype=cscx.dtype)
            for i, (b0, b1, p0, p1) in enumerate(offset_nr_by_atom):
                lib.dot(csc[p0:p1].conj().T, cscx[p0:p1], c=projx[i])
            return projx

        return proj, proj_op

    def get_proj_op_biorth(mo_coeff, proj_coeff, projtild_coeff, offset_nr_by_atom):
        natm = len(offset_nr_by_atom)

        csc = lib.dot(proj_coeff.conj().T, mo_coeff)            # nproj,nmo
        csctild = lib.dot(projtild_coeff.conj().T, mo_coeff)    # nproj,nmo
        proj = numpy.empty((natm,nmo,nmo), dtype=csc.dtype)
        for i, (b0, b1, p0, p1) in enumerate(offset_nr_by_atom):
            lib.dot(csc[p0:p1].conj().T, csctild[p0:p1], c=proj[i], alpha=0.5)
            proj[i] += proj[i].conj().T

        def proj_op(x):
            cscx = lib.dot(csc, x)
            csctildx = lib.dot(csctild, x)
            projx = numpy.empty((natm,nmo,nmo), dtype=cscx.dtype)
            for i, (b0, b1, p0, p1) in enumerate(offset_nr_by_atom):
                lib.dot(csc[p0:p1].conj().T, csctildx[p0:p1], c=projx[i])
                lib.dot(csctild[p0:p1].conj().T, cscx[p0:p1], c=projx[i], beta=1)
            projx *= 0.5
            return projx

        return proj, proj_op


    if proj_data is None:
        proj_data = get_proj_data(mol, mo_coeff, method, kpt)

    if method == 'becke':
        charge_matrices = proj_data
        natm = len(charge_matrices)

        cms = [lib.dot(mo_coeff.conj().T, charge_matrices[i]) for i in range(natm)]

        proj = numpy.empty((natm,nmo,nmo), dtype=mo_coeff.dtype)
        for i in range(natm):
            proj[i] = lib.dot(cms[i], mo_coeff)

        def proj_op(x):
            cx = lib.dot(mo_coeff, x)
            projx = numpy.empty((natm,nmo,nmo), dtype=cx.dtype)
            for i in range(natm):
                projx[i] = lib.dot(cms[i], cx)
            return projx

    elif method == 'mulliken':
        natm = mol.natm
        s = get_ovlp(mol, kpt)
        csc = mo_coeff
        csctild = lib.dot(s, mo_coeff)

        proj = numpy.empty((mol.natm,nmo,nmo), dtype=mo_coeff.dtype)
        for i, (b0, b1, p0, p1) in enumerate(mol.offset_nr_by_atom()):
            lib.dot(csc[p0:p1].conj().T, csctild[p0:p1], c=proj[i], alpha=0.5)
            proj[i] += proj[i].conj().T

        def proj_op(x):
            cscx = lib.dot(csc, x)
            csctildx = lib.dot(csctild, x)
            projx = numpy.empty((natm, nmo, nmo), dtype=cscx.dtype)
            for i, (b0, b1, p0, p1) in enumerate(mol.offset_nr_by_atom()):
                lib.dot(csc[p0:p1].conj().T, csctildx[p0:p1], c=projx[i])
                lib.dot(csctild[p0:p1].conj().T, cscx[p0:p1], c=projx[i], beta=1)
            projx *= 0.5
            return projx

    elif method in ('lowdin', 'meta-lowdin'):
        proj_coeff, offset_nr_by_atom = proj_data
        proj, proj_op = get_proj_op_orth(mo_coeff, proj_coeff, offset_nr_by_atom)

    elif method == 'iao-biorth':
        proj_coeff, projtild_coeff, offset_nr_by_atom = proj_data
        proj, proj_op = get_proj_op_biorth(mo_coeff, proj_coeff, projtild_coeff, offset_nr_by_atom)

    elif method in ('iao', 'ibo'):  # Why is 'ibo' the same as 'iao'...?
        proj_coeff, offset_nr_by_atom = proj_data
        proj, proj_op = get_proj_op_orth(mo_coeff, proj_coeff, offset_nr_by_atom)

    else:
        raise KeyError('method = %s' % method)

    return proj, proj_op


def get_ovlp(mol, kpt=None):
    if getattr(mol, 'pbc_intor', None):  # whether mol object is a cell
        s = mol.pbc_intor('int1e_ovlp', hermi=1, kpt=kpt)
    else:
        s = mol.intor_symmetric('int1e_ovlp')
    return s


def becke_charge_matrices(mol):
    from pyscf.dft import gen_grid
    # Call DFT to initialize grids and numint objects
    mf = mol.RKS()
    grids = mf.grids
    ni = mf._numint

    if not isinstance(grids, gen_grid.Grids):
        raise NotImplementedError('PM becke scheme for PBC systems')

    # The atom-wise Becke grids (without concatenated to a vector of grids)
    coords, weights = grids.get_partition(mol, concat=False)

    charge_matrices = []
    for i in range(mol.natm):
        ao = ni.eval_ao(mol, coords[i], deriv=0)
        aow = numpy.einsum('pi,p->pi', ao, weights[i])
        charge_matrices.append(lib.dot(aow.conj().T, ao))

    return charge_matrices


def get_proj_data(mol, mo_coeff, method, kpt, minao=None):
    ''' Precompute data for atomic projectors
    '''
    if method is None:  # allow customized population method to skip this precompute
        return None

    method = method.lower().replace('_', '-')

    if method == 'becke':
        proj_data = becke_charge_matrices(mol)

    elif method == 'mulliken':
        proj_data = None

    elif method in ('lowdin', 'meta-lowdin'):
        s = get_ovlp(mol, kpt)
        proj_coeff = orth.orth_ao(mol, method, 'ANO', s=s, adjust_phase=False)
        proj_coeff = lib.dot(s, proj_coeff)
        proj_data = (proj_coeff, mol.offset_nr_by_atom())

    elif method in ('iao', 'ibo', 'iao-biorth'):
        if minao is None: minao = 'minao'
        s = get_ovlp(mol, kpt)
        if kpt is None:
            iao_coeff = iao.iao(mol, mo_coeff, minao=minao)
        else:
            iao_coeff = iao.iao(mol, [mo_coeff], kpts=[kpt], minao=minao)[0]
        iao_mol = iao.reference_mol(mol, minao=minao)

        if method == 'iao-biorth':
            ovlp = reduce(lib.dot, (iao_coeff.conj().T, s, iao_coeff))
            iaotild_coeff = numpy.asarray(numpy.linalg.solve(ovlp,
                                          iao_coeff.conj().T).conj().T, order='C')
            proj_coeff = lib.dot(s, iao_coeff)
            projtild_coeff = lib.dot(s, iaotild_coeff)
            proj_data = (proj_coeff, projtild_coeff, iao_mol.offset_nr_by_atom())
        else:
            iao_coeff = orth.vec_lowdin(iao_coeff, s)
            proj_coeff = lib.dot(s, iao_coeff)
            proj_data = (proj_coeff, iao_mol.offset_nr_by_atom())

    else:
        raise KeyError('method = %s' % method)

    return proj_data


class PipekMezey(boys.OrbitalLocalizer):
    '''The Pipek-Mezey localization optimizer that maximizes the orbital
    population using real-valued orthogonal rotation.

    Args:
        mol : Mole object

        mo_coeff : size (N,N) numpy.array
            The orbital space to localize for PM localization.
            When initializing the localization optimizer ``bopt = PM(mo_coeff)``,

            Note these orbitals ``mo_coeff`` may or may not be used as initial
            guess, depending on the attribute ``.init_guess`` . If ``.init_guess``
            is set to None, the ``mo_coeff`` will be used as initial guess. If
            ``.init_guess`` is 'atomic', a few atomic orbitals will be
            constructed inside the space of the input orbitals and the atomic
            orbitals will be used as initial guess.

            Note when calling .kernel(orb) method with a set of orbitals as
            argument, the orbitals will be used as initial guess regardless of
            the value of the attributes .mo_coeff and .init_guess.

    Attributes for PM class:
        verbose : int
            Print level.  Default value equals to :class:`Mole.verbose`.
        max_memory : float or int
            Allowed memory in MB.  Default value equals to :class:`Mole.max_memory`.
        conv_tol : float
            Converge threshold.  Default 1e-6
        conv_tol_grad : float
            Converge threshold for orbital rotation gradients.  Default 1e-3
        max_cycle : int
            The max. number of macro iterations. Default 100
        max_iters : int
            The max. number of iterations in each macro iteration. Default 20
        max_stepsize : float
            The step size for orbital rotation.  Small step (0.005 - 0.05) is preferred.
            Default 0.03.
        init_guess : str or None
            Initial guess for optimization. If set to None, orbitals defined
            by the attribute .mo_coeff will be used as initial guess. If set
            to 'atomic', atomic orbitals will be used as initial guess.
            Default 'atomic'
        pop_method : str
            How the orbital population is calculated, see JCTC 10, 642
            (2014) for discussion. Options are:
            - 'meta-lowdin' (default) as defined in JCTC 10, 3784 (2014)
            - 'mulliken' original Pipek-Mezey scheme, JCP 90, 4916 (1989)
            - 'lowdin' Lowdin charges, JCTC 10, 642 (2014)
            - 'iao' or 'ibo' intrinsic atomic orbitals with symmetric (i.e., Lowdin)
              orthogonalization, JCTC 9, 4384 (2013)
            - 'iao-biorth' biorthogonalized IAOs, JPCA 128, 8570 (2024)
            - 'becke' Becke charges, JCTC 10, 642 (2014)
            The IAO and Becke charges do not depend explicitly on the
            basis set, and have a complete basis set limit [JCTC 10,
            642 (2014)].
        exponent : int
            The power to define norm. It can be any integer >= 2. Default 2.
        algorithm : str
            Algorithm for maximizing the PM metric function. Currently support
            'ciah' and 'bfgs'. Default 'ciah'.
        minao : str or basis
            MINAO for constructing IAO. This switch only affects calculations with
            `pop_method` = 'iao'/'ibo'/'iao-biorth'. Default 'minao'.

    Saved results

        mo_coeff : ndarray
            Localized orbitals

    '''


    pop_method = getattr(__config__, 'lo_pipek_PM_pop_method', 'meta_lowdin')
    conv_tol = getattr(__config__, 'lo_pipek_PM_conv_tol', 1e-6)
    exponent = getattr(__config__, 'lo_pipek_PM_exponent', 2)   # any integer >= 2
    minao = getattr(__config__, 'lo_pipek_PM_minao', 'minao')   # allow user defined MINAO

    _keys = {'pop_method', 'conv_tol', 'exponent', 'kpt', '_proj_data', 'minao'}

    def __init__(self, mol, mo_coeff, pop_method=None, kpt=None):
        boys.OrbitalLocalizer.__init__(self, mol, mo_coeff)
        self.maximize = True
        if pop_method is not None:
            self.pop_method = pop_method
        self.kpt = kpt
        self._proj_data = None

    def dump_flags(self, verbose=None):
        boys.OrbitalLocalizer.dump_flags(self, verbose)
        logger.info(self, 'pop_method = %s',self.pop_method)
        logger.info(self, 'exponent = %s',self.exponent)

    def get_proj_data(self, mol=None, mo_coeff=None, method=None, kpt=None, minao=None):
        if mol is None: mol = self.mol
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if method is None: method = self.pop_method
        if kpt is None: kpt = self.kpt
        if minao is None: minao = self.minao

        log = logger.new_logger(self, verbose=self.verbose-1)
        cput0 = (logger.process_clock(), logger.perf_counter())

        proj_data = get_proj_data(mol, mo_coeff, method, kpt, minao=minao)

        log.timer('get_proj_data', *cput0)

        return proj_data

    def gen_g_hop(self, u=None):
        exponent = self.exponent
        mo_coeff = self.rotate_orb(u)

        if self.pop_method is None: # customized pop method
            proj = self.atomic_pops(self.mol, mo_coeff)
            proj_op = None
        else:
            proj, proj_op = gen_proj_op(self.mol, mo_coeff, method=self.pop_method,
                                        kpt=self.kpt, proj_data=self._proj_data,
                                        verbose=self.verbose)

        # Only the real part of proj is needed for real rotations
        proj = numpy.ascontiguousarray(proj.real.transpose(1,2,0)) # i,j,x
        pop = numpy.ascontiguousarray(lib.einsum('iix->ix', proj))
        popexp1 = pop**(exponent-1)
        popexp2 = pop**(exponent-2)

        # gradient
        g = self.get_grad(u, proj=proj)

        # hessian diagonal
        g1 = lib.einsum('ix,ix->i', popexp1, pop)
        g2 = lib.einsum('ix,jx->ij', popexp1, pop)
        h_diag  = 2 * exponent * (g1[:,None] - g2)
        g1 = lib.einsum('ijx,jx->ij', proj**2, popexp2)
        h_diag += -4 * exponent * (exponent-1) * g1
        h_diag = self.pack_uniq_var(h_diag + h_diag.T)

        # hessian vector product
        G = lib.einsum('ijx,jx->ij', proj, popexp1)

        def h_op(x):
            x = self.unpack_uniq_var(x)

            # Only the real part of projx is needed for real rotations
            if proj_op is None:
                projx = lib.einsum('ilx,lj->ijx', proj, x)
            else:
                projx = numpy.ascontiguousarray(proj_op(x).real.transpose(1,2,0))

            # disconnected
            j0 = popexp2 * numpy.ascontiguousarray(lib.einsum('iix->ix', projx.real))
            j1 = lib.einsum('ijx,jx->ij', proj, j0)
            hx = -4 * exponent * (exponent-1) * j1

            # connected symmetric
            hx += -2 * exponent * lib.einsum('ijx,jx->ij', projx, popexp1)

            # connected asymmetric
            hx += -exponent * lib.dot(G, x.T)
            hx += exponent * lib.einsum('ijx,ix->ij', projx, popexp1)

            return self.pack_uniq_var(hx - hx.T)

        return g, h_op, h_diag

    def get_grad(self, u=None, proj=None):
        if proj is None:
            # Only the real part of proj is needed for real rotations
            mo_coeff = self.rotate_orb(u)
            proj = self.atomic_pops(self.mol, mo_coeff)
            proj = numpy.ascontiguousarray(proj.transpose(1,2,0))   # i,j,x

        exponent = self.exponent
        popexp1 = numpy.ascontiguousarray(lib.einsum('iix->ix', proj.real))**(exponent-1)
        g = -lib.einsum('ijx,jx->ij', proj, popexp1)
        return 2 * exponent * self.pack_uniq_var(g - g.T)

    def cost_function(self, u=None, mode='pop'):
        mo_coeff = self.rotate_orb(u)
        if mode == 'pop':
            pop = self.atomic_pops(self.mol, mo_coeff, mode='pop')
            return (pop**self.exponent).sum()
        else:
            proj = self.atomic_pops(self.mol, mo_coeff)
            return (lib.einsum('xii->xi', proj.real)**self.exponent).sum()

    @lib.with_doc(atomic_pops.__doc__)
    def atomic_pops(self, mol, mo_coeff, method=None, kpt=None, proj_data=None, mode=None,
                    verbose=None):
        if method is None: method = self.pop_method
        if proj_data is None: proj_data = self._proj_data
        if kpt is None: kpt = self.kpt
        if verbose is None: verbose = self.verbose

        return atomic_pops(mol, mo_coeff, method, kpt, proj_data, mode, verbose)

    def kernel(self, mo_coeff=None, callback=None, verbose=None):
        self._proj_data = self.get_proj_data()
        mo_coeff = boys.kernel(self, mo_coeff, callback, verbose)
        self._proj_data = None

        return mo_coeff

    def stability_jacobi(self, verbose=None, return_status=False):
        self._proj_data = self.get_proj_data()
        res = pipek_stability_jacobi(self, verbose=verbose, return_status=return_status)
        self._proj_data = None

        return res

    def stability(self, verbose=None, return_status=False):
        self._proj_data = self.get_proj_data()
        res = stability_newton(self, verbose=verbose, return_status=return_status)
        self._proj_data = None

        return res


PM = Pipek = PipekMezey


@lib.with_doc(PipekMezey.__doc__)
class PipekMezeyComplex(PipekMezey, boys.OrbitalLocalizerComplex):
    '''The Pipek-Mezey localization optimizer that maximizes the orbital
    population using complex unitary rotation.

    See the docstring of PipekMezey for Args/Kwargs.
    '''
    def __init__(self, mol, mo_coeff, pop_method=None, kpt=None):
        boys.OrbitalLocalizerComplex.__init__(self, mol, mo_coeff)
        self.maximize = True
        if pop_method is not None:
            self.pop_method = pop_method
        self.kpt = kpt
        self._proj_data = None

    def gen_g_hop(self, u=None):
        exponent = self.exponent
        mo_coeff = self.rotate_orb(u)

        if self.pop_method is None: # customized pop method
            proj = self.atomic_pops(self.mol, mo_coeff)
            proj_op = None
        else:
            proj, proj_op = gen_proj_op(self.mol, mo_coeff, method=self.pop_method,
                                        kpt=self.kpt, proj_data=self._proj_data,
                                        verbose=self.verbose)

        proj = numpy.ascontiguousarray(proj.transpose(1,2,0)) # i,j,x
        pop = numpy.ascontiguousarray(lib.einsum('iix->ix', proj.real))
        popexp1 = pop**(exponent-1)
        popexp2 = pop**(exponent-2)

        # gradient
        g = self.get_grad(u, proj=proj)

        # hessian diagonal
        g1 = lib.einsum('ix,ix->i', popexp1, pop)
        g2 = lib.einsum('ix,jx->ij', popexp1, pop)
        h_diag = 2 * exponent * (g1[:,None] - g2) * (1 + 1j)
        g1 = lib.einsum('ijx,jx->ij', proj.real**2, popexp2)
        g2 = lib.einsum('ijx,jx->ij', proj.imag**2, popexp2)
        h_diag += -4 * exponent * (exponent - 1) * (g1 + g2 * 1j)
        h_diag = self.pack_uniq_var(h_diag + h_diag.T)

        # hessian vector product
        G = lib.einsum('ijx,jx->ij', proj, popexp1)

        def h_op(x):
            x = self.unpack_uniq_var(x)

            if proj_op is None:
                projx = lib.einsum('ilx,lj->ijx', proj, x)
            else:
                projx = numpy.ascontiguousarray(proj_op(x).transpose(1,2,0))

            # disconnected
            j0 = popexp2 * numpy.ascontiguousarray(lib.einsum('iix->ix', projx.real))
            j1 = lib.einsum('ijx,jx->ij', proj, j0)
            hx = -4 * exponent * (exponent-1) * j1.astype(numpy.complex128)

            # connected symmetric
            hx += -2 * exponent * lib.einsum('ijx,jx->ij', projx, popexp1)

            # connected asymmetric
            hx += -exponent * lib.dot(G, x.conj().T)
            hx += exponent * lib.einsum('ijx,ix->ij', projx, popexp1)

            return self.pack_uniq_var(hx - hx.conj().T)

        return g, h_op, h_diag

    def get_grad(self, u=None, proj=None):
        if proj is None:
            mo_coeff = self.rotate_orb(u)
            proj = self.atomic_pops(self.mol, mo_coeff)
            proj = numpy.ascontiguousarray(proj.transpose(1,2,0))   # i,j,x

        exponent = self.exponent
        popexp1 = numpy.ascontiguousarray(lib.einsum('iix->ix', proj.real))**(exponent-1)
        g = -lib.einsum('ijx,jx->ij', proj, popexp1)
        return 2 * exponent * self.pack_uniq_var(g - g.conj().T)


PMComplex = PipekComplex = PipekMezeyComplex


if __name__ == '__main__':
    from pyscf import gto, scf
    from pyscf.lo.tools import findiff_grad, findiff_hess

    mol = gto.Mole()
    mol.atom = '''
    O          0.00000        0.00000        0.11779
    H          0.00000        0.75545       -0.47116
    H          0.00000       -0.75545       -0.47116
    '''
    mol.basis = 'ccpvdz'
    mol.build()
    mf = scf.RHF(mol).run()

    log = logger.new_logger(mol, verbose=6)

    mo = mf.mo_coeff[:,:mol.nelectron//2]
    mlo = PipekMezey(mol, mo)

    # Validate gradient and Hessian against finite difference
    g, h_op, hdiag = mlo.gen_g_hop()

    h = numpy.zeros((mlo.pdim,mlo.pdim))
    x0 = mlo.zero_uniq_var()
    for i in range(mlo.pdim):
        x0[i] = 1
        h[:,i] = h_op(x0)
        x0[i] = 0

    def func(x):
        u = mlo.extract_rotation(x)
        f = mlo.cost_function(u)
        if mlo.maximize:
            return -f
        else:
            return f

    def fgrad(x):
        u = mlo.extract_rotation(x)
        return mlo.get_grad(u)

    g_num = findiff_grad(func, x0)
    h_num = findiff_hess(fgrad, x0)
    hdiag_num = numpy.diag(h_num)

    log.info('Grad  error: %.3e', abs(g-g_num).max())
    log.info('Hess  error: %.3e', abs(h-h_num).max())
    log.info('Hdiag error: %.3e', abs(hdiag-hdiag_num).max())

    # localization + stability check using CIAH
    mlo.verbose = 4
    mlo.algorithm = 'ciah'
    mlo.kernel()

    while True:
        mo, stable = mlo.stability(return_status=True)
        if stable:
            break
        mlo.kernel(mo)

    # localization + Jacobi-based stability check using BFGS
    mlo.algorithm = 'bfgs'
    mlo.kernel()

    while True:
        mo, stable = mlo.stability_jacobi(return_status=True)
        if stable:
            break
        mlo.kernel(mo)
