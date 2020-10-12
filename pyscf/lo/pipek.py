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
Pipek-Mezey localization

ref. JCTC, 10, 642 (2014); DOI:10.1021/ct401016x
'''

import numpy
from functools import reduce

from pyscf import lib
from pyscf.lib import logger
from pyscf.lo import orth
from pyscf.lo import boys
from pyscf import __config__

def atomic_pops(mol, mo_coeff, method='meta_lowdin', mf=None):
    '''
    Kwargs:
        method : string
            The atomic population projection scheme. It can be mulliken,
            lowdin, meta_lowdin, iao, or becke

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
    proj = numpy.empty((mol.natm,nmo,nmo))

    if getattr(mol, 'pbc_intor', None):  # whether mol object is a cell
        s = mol.pbc_intor('int1e_ovlp_sph', hermi=1)
    else:
        s = mol.intor_symmetric('int1e_ovlp')

    if method == 'becke':
        from pyscf.dft import gen_grid
        if not (getattr(mf, 'grids', None) and getattr(mf, '_numint', None)):
            # Call DFT to initialize grids and numint objects
            mf = mol.RKS()
        grids = mf.grids
        ni = mf._numint

        if not isinstance(grids, gen_grid.Grids):
            raise NotImplementedError('PM becke scheme for PBC systems')

        # The atom-wise Becke grids (without concatenated to a vector of grids)
        coords, weights = grids.get_partition(mol, concat=False)

        for i in range(mol.natm):
            ao = ni.eval_ao(mol, coords[i], deriv=0)
            aow = numpy.einsum('pi,p->pi', ao, weights[i])
            charge_matrix = lib.dot(aow.conj().T, ao)
            proj[i] = reduce(lib.dot, (mo_coeff.conj().T, charge_matrix, mo_coeff))

    elif method == 'mulliken':
        for i, (b0, b1, p0, p1) in enumerate(mol.offset_nr_by_atom()):
            csc = reduce(numpy.dot, (mo_coeff[p0:p1].conj().T, s[p0:p1], mo_coeff))
            proj[i] = (csc + csc.conj().T) * .5

    elif method in ('lowdin', 'meta-lowdin'):
        c = orth.restore_ao_character(mol, 'ANO')
        #csc = reduce(lib.dot, (mo_coeff.conj().T, s, orth_local_ao_coeff))
        csc = reduce(lib.dot, (mo_coeff.conj().T, s, orth.orth_ao(mol, method, c, s=s)))
        for i, (b0, b1, p0, p1) in enumerate(mol.offset_nr_by_atom()):
            proj[i] = numpy.dot(csc[:,p0:p1], csc[:,p0:p1].conj().T)

    elif method in ('iao', 'ibo'):
        from pyscf.lo import iao
        assert mf is not None
        # FIXME: How to handle UHF/UKS object?
        orb_occ = mf.mo_coeff[:,mf.mo_occ>0]

        iao_coeff = iao.iao(mol, orb_occ)
        #
        # IAO is generally not orthogonalized. For simplicity, we take Lowdin
        # orthogonalization here. Other orthogonalization can be used. Results
        # should be very closed to the Lowdin-orth orbitals
        #
        # PM with Mulliken population of non-orth IAOs can be found in
        # ibo.PipekMezey function
        #
        iao_coeff = orth.vec_lowdin(iao_coeff, s)
        csc = reduce(lib.dot, (mo_coeff.conj().T, s, iao_coeff))

        iao_mol = iao.reference_mol(mol)
        for i, (b0, b1, p0, p1) in enumerate(iao_mol.offset_nr_by_atom()):
            proj[i] = numpy.dot(csc[:,p0:p1], csc[:,p0:p1].conj().T)

    else:
        raise KeyError('method = %s' % method)

    return proj


class PipekMezey(boys.Boys):
    '''
    The Pipek-Mezey localization optimizer that maximizes the orbital
    population

    Args:
        mol : Mole object

    Kwargs:
        mo_coeff : size (N,N) np.array
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
            The step size for orbital rotation.  Small step (0.005 - 0.05) is prefered.
            Default 0.03.
        init_guess : str or None
            Initial guess for optimization. If set to None, orbitals defined
            by the attribute .mo_coeff will be used as initial guess. If set
            to 'atomic', atomic orbitals will be used as initial guess.
            Default 'atomic'
        pop_method : str
            How the orbital population is calculated. By default, meta-lowdin
            population (JCTC, 10, 3784) is used. It can be set to 'mulliken',
            or 'lowdin' for other population definition
        exponent : int
            The power to define norm. It can be 2 or 4. Default 2.

    Saved results

        mo_coeff : ndarray
            Localized orbitals

    '''


    pop_method = getattr(__config__, 'lo_pipek_PM_pop_method', 'meta_lowdin')
    conv_tol = getattr(__config__, 'lo_pipek_PM_conv_tol', 1e-6)
    exponent = getattr(__config__, 'lo_pipek_PM_exponent', 2)  # should be 2 or 4

    def __init__(self, mol, mo_coeff=None, mf=None):
        boys.Boys.__init__(self, mol, mo_coeff)
        self._scf = mf
        self._keys = self._keys.union(['pop_method', 'exponent', '_scf'])

    def dump_flags(self, verbose=None):
        boys.Boys.dump_flags(self, verbose)
        logger.info(self, 'pop_method = %s',self.pop_method)

    def gen_g_hop(self, u):
        mo_coeff = lib.dot(self.mo_coeff, u)
        pop = self.atomic_pops(self.mol, mo_coeff, self.pop_method)
        if self.exponent == 2:
            g0 = numpy.einsum('xii,xip->pi', pop, pop)
            g = -self.pack_uniq_var(g0-g0.conj().T) * 2
        elif self.exponent == 4:
            pop3 = numpy.einsum('xii->xi', pop)**3
            g0 = numpy.einsum('xi,xip->pi', pop3, pop)
            g = -self.pack_uniq_var(g0-g0.conj().T) * 4
        else:
            raise NotImplementedError('exponent %s' % self.exponent)

        h_diag = numpy.einsum('xii,xpp->pi', pop, pop) * 2
        g_diag = g0.diagonal()
        h_diag-= g_diag + g_diag.reshape(-1,1)
        h_diag+= numpy.einsum('xip,xip->pi', pop, pop) * 2
        h_diag+= numpy.einsum('xip,xpi->pi', pop, pop) * 2
        h_diag = -self.pack_uniq_var(h_diag) * 2

        g0 = g0 + g0.conj().T
        if self.exponent == 2:
            def h_op(x):
                x = self.unpack_uniq_var(x)
                norb = x.shape[0]
                hx = lib.dot(x.T, g0.T).conj()
                hx+= numpy.einsum('xip,xi->pi', pop, numpy.einsum('qi,xiq->xi', x, pop)) * 2
                hx-= numpy.einsum('xpp,xip->pi', pop,
                                  lib.dot(pop.reshape(-1,norb), x).reshape(-1,norb,norb)) * 2
                hx-= numpy.einsum('xip,xp->pi', pop, numpy.einsum('qp,xpq->xp', x, pop)) * 2
                return -self.pack_uniq_var(hx-hx.conj().T)
        else:
            def h_op(x):
                x = self.unpack_uniq_var(x)
                norb = x.shape[0]
                hx = lib.dot(x.T, g0.T).conj() * 2
                pop2 = numpy.einsum('xii->xi', pop)**2
                pop3 = numpy.einsum('xii->xi', pop)**3
                tmp = numpy.einsum('qi,xiq->xi', x, pop) * pop2
                hx+= numpy.einsum('xip,xi->pi', pop, tmp) * 12
                hx-= numpy.einsum('xp,xip->pi', pop3,
                                  lib.dot(pop.reshape(-1,norb), x).reshape(-1,norb,norb)) * 4
                tmp = numpy.einsum('qp,xpq->xp', x, pop) * pop2
                hx-= numpy.einsum('xip,xp->pi', pop, tmp) * 12
                return -self.pack_uniq_var(hx-hx.conj().T)

        return g, h_op, h_diag

    def get_grad(self, u=None):
        if u is None: u = numpy.eye(self.mo_coeff.shape[1])
        mo_coeff = lib.dot(self.mo_coeff, u)
        pop = self.atomic_pops(self.mol, mo_coeff, self.pop_method)
        if self.exponent == 2:
            g0 = numpy.einsum('xii,xip->pi', pop, pop)
            g = -self.pack_uniq_var(g0-g0.conj().T) * 2
        else:
            pop3 = numpy.einsum('xii->xi', pop)**3
            g0 = numpy.einsum('xi,xip->pi', pop3, pop)
            g = -self.pack_uniq_var(g0-g0.conj().T) * 4
        return g

    def cost_function(self, u=None):
        if u is None: u = numpy.eye(self.mo_coeff.shape[1])
        mo_coeff = lib.dot(self.mo_coeff, u)
        pop = self.atomic_pops(self.mol, mo_coeff, self.pop_method)
        if self.exponent == 2:
            return numpy.einsum('xii,xii->', pop, pop)
        else:
            pop2 = numpy.einsum('xii->xi', pop)**2
            return numpy.einsum('xi,xi', pop2, pop2)

    @lib.with_doc(atomic_pops.__doc__)
    def atomic_pops(self, mol, mo_coeff, method=None):
        if method is None:
            method = self.pop_method

        if method.lower() in ('iao', 'ibo') and self._scf is None:
            logger.error(self, 'PM with IAO scheme should include an scf '
                         'object when creating PM object.\n    PM(mol, mf=scf_object)')
            raise ValueError('PM attribute method is not valid')
        return atomic_pops(mol, mo_coeff, method, self._scf)

PM = Pipek = PipekMezey

if __name__ == '__main__':
    from pyscf import gto, scf

    mol = gto.Mole()
    mol.atom = '''
         O   0.    0.     0.2
         H    0.   -0.5   -0.4
         H    0.    0.5   -0.4
      '''
    mol.basis = 'ccpvdz'
    mol.build()
    mf = scf.RHF(mol).run()

    mo = PM(mol).kernel(mf.mo_coeff[:,5:9], verbose=4)
