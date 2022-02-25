'''
Modifications to k-point sampled SCF classes,
to enforce \phi(k) = \phi(-k)* symmetry
'''

import numpy as np

from pyscf.scf.uhf import UHF
from pyscf.scf.ghf import GHF
from pyscf.soscf.newton_ah import _CIAH_SOSCF
from pyscf.lib import logger
from pyscf.pbc.lib import kpts_helper

# --- Helper functions:

def _get_kconj_indices(cell, kpts):
    '''Get list of indices of unique k-point k, and indices of conjugated k-points -k.'''
    conj_index = kpts_helper.conj_mapping(cell, kpts)
    skip = np.zeros(len(kpts), dtype=bool)
    unique = []
    conjugated = []
    for k in range(len(kpts)):
        if skip[k]:
            continue
        kconj = conj_index[k]
        skip[k] = skip[kconj] = True
        unique.append(k)
        conjugated.append(kconj)
    assert set(unique + conjugated) == set(range(len(kpts)))
    return unique, conjugated

def _kconj_symmetrize_rdm1(cell, kpts, dm1):
    '''Get (k,-k)-symmetric 1-DM, by arithmetric averaging.'''
    conj_index = kpts_helper.conj_mapping(cell, kpts)
    dm1 = (dm1 + dm1[...,conj_index,:,:].conj()) / 2
    return dm1

def _rdm1_kconj_symmetry_error(cell, kpts, dm1):
    '''Determine (k,-k)-symmetry error from the 1-DM.'''
    kuniq, kconj = _get_kconj_indices(cell, kpts)
    err = abs(dm1[...,kuniq,:,:] - dm1[...,kconj,:,:].conj()).max()
    return err

def kscf_with_kconjsym(cls):
    '''Create SCF class with k-point sampling and phi(k) = phi(-k)* symmetry.

    Args:
        cls: SCF class with k-point sampling.

    Returns:
        cls_sym: SCF class with k-point sampling and \phi(k) = \phi(-k)* symmetry.
    '''
    is_uhf = issubclass(cls, UHF)
    if issubclass(cls, GHF):
        raise NotImplementedError("(k,-k)-symmetry not implemented for GHF.")
    if issubclass(cls, _CIAH_SOSCF):
        raise NotImplementedError("Newton solver with (k,-k)-symmetry not implemented.")
    # TODO: other classes which should be excluded here?

    # --- Replacement methods:

    def get_jk(self, cell=None, dm_kpts=None, hermi=1, kpts=None, kpts_band=None,
            with_j=True, with_k=True, omega=None, **kwargs):
        # If kpts_band is provided, do not use (k,-k)-symmetry:
        if kpts_band is not None:
            return super(type(self), self).get_jk(cell=cell, dm_kpts=dm_kpts, hermi=hermi, kpts=kpts,
                kpts_band=kpts_band, with_j=with_j, with_k=with_k, omega=omega, **kwargs)

        if cell is None: cell = self.cell
        if kpts is None: kpts = self.kpts
        if dm_kpts is None: dm_kpts = self.make_rdm1()
        nkpts = len(kpts)
        nao = dm_kpts.shape[-1]
        # Use kpts_band to evaluate J and K matrix at reduced set of unique k-points:
        kuniq, kconj = _get_kconj_indices(cell, kpts)
        kpts_red = kpts[kuniq]
        vj_red, vk_red = super(type(self), self).get_jk(cell=cell, dm_kpts=dm_kpts, hermi=hermi, kpts=kpts,
                kpts_band=kpts_red, with_j=with_j, with_k=with_k, omega=omega, **kwargs)
        selfconj = np.equal(kuniq, kconj)
        vj = vk = None
        vshape = (2, nkpts, nao, nao) if is_uhf else (nkpts, nao, nao)
        # Use vj_red and vk_red evaluated at the unique k-points k to fill the values
        # for vj and vk at both k and -k:
        if with_j:
            vj_red[...,selfconj,:,:] = vj_red[...,selfconj,:,:].real
            vj = np.zeros(vshape, dtype=complex)
            vj[...,kconj,:,:] = vj_red.conj()
            vj[...,kuniq,:,:] = vj_red
        if with_k:
            vk_red[...,selfconj,:,:] = vk_red[...,selfconj,:,:].real
            vk = np.zeros(vshape, dtype=complex)
            vk[...,kconj,:,:] = vk_red.conj()
            vk[...,kuniq,:,:] = vk_red
        return vj, vk

    if is_uhf:
        def eig(self, h_kpts, s_kpts):
            nkpts = len(s_kpts)
            kuniq, kconj = _get_kconj_indices(self.cell, self.kpts)
            # Only diagonalize unique k-points:
            h_red = h_kpts[:,kuniq]
            s_red = s_kpts[kuniq]
            # Self-conjugated k-points (e.g. Gamma) should have a real Fock matrix:
            selfconj = np.equal(kuniq, kconj)
            h_red[:,selfconj] = h_red[:,selfconj].real
            # Diagonalize reduced set of k-points:
            e_red, c_red = super(type(self), self).eig(h_red, s_red)
            # Use lists and loops instead of NumPy arrays, since the number of MOs can be k-point dependent:
            mo_energy_k = (nkpts*[None], nkpts*[None])
            mo_coeff_k = (nkpts*[None], nkpts*[None])
            for idx, (k, kc) in enumerate(zip(kuniq, kconj)):
                mo_energy_k[0][k] = e_red[0][idx]
                mo_energy_k[1][k] = e_red[1][idx]
                mo_coeff_k[0][k] = c_red[0][idx]
                mo_coeff_k[1][k] = c_red[1][idx]
                if (kc == k):
                    continue
                # Shift conjugated energies slightly, to avoid a possible exact degeneracy
                # between HOMO and LUMO. The exact degeneracy is avoided at the price of
                # breaking the (k,-k)-symmetry via the MO occupations in this case,
                # but it is possible that a real gap opens during the SCF loop,
                # such that the symmetry is restored in the end.
                mo_energy_k[0][kc] = e_red[0][idx] + 1e-15
                mo_energy_k[1][kc] = e_red[1][idx] + 1e-15
                mo_coeff_k[0][kc] = c_red[0][idx].conj()
                mo_coeff_k[1][kc] = c_red[1][idx].conj()

            # Write out (k,-k)-symmetry-error
            mo_occ_k = self.get_occ(mo_energy_kpts=mo_energy_k)
            dm1 = self.make_rdm1(mo_coeff_kpts=mo_coeff_k, mo_occ_kpts=mo_occ_k)
            err = _rdm1_kconj_symmetry_error(self.cell, self.kpts, dm1)
            logger.info(self, "Max (k,-k)-symmetry error in 1-DM: %.3e", err)

            return mo_energy_k, mo_coeff_k

    else:   # RHF
        def eig(self, h_kpts, s_kpts):
            nkpts = len(s_kpts)
            kuniq, kconj = _get_kconj_indices(self.cell, self.kpts)
            # Only diagonalize unique k-points:
            h_red = h_kpts[kuniq]
            s_red = s_kpts[kuniq]
            # Self-conjugated k-points (e.g. Gamma) should have a real Fock matrix:
            selfconj = np.equal(kuniq, kconj)
            h_red[selfconj] = h_red[selfconj].real
            # Diagonalize reduced set of k-points:
            e_red, c_red = super(type(self), self).eig(h_red, s_red)
            # Use lists and loops instead of NumPy arrays, since the number of MOs can be k-point dependent:
            mo_energy_k = nkpts*[None]
            mo_coeff_k = nkpts*[None]
            for idx, (k, kc) in enumerate(zip(kuniq, kconj)):
                mo_energy_k[k] = e_red[idx]
                mo_coeff_k[k] = c_red[idx]
                if (kc == k):
                    continue
                # Shift conjugated energies slightly, to avoid a possible exact degeneracy
                # between HOMO and LUMO. The exact degeneracy is avoided at the price of
                # breaking the (k,-k)-symmetry via the MO occupations in this case,
                # but it is possible that a real gap opens during the SCF loop,
                # such that the symmetry is restored in the end.
                mo_energy_k[kc] = e_red[idx] + 1e-15
                mo_coeff_k[kc] = c_red[idx].conj()

            # Write out (k,-k)-symmetry-error
            mo_occ_k = self.get_occ(mo_energy_kpts=mo_energy_k)
            dm1 = self.make_rdm1(mo_coeff_kpts=mo_coeff_k, mo_occ_kpts=mo_occ_k)
            err = _rdm1_kconj_symmetry_error(self.cell, self.kpts, dm1)
            logger.info(self, "Max (k,-k)-symmetry error in 1-DM: %.3e", err)

            return mo_energy_k, mo_coeff_k

    def get_init_guess(self, *args, **kwargs):
        dm0 = super(type(self), self).get_init_guess(*args, **kwargs)
        dm0 = _kconj_symmetrize_rdm1(self.cell, self.kpts, dm0)
        return dm0

    def kernel(self, dm0=None, *args, **kwargs):
        if dm0 is not None:
            dm0 = _kconj_symmetrize_rdm1(self.cell, self.kpts, dm0)
        res = super(type(self), self).kernel(dm0=dm0, *args, **kwargs)
        # Print a warning to the log, if (k,-k)-symmetry is broken in the final 1-DM:
        dm1 = self.make_rdm1()
        kuniq, kconj = _get_kconj_indices(self.cell, self.kpts)
        err = _rdm1_kconj_symmetry_error(self.cell, self.kpts, dm1)
        if (err > 1e-8):
            logger.warn(self, "(k,-k)-symmetry broken in 1-DM! Max error= %.3e", err)
            self.converged = False
        return res

    def newton(self, *args, **kwargs):
        raise NotImplementedError("Newton solver with (k,-k)-symmetry not implemented.")

    # Dynamically create new class `cls_sym`, inheriting from `cls`:
    cls_sym = type('%s_kConjSym' % cls.__name__, (cls,), {
        'get_jk': get_jk,
        'eig': eig,
        'get_init_guess': get_init_guess,
        'kernel': kernel,
        'newton': newton})
    return cls_sym
