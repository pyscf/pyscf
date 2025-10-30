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
Restricted Open-shell Hartree-Fock
'''

from functools import reduce
import numpy
import pyscf.gto
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import hf
from pyscf.scf import uhf
from pyscf import __config__

WITH_META_LOWDIN = getattr(__config__, 'scf_analyze_with_meta_lowdin', True)
MO_BASE = getattr(__config__, 'MO_BASE', 1)


def init_guess_by_minao(mol):
    dm = hf.init_guess_by_minao(mol)
    dm = numpy.array((dm*.5, dm*.5))
    if hasattr(dm, 'mo_coeff'):
        dm = lib.tag_array(dm, mo_coeff=dm.mo_coeff, mo_occ=dm.mo_occ)
    return dm

def init_guess_by_atom(mol):
    dm = hf.init_guess_by_atom(mol)
    dm = numpy.array((dm*.5, dm*.5))
    if hasattr(dm, 'mo_coeff'):
        dm = lib.tag_array(dm, mo_coeff=dm.mo_coeff, mo_occ=dm.mo_occ)
    return dm

def init_guess_by_sap(mol, sap_basis, **kwargs):
    dm = hf.init_guess_by_sap(mol, sap_basis, **kwargs)
    dm = numpy.array((dm*.5, dm*.5))
    if hasattr(dm, 'mo_coeff'):
        dm = lib.tag_array(dm, mo_coeff=dm.mo_coeff, mo_occ=dm.mo_occ)
    return dm

init_guess_by_huckel = uhf.init_guess_by_huckel
init_guess_by_mod_huckel = uhf.init_guess_by_mod_huckel

def init_guess_by_chkfile(mol, chkfile_name, project=None):
    '''Read SCF chkfile and make the density matrix for ROHF initial guess.

    Kwargs:
        project : None or bool
            Whether to project chkfile's orbitals to the new basis.  Note when
            the geometry of the chkfile and the given molecule are very
            different, this projection can produce very poor initial guess.
            In PES scanning, it is recommended to switch off project.

            If project is set to None, the projection is only applied when the
            basis sets of the chkfile's molecule are different to the basis
            sets of the given molecule (regardless whether the geometry of
            the two molecules are different).  Note the basis sets are
            considered to be different if the two molecules are derived from
            the same molecule with different ordering of atoms.
    '''
    dm = uhf.init_guess_by_chkfile(mol, chkfile_name, project)
    mo_coeff = dm.mo_coeff[0]
    mo_occ = dm.mo_occ[0] + dm.mo_occ[1]
    return lib.tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)

def get_fock(mf, h1e=None, s1e=None, vhf=None, dm=None, cycle=-1, diis=None,
             diis_start_cycle=None, level_shift_factor=None, damp_factor=None,
             fock_last=None):
    '''Build fock matrix based on Roothaan's effective fock.
    See also :func:`get_roothaan_fock`
    '''
    if h1e is None: h1e = mf.get_hcore()
    if s1e is None: s1e = mf.get_ovlp()
    if vhf is None: vhf = mf.get_veff(mf.mol, dm)
    if dm is None: dm = mf.make_rdm1()
    if isinstance(dm, numpy.ndarray) and dm.ndim == 2:
        dm = numpy.array((dm*.5, dm*.5))
# To Get orbital energy in get_occ, we saved alpha and beta fock, because
# Roothaan effective Fock cannot provide correct orbital energy with `eig`
# TODO, check other treatment  J. Chem. Phys. 133, 141102
    focka = h1e + vhf[0]
    fockb = h1e + vhf[1]
    f = get_roothaan_fock((focka,fockb), dm, s1e)
    if cycle < 0 and diis is None:  # Not inside the SCF iteration
        return f

    if diis_start_cycle is None:
        diis_start_cycle = mf.diis_start_cycle
    if level_shift_factor is None:
        level_shift_factor = mf.level_shift
    if damp_factor is None:
        damp_factor = mf.damp

    dm_tot = dm[0] + dm[1]
    if 0 <= cycle < diis_start_cycle-1 and abs(damp_factor) > 1e-4 and fock_last is not None:
        raise NotImplementedError('ROHF Fock-damping')
    if diis and cycle >= diis_start_cycle:
        f = diis.update(s1e, dm_tot, f, mf, h1e, vhf, f_prev=fock_last)
    if abs(level_shift_factor) > 1e-4:
        f = hf.level_shift(s1e, dm_tot*.5, f, level_shift_factor)
    f = lib.tag_array(f, focka=focka, fockb=fockb)
    return f

def get_roothaan_fock(focka_fockb, dma_dmb, s):
    '''Roothaan's effective fock.
    Ref. http://www-theor.ch.cam.ac.uk/people/ross/thesis/node15.html

    ======== ======== ====== =========
    space     closed   open   virtual
    ======== ======== ====== =========
    closed      Fc      Fb     Fc
    open        Fb      Fc     Fa
    virtual     Fc      Fa     Fc
    ======== ======== ====== =========

    where Fc = (Fa + Fb) / 2

    Returns:
        Roothaan effective Fock matrix
    '''
    nao = s.shape[0]
    focka, fockb = focka_fockb
    dma, dmb = dma_dmb
    fc = (focka + fockb) * .5
# Projector for core, open-shell, and virtual
    pc = numpy.dot(dmb, s)
    po = numpy.dot(dma-dmb, s)
    pv = numpy.eye(nao) - numpy.dot(dma, s)
    fock  = reduce(numpy.dot, (pc.conj().T, fc, pc)) * .5
    fock += reduce(numpy.dot, (po.conj().T, fc, po)) * .5
    fock += reduce(numpy.dot, (pv.conj().T, fc, pv)) * .5
    fock += reduce(numpy.dot, (po.conj().T, fockb, pc))
    fock += reduce(numpy.dot, (po.conj().T, focka, pv))
    fock += reduce(numpy.dot, (pv.conj().T, fc, pc))
    fock = fock + fock.conj().T
    fock = lib.tag_array(fock, focka=focka, fockb=fockb)
    return fock


def get_occ(mf, mo_energy=None, mo_coeff=None):
    '''Label the occupancies for each orbital.

    Examples:

    >>> mol = gto.M(atom='H 0 0 0; O 0 0 1.1', spin=1)
    >>> mf = scf.hf.SCF(mol)
    >>> energy = numpy.array([-10., -1., 1, -2., 0, -3])
    >>> mf.get_occ(energy)
    array([2, 2, 0, 2, 1, 2])
    '''

    if mo_energy is None: mo_energy = mf.mo_energy
    if getattr(mo_energy, 'mo_ea', None) is not None:
        mo_ea = mo_energy.mo_ea
        mo_eb = mo_energy.mo_eb
    else:
        mo_ea = mo_eb = mo_energy
    nmo = mo_ea.size
    if getattr(mf, 'nelec', None) is None:
        nelec = mf.mol.nelec
    else:
        nelec = mf.nelec
    if nelec[0] > nelec[1]:
        nocc, ncore = nelec
    else:
        ncore, nocc = nelec
    if nocc > nmo:
        raise RuntimeError('Failed to assign mo_occ. '
                           f'Nocc ({nocc}) > Nmo ({nmo})')
    nopen = nocc - ncore
    mo_occ = _fill_rohf_occ(mo_energy, mo_ea, mo_eb, ncore, nopen)

    if mf.verbose >= logger.INFO and nocc < nmo and ncore > 0:
        ehomo = max(mo_energy[mo_occ> 0])
        elumo = min(mo_energy[mo_occ==0])
        if ehomo+1e-3 > elumo:
            logger.warn(mf, 'HOMO %.15g >= LUMO %.15g', ehomo, elumo)
        else:
            logger.info(mf, '  HOMO = %.15g  LUMO = %.15g', ehomo, elumo)
        if nopen > 0 and mf.verbose >= logger.DEBUG:
            core_idx = mo_occ == 2
            open_idx = mo_occ == 1
            vir_idx = mo_occ == 0
            logger.debug(mf, '                  Roothaan           | alpha              | beta')
            logger.debug(mf, '  Highest 2-occ = %18.15g | %18.15g | %18.15g',
                         max(mo_energy[core_idx]),
                         max(mo_ea[core_idx]), max(mo_eb[core_idx]))
            logger.debug(mf, '  Lowest 0-occ =  %18.15g | %18.15g | %18.15g',
                         min(mo_energy[vir_idx]),
                         min(mo_ea[vir_idx]), min(mo_eb[vir_idx]))
            for i in numpy.where(open_idx)[0]:
                logger.debug(mf, '  1-occ =         %18.15g | %18.15g | %18.15g',
                             mo_energy[i], mo_ea[i], mo_eb[i])

        if mf.verbose >= logger.DEBUG:
            numpy.set_printoptions(threshold=nmo)
            logger.debug(mf, '  Roothaan mo_energy =\n%s', mo_energy)
            logger.debug1(mf, '  alpha mo_energy =\n%s', mo_ea)
            logger.debug1(mf, '  beta  mo_energy =\n%s', mo_eb)
            numpy.set_printoptions(threshold=1000)
    return mo_occ

def _fill_rohf_occ(mo_energy, mo_energy_a, mo_energy_b, ncore, nopen):
    mo_occ = numpy.zeros_like(mo_energy)
    open_idx = []
    core_sort = numpy.argsort(mo_energy)
    core_idx = core_sort[:ncore]
    if nopen > 0:
        open_idx = core_sort[ncore:]
        open_sort = numpy.argsort(mo_energy_a[open_idx])
        open_idx = open_idx[open_sort[:nopen]]
    mo_occ[core_idx] = 2
    mo_occ[open_idx] = 1
    return mo_occ

def get_grad(mo_coeff, mo_occ, fock):
    '''ROHF gradients is the off-diagonal block [co + cv + ov], where
    [ cc co cv ]
    [ oc oo ov ]
    [ vc vo vv ]
    '''
    occidxa = mo_occ > 0
    occidxb = mo_occ == 2
    viridxa = ~occidxa
    viridxb = ~occidxb
    uniq_var_a = viridxa.reshape(-1,1) & occidxa
    uniq_var_b = viridxb.reshape(-1,1) & occidxb

    if getattr(fock, 'focka', None) is not None:
        focka = fock.focka
        fockb = fock.fockb
    elif isinstance(fock, (tuple, list)) or getattr(fock, 'ndim', None) == 3:
        focka, fockb = fock
    else:
        focka = fockb = fock
    focka = mo_coeff.conj().T.dot(focka).dot(mo_coeff)
    fockb = mo_coeff.conj().T.dot(fockb).dot(mo_coeff)

    g = numpy.zeros_like(focka)
    g[uniq_var_a]  = focka[uniq_var_a]
    g[uniq_var_b] += fockb[uniq_var_b]
    return g[uniq_var_a | uniq_var_b]

def make_rdm1(mo_coeff, mo_occ, **kwargs):
    '''One-particle density matrix.  mo_occ is a 1D array, with occupancy 1 or 2.
    '''
    if isinstance(mo_occ, numpy.ndarray) and mo_occ.ndim == 1:
        mo_occa = (mo_occ > 0).astype(numpy.double)
        mo_occb = (mo_occ ==2).astype(numpy.double)
    else:
        mo_occa, mo_occb = mo_occ
    dm_a = numpy.dot(mo_coeff*mo_occa, mo_coeff.conj().T)
    dm_b = numpy.dot(mo_coeff*mo_occb, mo_coeff.conj().T)
    return lib.tag_array((dm_a, dm_b), mo_coeff=mo_coeff, mo_occ=mo_occ)

def energy_elec(mf, dm=None, h1e=None, vhf=None):
    if dm is None: dm = mf.make_rdm1()
    elif isinstance(dm, numpy.ndarray) and dm.ndim == 2:
        dm = numpy.array((dm*.5, dm*.5))
    return uhf.energy_elec(mf, dm, h1e, vhf)

get_veff = uhf.get_veff

def analyze(mf, verbose=logger.DEBUG, with_meta_lowdin=WITH_META_LOWDIN,
            origin=None, **kwargs):
    '''Analyze the given SCF object:  print orbital energies, occupancies;
    print orbital coefficients; Mulliken population analysis
    '''
    from pyscf.lo import orth
    from pyscf.tools import dump_mat
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    mo_coeff = mf.mo_coeff
    log = logger.new_logger(mf, verbose)
    if log.verbose >= logger.NOTE:
        mf.dump_scf_summary(log)

        log.note('**** MO energy ****')
        if getattr(mo_energy, 'mo_ea', None) is not None:
            mo_ea = mo_energy.mo_ea
            mo_eb = mo_energy.mo_eb
            log.note('                Roothaan           | alpha              | beta')
            for i,c in enumerate(mo_occ):
                log.note('MO #%-3d energy= %-18.15g | %-18.15g | %-18.15g occ= %g',
                         i+MO_BASE, mo_energy[i], mo_ea[i], mo_eb[i], c)
        else:
            for i,c in enumerate(mo_occ):
                log.note('MO #%-3d energy= %-18.15g occ= %g',
                         i+MO_BASE, mo_energy[i], c)

    ovlp_ao = mf.get_ovlp()
    if log.verbose >= logger.DEBUG:
        label = mf.mol.ao_labels()
        if with_meta_lowdin:
            log.debug(' ** MO coefficients (expansion on meta-Lowdin AOs) **')
            orth_coeff = orth.orth_ao(mf.mol, 'meta_lowdin', s=ovlp_ao)
            c = reduce(numpy.dot, (orth_coeff.conj().T, ovlp_ao, mo_coeff))
        else:
            log.debug(' ** MO coefficients (expansion on AOs) **')
            c = mo_coeff
        dump_mat.dump_rec(mf.stdout, c, label, start=MO_BASE, **kwargs)
    dm = mf.make_rdm1(mo_coeff, mo_occ)
    if with_meta_lowdin:
        pop_and_charge = mf.mulliken_meta(mf.mol, dm, s=ovlp_ao, verbose=log)
    else:
        pop_and_charge = mf.mulliken_pop(mf.mol, dm, s=ovlp_ao, verbose=log)
    dip = mf.dip_moment(mf.mol, dm, origin=origin, verbose=log)
    return pop_and_charge, dip

mulliken_pop = hf.mulliken_pop
mulliken_meta = hf.mulliken_meta

def canonicalize(mf, mo_coeff, mo_occ, fock=None):
    '''Canonicalization diagonalizes the Fock matrix within occupied, open,
    virtual subspaces separatedly (without change occupancy).
    '''
    if getattr(fock, 'focka', None) is None:
        dm = mf.make_rdm1(mo_coeff, mo_occ)
        fock = mf.get_fock(dm=dm)
    mo_e, mo_coeff = hf.canonicalize(mf, mo_coeff, mo_occ, fock)
    fa, fb = fock.focka, fock.fockb
    mo_ea = numpy.einsum('pi,pi->i', mo_coeff.conj(), fa.dot(mo_coeff)).real
    mo_eb = numpy.einsum('pi,pi->i', mo_coeff.conj(), fb.dot(mo_coeff)).real
    mo_e = lib.tag_array(mo_e, mo_ea=mo_ea, mo_eb=mo_eb)
    return mo_e, mo_coeff

dip_moment = hf.dip_moment


# use UHF init_guess, get_veff, diis, and intermediates such as fock, vhf, dm
# keep mo_energy, mo_coeff, mo_occ as RHF structure

class ROHF(hf.RHF):
    __doc__ = hf.SCF.__doc__

    def __init__(self, mol):
        hf.SCF.__init__(self, mol)
        self.nelec = None

    @property
    def nelec(self):
        if getattr(self, '_nelec', None) is not None:
            return self._nelec
        else:
            return self.mol.nelec
    @nelec.setter
    def nelec(self, x):
        self._nelec = x

    @property
    def nelectron_alpha(self):
        return self.nelec[0]
    @nelectron_alpha.setter
    def nelectron_alpha(self, x):
        logger.warn(self, 'WARN: Attribute .nelectron_alpha is deprecated. '
                    'Set .nelec instead')
        #raise RuntimeError('API updates')
        self.nelec = (x, self.mol.nelectron-x)

    check_sanity = hf.SCF.check_sanity

    def dump_flags(self, verbose=None):
        hf.SCF.dump_flags(self, verbose)
        nelec = self.nelec
        logger.info(self, 'num. doubly occ = %d  num. singly occ = %d',
                    nelec[1], nelec[0]-nelec[1])

    get_init_guess = uhf.UHF.get_init_guess

    def init_guess_by_minao(self, mol=None):
        if mol is None: mol = self.mol
        return init_guess_by_minao(mol)

    def init_guess_by_atom(self, mol=None):
        if mol is None: mol = self.mol
        logger.info(self, 'Initial guess from the superposition of atomic densities.')
        return init_guess_by_atom(mol)

    def init_guess_by_huckel(self, mol=None):
        if mol is None: mol = self.mol
        logger.info(self, 'Initial guess from on-the-fly Huckel, doi:10.1021/acs.jctc.8b01089.')
        return init_guess_by_huckel(mol)

    def init_guess_by_mod_huckel(self, mol=None):
        if mol is None: mol = self.mol
        logger.info(self, '''Initial guess from on-the-fly Huckel, doi:10.1021/acs.jctc.8b01089,
employing the updated GWH rule from doi:10.1021/ja00480a005.''')
        return init_guess_by_mod_huckel(mol)

    def init_guess_by_1e(self, mol=None):
        if mol is None: mol = self.mol
        logger.info(self, 'Initial guess from hcore.')
        h1e = self.get_hcore(mol)
        s1e = self.get_ovlp(mol)
        mo_energy, mo_coeff = self.eig(h1e, s1e)
        mo_occ = self.get_occ(mo_energy, mo_coeff)
        return self.make_rdm1(mo_coeff, mo_occ)

    def init_guess_by_sap(self, mol=None, **kwargs):
        from pyscf.gto.basis import load
        if mol is None: mol = self.mol
        sap_basis = self.sap_basis
        logger.info(self, '''Initial guess from superposition of atomic potentials (doi:10.1021/acs.jctc.8b01089)
This is the Gaussian fit version as described in doi:10.1063/5.0004046.''')
        if isinstance(sap_basis, str):
            atoms = [coord[0] for coord in mol._atom]
            sapbas = {}
            for atom in set(atoms):
                single_element_bs = load(sap_basis, atom)
                if isinstance(single_element_bs, dict):
                    sapbas[atom] = numpy.asarray(single_element_bs[atom][0][1:], dtype=float)
                else:
                    sapbas[atom] = numpy.asarray(single_element_bs[0][1:], dtype=float)
            logger.note(self, f'Found SAP basis {sap_basis.split("/")[-1]}')
        elif isinstance(sap_basis, dict):
            sapbas = {}
            for key in sap_basis:
                sapbas[key] = numpy.asarray(sap_basis[key][0][1:], dtype=float)
        else:
            logger.error(self, 'sap_basis is of an unexpected datatype.')
        return init_guess_by_sap(mol, sap_basis=sapbas, **kwargs)

    def init_guess_by_chkfile(self, chkfile=None, project=None):
        if chkfile is None: chkfile = self.chkfile
        return init_guess_by_chkfile(self.mol, chkfile, project=project)

    get_fock = get_fock
    get_occ = get_occ

    @lib.with_doc(hf.eig.__doc__)
    def eig(self, fock, s):
        e, c = self._eigh(fock, s)
        if getattr(fock, 'focka', None) is not None:
            mo_ea = numpy.einsum('pi,pi->i', c.conj(), fock.focka.dot(c)).real
            mo_eb = numpy.einsum('pi,pi->i', c.conj(), fock.fockb.dot(c)).real
            e = lib.tag_array(e, mo_ea=mo_ea, mo_eb=mo_eb)
        return e, c

    @lib.with_doc(get_grad.__doc__)
    def get_grad(self, mo_coeff, mo_occ, fock=None):
        if fock is None:
            dm1 = self.make_rdm1(mo_coeff, mo_occ)
            fock = self.get_hcore(self.mol) + self.get_veff(self.mol, dm1)
        return get_grad(mo_coeff, mo_occ, fock)

    @lib.with_doc(make_rdm1.__doc__)
    def make_rdm1(self, mo_coeff=None, mo_occ=None, **kwargs):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if mo_occ is None: mo_occ = self.mo_occ
        if self.mol.spin < 0:
            # Flip occupancies of alpha and beta orbitals
            mo_occ = (numpy.asarray(mo_occ == 2, dtype=numpy.double),
                      numpy.asarray(mo_occ > 0, dtype=numpy.double))
        return make_rdm1(mo_coeff, mo_occ, **kwargs)

    energy_elec = energy_elec

    @lib.with_doc(uhf.get_veff.__doc__)
    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        if isinstance(dm, numpy.ndarray) and dm.ndim == 2:
            dm = numpy.array((dm*.5, dm*.5))

        if self._eri is not None or not self.direct_scf:
            if hasattr(dm, 'mo_occ') and numpy.ndim(dm.mo_occ) == 1:
                mo_occa = (dm.mo_occ > 0).astype(numpy.double)
                mo_occb = (dm.mo_occ ==2).astype(numpy.double)
                dm = lib.tag_array(dm, mo_coeff=(dm.mo_coeff,)*2,
                                   mo_occ=(mo_occa,mo_occb))
            vj, vk = self.get_jk(mol, dm, hermi)
            vhf = vj[0] + vj[1] - vk
        else:
            ddm = dm - numpy.asarray(dm_last)
            vj, vk = self.get_jk(mol, ddm, hermi)
            vhf = vj[0] + vj[1] - vk
            vhf += numpy.asarray(vhf_last)
        return vhf

    @lib.with_doc(analyze.__doc__)
    def analyze(self, verbose=None, with_meta_lowdin=WITH_META_LOWDIN,
                **kwargs):
        if verbose is None: verbose = self.verbose
        return analyze(self, verbose, with_meta_lowdin, **kwargs)

    canonicalize = canonicalize

    def spin_square(self, mo_coeff=None, s=None):
        '''Spin square and multiplicity of RHF determinant'''
        neleca, nelecb = self.nelec
        ms = (neleca - nelecb) * .5
        ss = ms * (ms + 1)
        return ss, ms*2+1

    def stability(self,
                  internal=getattr(__config__, 'scf_stability_internal', True),
                  external=getattr(__config__, 'scf_stability_external', False),
                  verbose=None,
                  return_status=False,
                  **kwargs):
        '''
        ROHF/ROKS stability analysis.

        See also pyscf.scf.stability.rohf_stability function.

        Kwargs:
            internal : bool
                Internal stability, within the RHF optimization space.
            external : bool
                External stability. It is not available in current version.
            return_status: bool
                Whether to return `stable_i` and `stable_e`

        Returns:
            If return_status is False (default), the return value includes
            two set of orbitals, which are more close to the stable condition.
            The first corresponds to the internal stability
            and the second corresponds to the external stability.

            Else, another two boolean variables (indicating current status:
            stable or unstable) are returned.
            The first corresponds to the internal stability
            and the second corresponds to the external stability.
        '''
        from pyscf.scf.stability import rohf_stability
        return rohf_stability(self, internal, external, verbose, return_status, **kwargs)

    def nuc_grad_method(self):
        from pyscf.grad import rohf
        return rohf.Gradients(self)

    convert_from_ = hf.RHF.convert_from_

    def to_ks(self, xc='HF'):
        '''Convert to ROKS object.
        '''
        from pyscf import dft
        return self._transfer_attrs_(dft.ROKS(self.mol, xc=xc))

    to_gpu = lib.to_gpu


class HF1e(ROHF):
    scf = hf._hf1e_scf


del (WITH_META_LOWDIN)
