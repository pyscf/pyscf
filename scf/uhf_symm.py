#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import time
from functools import reduce
import numpy
import scipy.linalg
import pyscf.lib
from pyscf.lib import logger
from pyscf import symm
from pyscf.scf import hf
from pyscf.scf import hf_symm
from pyscf.scf import uhf
from pyscf.scf import chkfile


def analyze(mf, verbose=logger.DEBUG):
    from pyscf.tools import dump_mat
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    mo_coeff = mf.mo_coeff
    mol = mf.mol
    log = pyscf.lib.logger.Logger(mf.stdout, verbose)
    nirrep = len(mol.irrep_id)
    ovlp_ao = mf.get_ovlp()
    orbsyma = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb,
                                  mo_coeff[0], s=ovlp_ao)
    orbsymb = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb,
                                  mo_coeff[1], s=ovlp_ao)
    orbsyma = numpy.array(orbsyma)
    orbsymb = numpy.array(orbsymb)
    tot_sym = 0
    noccsa = [sum(orbsyma[mo_occ[0]>0]==ir) for ir in mol.irrep_id]
    noccsb = [sum(orbsymb[mo_occ[1]>0]==ir) for ir in mol.irrep_id]
    for ir in range(nirrep):
        if (noccsa[ir]+noccsb[ir]) % 2:
            tot_sym ^= mol.irrep_id[ir]
    if mol.groupname in ('Dooh', 'Coov', 'SO3'):
        log.info('TODO: total symmetry for %s', mol.groupname)
    else:
        log.info('total symmetry = %s',
                 symm.irrep_id2name(mol.groupname, tot_sym))
    log.info('alpha occupancy for each irrep:  '+(' %4s'*nirrep),
             *mol.irrep_name)
    log.info('                                 '+(' %4d'*nirrep),
             *noccsa)
    log.info('beta  occupancy for each irrep:  '+(' %4s'*nirrep),
             *mol.irrep_name)
    log.info('                                 '+(' %4d'*nirrep),
             *noccsb)

    ss, s = mf.spin_square((mo_coeff[0][:,mo_occ[0]>0],
                            mo_coeff[1][:,mo_occ[1]>0]), ovlp_ao)
    log.info('multiplicity <S^2> = %.8g  2S+1 = %.8g', ss, s)

    if verbose >= logger.INFO:
        log.info('**** MO energy ****')
        irname_full = {}
        for k,ir in enumerate(mol.irrep_id):
            irname_full[ir] = mol.irrep_name[k]
        irorbcnt = {}
        for k, j in enumerate(orbsyma):
            if j in irorbcnt:
                irorbcnt[j] += 1
            else:
                irorbcnt[j] = 1
            log.info('alpha MO #%d (%s #%d), energy= %.15g occ= %g', \
                     k+1, irname_full[j], irorbcnt[j], mo_energy[0][k], mo_occ[0][k])
        irorbcnt = {}
        for k, j in enumerate(orbsymb):
            if j in irorbcnt:
                irorbcnt[j] += 1
            else:
                irorbcnt[j] = 1
            log.info('beta  MO #%d (%s #%d), energy= %.15g occ= %g', \
                     k+1, irname_full[j], irorbcnt[j], mo_energy[1][k], mo_occ[1][k])

    if mf.verbose >= logger.DEBUG:
        label = mol.spheric_labels(True)
        molabel = []
        irorbcnt = {}
        for k, j in enumerate(orbsyma):
            if j in irorbcnt:
                irorbcnt[j] += 1
            else:
                irorbcnt[j] = 1
            molabel.append('#%-d(%s #%d)' % (k+1, irname_full[j], irorbcnt[j]))
        log.debug(' ** alpha MO coefficients **')
        dump_mat.dump_rec(mol.stdout, mo_coeff[0], label, molabel, start=1)

        molabel = []
        irorbcnt = {}
        for k, j in enumerate(orbsymb):
            if j in irorbcnt:
                irorbcnt[j] += 1
            else:
                irorbcnt[j] = 1
            molabel.append('#%-d(%s #%d)' % (k+1, irname_full[j], irorbcnt[j]))
        log.debug(' ** beta MO coefficients **')
        dump_mat.dump_rec(mol.stdout, mo_coeff[1], label, molabel, start=1)

    dm = mf.make_rdm1(mo_coeff, mo_occ)
    return mf.mulliken_meta(mol, dm, s=ovlp_ao, verbose=log)

def get_irrep_nelec(mol, mo_coeff, mo_occ, s=None):
    '''Alpha/beta electron numbers for each irreducible representation.

    Args:
        mol : an instance of :class:`Mole`
            To provide irrep_id, and spin-adapted basis
        mo_occ : a list of 1D ndarray
            Regular occupancy, without grouping for irreps
        mo_coeff : a list of 2D ndarray
            Regular orbital coefficients, without grouping for irreps

    Returns:
        irrep_nelec : dict
            The number of alpha/beta electrons for each irrep {'ir_name':(int,int), ...}.

    Examples:

    >>> mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='ccpvdz', symmetry=True, charge=1, spin=1, verbose=0)
    >>> mf = scf.UHF(mol)
    >>> mf.scf()
    -75.623975516256721
    >>> scf.uhf_symm.get_irrep_nelec(mol, mf.mo_coeff, mf.mo_occ)
    {'A1': (3, 3), 'A2': (0, 0), 'B1': (1, 1), 'B2': (1, 0)}
    '''
    orbsyma = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb,
                                  mo_coeff[0], s)
    orbsymb = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb,
                                  mo_coeff[1], s)
    orbsyma = numpy.array(orbsyma)
    orbsymb = numpy.array(orbsymb)
    irrep_nelec = dict([(mol.irrep_name[k], (int(sum(mo_occ[0][orbsyma==ir])),
                                             int(sum(mo_occ[1][orbsymb==ir]))))
                        for k, ir in enumerate(mol.irrep_id)])
    return irrep_nelec

def map_rhf_to_uhf(rhf):
    '''Take the settings from RHF object'''
    return uhf.map_rhf_to_uhf(rhf)


class UHF(uhf.UHF):
    __doc__ = uhf.UHF.__doc__ + '''
    Attributes for symmetry allowed UHF:
        irrep_nelec : dict
            Specify the number of alpha/beta electrons for particular irrep
            {'ir_name':(int,int), ...}.
            For the irreps not listed in these dicts, the program will choose the
            occupancy based on the orbital energies.

    Examples:

    >>> mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='ccpvdz', symmetry=True, charge=1, spin=1, verbose=0)
    >>> mf = scf.RHF(mol)
    >>> mf.scf()
    -75.623975516256692
    >>> mf.get_irrep_nelec()
    {'A1': (3, 3), 'A2': (0, 0), 'B1': (1, 1), 'B2': (1, 0)}
    >>> mf.irrep_nelec = {'B1': (1, 0)}
    >>> mf.scf()
    -75.429189192031131
    >>> mf.get_irrep_nelec()
    {'A1': (3, 3), 'A2': (0, 0), 'B1': (1, 0), 'B2': (1, 1)}
    '''
    def __init__(self, mol):
        uhf.UHF.__init__(self, mol)
        # number of electrons for each irreps
        self.irrep_nelec = {}
        self._keys = self._keys.union(['irrep_nelec'])

    def dump_flags(self):
        hf.SCF.dump_flags(self)
        logger.info(self, '%s with symmetry adapted basis',
                    self.__class__.__name__)
        float_irname = []
        fix_na = 0
        fix_nb = 0
        for ir in range(self.mol.symm_orb.__len__()):
            irname = self.mol.irrep_name[ir]
            if irname in self.irrep_nelec:
                fix_na += self.irrep_nelec[irname][0]
                fix_nb += self.irrep_nelec[irname][1]
            else:
                float_irname.append(irname)
        float_irname = set(float_irname)
        if fix_na+fix_nb > 0:
            logger.info(self, 'fix %d electrons in irreps: %s',
                        fix_na+fix_nb, str(self.irrep_nelec.items()))
            if ((fix_na+fix_nb > self.mol.nelectron) or
                (fix_na>self.nelec[0]) or (fix_nb>self.nelec[1]) or
                (fix_na+self.nelec[1]>self.mol.nelectron) or
                (fix_nb+self.nelec[0]>self.mol.nelectron)):
                logger.error(self, 'electron number error in irrep_nelec %s',
                             self.irrep_nelec.items())
                raise ValueError('irrep_nelec')
        if float_irname:
            logger.info(self, '%d free electrons in irreps %s',
                        self.mol.nelectron-fix_na-fix_nb,
                        ' '.join(float_irname))
        elif fix_na+fix_nb != self.mol.nelectron:
            logger.error(self, 'electron number error in irrep_nelec %d',
                         self.irrep_nelec.items())
            raise ValueError('irrep_nelec')

    def build_(self, mol=None):
        for irname in self.irrep_nelec.keys():
            if irname not in self.mol.irrep_name:
                logger.warn(self, '!! No irrep %s', irname)
        return uhf.UHF.build_(self, mol)

    def eig(self, h, s):
        nirrep = self.mol.symm_orb.__len__()
        s = symm.symmetrize_matrix(s, self.mol.symm_orb)
        ha = symm.symmetrize_matrix(h[0], self.mol.symm_orb)
        cs = []
        es = []
        for ir in range(nirrep):
            e, c = hf.SCF.eig(self, ha[ir], s[ir])
            cs.append(c)
            es.append(e)
        ea = numpy.hstack(es)
        ca = hf_symm.so2ao_mo_coeff(self.mol.symm_orb, cs)

        hb = symm.symmetrize_matrix(h[1], self.mol.symm_orb)
        cs = []
        es = []
        for ir in range(nirrep):
            e, c = scipy.linalg.eigh(hb[ir], s[ir])
            cs.append(c)
            es.append(e)
        eb = numpy.hstack(es)
        cb = hf_symm.so2ao_mo_coeff(self.mol.symm_orb, cs)
        return numpy.array((ea,eb)), (ca,cb)

    def get_occ(self, mo_energy, mo_coeff=None):
        ''' We cannot assume default mo_energy value, because the orbital
        energies are sorted after doing SCF.  But in this function, we need
        the orbital energies are grouped by symmetry irreps
        '''
        mol = self.mol
        nirrep = len(mol.symm_orb)
        if mo_coeff is not None:
            ovlp_ao = self.get_ovlp()
            orbsyma = symm.label_orb_symm(self, mol.irrep_id, mol.symm_orb,
                                          mo_coeff[0], ovlp_ao, False)
            orbsymb = symm.label_orb_symm(self, mol.irrep_id, mol.symm_orb,
                                          mo_coeff[1], ovlp_ao, False)
            orbsyma = numpy.asarray(orbsyma)
            orbsymb = numpy.asarray(orbsymb)
        else:
            orbsyma = [numpy.repeat(ir, mol.symm_orb[ir].shape[1])
                       for ir in range(nirrep)]
            orbsyma = orbsymb = numpy.hstack(orbsyma)

        mo_occ = numpy.zeros_like(mo_energy)
        idx_ea_left = []
        idx_eb_left = []
        neleca_fix = nelecb_fix = 0
        for ir in range(nirrep):
            irname = mol.irrep_name[ir]
            ir_idxa = numpy.where(orbsyma == ir)[0]
            ir_idxb = numpy.where(orbsymb == ir)[0]
            if irname in self.irrep_nelec:
                n = self.irrep_nelec[irname][0]
                e_idx = numpy.argsort(mo_energy[0][ir_idxa])
                mo_occ[0][ir_idxa[e_idx[:n]]] = 1
                neleca_fix += n
                n = self.irrep_nelec[irname][1]
                e_idx = numpy.argsort(mo_energy[1][ir_idxb])
                mo_occ[1][ir_idxb[e_idx[:n]]] = 1
                nelecb_fix += n
            else:
                idx_ea_left.append(ir_idxa)
                idx_eb_left.append(ir_idxb)

        neleca_float = self.nelec[0] - neleca_fix
        nelecb_float = self.nelec[1] - nelecb_fix
        assert(neleca_float >= 0)
        assert(nelecb_float >= 0)
        if len(idx_ea_left) > 0:
            idx_ea_left = numpy.hstack(idx_ea_left)
            ea_left = mo_energy[0][idx_ea_left]
            ea_sort = numpy.argsort(ea_left)
            occ_idx = idx_ea_left[ea_sort][:neleca_float]
            mo_occ[0][occ_idx] = 1
        if len(idx_eb_left) > 0:
            idx_eb_left = numpy.hstack(idx_eb_left)
            eb_left = mo_energy[1][idx_eb_left]
            eb_sort = numpy.argsort(eb_left)
            occ_idx = idx_eb_left[eb_sort][:nelecb_float]
            mo_occ[1][occ_idx] = 1

        viridx = (mo_occ[0]==0)
        if self.verbose < logger.INFO or viridx.sum() == 0:
            return mo_occ
        ehomoa = max(mo_energy[0][mo_occ[0]>0 ])
        elumoa = min(mo_energy[0][mo_occ[0]==0])
        ehomob = max(mo_energy[1][mo_occ[1]>0 ])
        elumob = min(mo_energy[1][mo_occ[1]==0])
        noccsa = []
        noccsb = []
        p0 = 0
        for ir in range(nirrep):
            irname = mol.irrep_name[ir]
            ir_idxa = orbsyma == ir
            ir_idxb = orbsymb == ir

            noccsa.append(int(mo_occ[0][ir_idxa].sum()))
            noccsb.append(int(mo_occ[1][ir_idxb].sum()))
            if ehomoa in mo_energy[0][ir_idxa]:
                irhomoa = irname
            if elumoa in mo_energy[0][ir_idxa]:
                irlumoa = irname
            if ehomob in mo_energy[1][ir_idxb]:
                irhomob = irname
            if elumob in mo_energy[1][ir_idxb]:
                irlumob = irname

        logger.info(self, 'alpha HOMO (%s) = %.15g  LUMO (%s) = %.15g',
                    irhomoa, ehomoa, irlumoa, elumoa)
        logger.info(self, 'beta  HOMO (%s) = %.15g  LUMO (%s) = %.15g',
                    irhomob, ehomob, irlumob, elumob)
        if self.verbose >= logger.DEBUG:
            ehomo = max(ehomoa,ehomob)
            elumo = min(elumoa,elumob)
            logger.debug(self, 'alpha irrep_nelec = %s', noccsa)
            logger.debug(self, 'beta  irrep_nelec = %s', noccsb)
            hf_symm._dump_mo_energy(mol, mo_energy[0], mo_occ[0], ehomo, elumo, 'alpha-')
            hf_symm._dump_mo_energy(mol, mo_energy[1], mo_occ[1], ehomo, elumo, 'beta-')

        if mo_coeff is not None:
            ss, s = self.spin_square((mo_coeff[0][:,mo_occ[0]>0],
                                      mo_coeff[1][:,mo_occ[1]>0]), ovlp_ao)
            logger.debug(self, 'multiplicity <S^2> = %.8g  2S+1 = %.8g', ss, s)
        return mo_occ

    def _finalize_(self):
        uhf.UHF._finalize_(self)

        ea = numpy.hstack(self.mo_energy[0])
        eb = numpy.hstack(self.mo_energy[0])
        oa_sort = numpy.argsort(ea[self.mo_occ[0]>0])
        va_sort = numpy.argsort(ea[self.mo_occ[0]==0])
        ob_sort = numpy.argsort(eb[self.mo_occ[1]>0])
        vb_sort = numpy.argsort(eb[self.mo_occ[1]==0])
        self.mo_energy = (numpy.hstack((ea[self.mo_occ[0]>0 ][oa_sort],
                                        ea[self.mo_occ[0]==0][va_sort])),
                          numpy.hstack((eb[self.mo_occ[1]>0 ][ob_sort],
                                        eb[self.mo_occ[1]==0][vb_sort])))
        ca = self.mo_coeff[0]
        cb = self.mo_coeff[1]
        self.mo_coeff = (numpy.hstack((ca[:,self.mo_occ[0]>0 ].take(oa_sort, axis=1),
                                       ca[:,self.mo_occ[0]==0].take(va_sort, axis=1))),
                         numpy.hstack((cb[:,self.mo_occ[1]>0 ].take(ob_sort, axis=1),
                                       cb[:,self.mo_occ[1]==0].take(vb_sort, axis=1))))
        nocc_a = int(self.mo_occ[0].sum())
        nocc_b = int(self.mo_occ[1].sum())
        self.mo_occ[0][:nocc_a] = 1
        self.mo_occ[0][nocc_a:] = 0
        self.mo_occ[1][:nocc_b] = 1
        self.mo_occ[1][nocc_b:] = 0
        if self.chkfile:
            chkfile.dump_scf(self.mol, self.chkfile,
                             self.e_tot, self.mo_energy,
                             self.mo_coeff, self.mo_occ)

    def analyze(self, mo_verbose=logger.DEBUG):
        return analyze(self, mo_verbose)

    def get_irrep_nelec(self, mol=None, mo_coeff=None, mo_occ=None, s=None):
        if mol is None: mol = self.mol
        if mo_occ is None: mo_occ = self.mo_occ
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if s is None: s = self.get_ovlp()
        return get_irrep_nelec(mol, mo_coeff, mo_occ, s)


