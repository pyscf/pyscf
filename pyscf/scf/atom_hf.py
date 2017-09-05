#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import numpy
from pyscf import gto
from pyscf.lib import logger
import pyscf.lib.parameters as param
from pyscf.scf import hf


def get_atm_nrhf(mol):
    if mol.has_ecp():
        raise NotImplementedError('Atomic calculation with ECP is not implemented')

    atm_scf_result = {}
    for a, b in mol._basis.items():
        atm = gto.Mole()
        atm.stdout = mol.stdout
        atm.atom = atm._atom = [[a, (0, 0, 0)]]
        atm._basis = {a: b}
        atm.nelectron = gto.mole._charge(a)
        atm.spin = atm.nelectron % 2
        atm._atm, atm._bas, atm._env = \
                atm.make_env(atm._atom, atm._basis, atm._env)
        atm._built = True
        if atm.nelectron == 0:  # GHOST
            nao = atm.nao_nr()
            mo_occ = mo_energy = numpy.zeros(nao)
            mo_coeff = numpy.zeros((nao,nao))
            atm_scf_result[a] = (0, mo_energy, mo_coeff, mo_occ)
        else:
            atm_hf = AtomSphericAverageRHF(atm)
            atm_hf.verbose = 0
            atm_scf_result[a] = atm_hf.scf()[1:]
            atm_hf._eri = None
    mol.stdout.flush()
    return atm_scf_result

class AtomSphericAverageRHF(hf.RHF):
    def __init__(self, mol):
        self._eri = None
        self._occ = None
        hf.SCF.__init__(self, mol)

    def dump_flags(self):
        hf.RHF.dump_flags(self)
        logger.debug(self.mol, 'occupation averaged SCF for atom  %s',
                     self.mol.atom_symbol(0))

    def eig(self, f, s):
        atm = self.mol
        symb = atm.atom_symbol(0)
        idx_by_l = [[] for i in range(param.L_MAX)]
        i0 = 0
        for ib in range(atm.nbas):
            l = atm.bas_angular(ib)
            nc = atm.bas_nctr(ib)
            i1 = i0 + nc * (l*2+1)
            idx_by_l[l].extend(range(i0, i1, l*2+1))
            i0 = i1

        nbf = atm.nao_nr()
        self._occ = numpy.zeros(nbf)
        mo_c = numpy.zeros((nbf, nbf))
        mo_e = numpy.zeros(nbf)

        # fraction occupation
        for l in range(param.L_MAX):
            if idx_by_l[l]:
                n2occ, frac = frac_occ(symb, l)
                logger.debug1(self, 'l = %d  occ = %d + %.4g', l, n2occ, frac)

                idx = numpy.array(idx_by_l[l])
                f1 = 0
                s1 = 0
                for m in range(l*2+1):
                    f1 = f1 + f[idx+m,:][:,idx+m]
                    s1 = s1 + s[idx+m,:][:,idx+m]
                f1 *= 1./(l*2+1)
                s1 *= 1./(l*2+1)
                e, c = self._eigh(f1, s1)
                for i, ei in enumerate(e):
                    logger.debug1(self, 'l = %d  e_%d = %.9g', l, i, ei)

                for m in range(l*2+1):
                    mo_e[idx] = e
                    self._occ[idx[:n2occ]] = 2
                    if frac > 1e-15:
                        self._occ[idx[n2occ]] = frac
                    for i,i1 in enumerate(idx):
                        mo_c[idx,i1] = c[:,i]
                    idx += 1
        return mo_e, mo_c

    def get_occ(self, mo_energy=None, mo_coeff=None):
        return self._occ

    def get_grad(self, mo_coeff, mo_occ, fock=None):
        return 0

    def scf(self, *args, **kwargs):
        self.build()
        return hf.kernel(self, *args, dump_chk=False, **kwargs)

def frac_occ(symb, l):
    nuc = gto.mole._charge(symb)
    if l < 4 and param.ELEMENTS[nuc][2][l] > 0:
        ne = param.ELEMENTS[nuc][2][l]
        nd = (l * 2 + 1) * 2
        ndocc = ne.__floordiv__(nd)
        frac = (float(ne) / nd - ndocc) * 2
    else:
        ndocc = frac = 0
    return ndocc, frac



if __name__ == '__main__':
    mol = gto.Mole()
    mol.verbose = 5
    mol.output = None

    mol.atom = [["N", (0. , 0., .5)],
                ["N", (0. , 0.,-.5)] ]

    mol.basis = {"N": '6-31g'}
    mol.build()
    print(get_atm_nrhf(mol))
