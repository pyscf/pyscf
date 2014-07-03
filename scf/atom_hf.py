#
# File: atom_hf.py
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import numpy
import scipy.linalg.flapack as lapack
from pyscf import gto
import pyscf.lib.logger as log
import pyscf.lib.parameters as param
import hf

class AtomSphericAverageRHF(hf.RHF):
    def __init__(self, mol):
        hf.SCF.__init__(self, mol)

        if mol.num_NR_function() < 350:
            self.eri_in_memory = True
        else:
            self.eri_in_memory = False
        self._eri = None

    def dump_scf_option(self):
        hf.RHF.dump_scf_option(self)
        log.debug(self.mol, 'occupation averaged SCF for atom  %s', \
                  self.mol.symbol_of_atm(0))

    def eig(self, f, s):
        atm = self.mol
        symb = atm.symbol_of_atm(0)
        nuc = gto.mole._charge(symb)
        idx_by_l = [[] for i in range(6)]
        i0 = 0
        for ib in range(atm.nbas):
            l = atm.angular_of_bas(ib)
            nc = atm.nctr_of_bas(ib)
            i1 = i0 + nc * (l*2+1)
            idx_by_l[l].extend(range(i0, i1, l*2+1))
            i0 = i1

        nbf = atm.num_NR_function()
        self._occ = numpy.zeros(nbf)
        mo_c = numpy.zeros((nbf, nbf))
        mo_e = numpy.zeros(nbf)

        # fraction occupation
        for l in range(4):
            if idx_by_l[l]:
                ne = param.ELEMENTS[nuc][2][l]
                if ne > 0:
                    nd = (l * 2 + 1) * 2
                    n2occ = ne.__floordiv__(nd)
                    frac = (float(ne) / nd - n2occ) * 2
                else:
                    n2occ = frac = 0
                log.debug(self, 'l = %d, occ = %d + %.4g', l, n2occ, frac)

                idx = numpy.array(idx_by_l[l])
                f1 = f[idx,:][:,idx]
                s1 = s[idx,:][:,idx]
                c, e, info = lapack.dsygv(f1, s1)
                for i, ei in enumerate(e):
                    log.debug(self, 'l = %d, e_%d = %.9g', l, i, ei)

                for m in range(l*2+1):
                    mo_e[idx] = e
                    self._occ[idx[:n2occ]] = 2
                    if frac > 1e-15:
                        self._occ[idx[n2occ]] = frac
                    for i,i1 in enumerate(idx):
                        mo_c[idx,i1] = c[:,i]
                    idx += 1
        return mo_e, mo_c, 0

    def set_mo_occ(self, mo_energy, mo_coeff):
        return self._occ

    def calc_den_mat(self, mo_coeff, mo_occ):
        mo = mo_coeff[:,mo_occ>0]
        return numpy.dot(mo*mo_occ[mo_occ>0], mo.T)

    def scf_cycle(self, mol, *args, **keys):
        return hf.scf_cycle(mol, self, *args, dump_chk=False, **keys)

def get_atm_nrhf_result(mol):
    atm_scf_result = {}
    for a, b in mol.basis.items():
        atm = gto.Mole()
        atm.fout = mol.fout
        atm.atom = [[a, (0, 0, 0)]]
        atm.basis = {a: b}
        atm.nelectron = gto.mole._charge(a)
        atm.make_env()
        atm_hf = AtomSphericAverageRHF(atm)
        atm_hf.verbose = 0
        atm_scf_result[a] = atm_hf.scf_cycle(atm)[1:]
    mol.fout.flush()
    return atm_scf_result


if __name__ == '__main__':
    mol = gto.Mole()
    mol.verbose = 5
    mol.output = None

    mol.atom = [["N", (0. , 0., .5)],
                ["N", (0. , 0.,-.5)] ]

    mol.basis = {"N": '6-31g'}
    mol.build()
    print get_atm_nrhf_result(mol)
