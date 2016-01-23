#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#


import sys
import time
from functools import reduce
import numpy
import pyscf.lib
from pyscf.lib import logger
from pyscf.scf import _vhf
from pyscf.scf import rhf_nmr
from pyscf.dft import vxc


class NMR(rhf_nmr.NMR):
    def __init__(self, scf_method):
        rhf_nmr.NMR.__init__(self, scf_method)
        if vxc.is_hybrid_xc(self._scf.xc) is None:
            self.cphf = False

    def make_h10(self, mol=None, dm0=None, gauge_orig=None):
        if mol is None: mol = self.mol
        if dm0 is None: dm0 = self._scf.make_rdm1()
        if gauge_orig is None: gauge_orig = self.gauge_orig

        if gauge_orig is None:
            log = logger.Logger(self.stdout, self.verbose)
            log.debug('First-order GIAO Fock matrix')
            h1 = .5 * mol.intor('cint1e_giao_irjxp_sph', 3)
            h1 += mol.intor_asymmetric('cint1e_ignuc_sph', 3)
            h1 += mol.intor('cint1e_igkin_sph', 3)

            hyb = vxc.hybrid_coeff(self._scf.xc, spin=(mol.spin>0)+1)
            if abs(hyb) > 1e-10:
                vk = _vhf.direct_mapdm('cint2e_ig1_sph',  # (g i,j|k,l)
                                       'a4ij', 'jk->s1il',
                                       dm0, 3, # xyz, 3 components
                                       mol._atm, mol._bas, mol._env)
                h1 -= .5 * hyb * vk
        else:
            mol.set_common_origin_(gauge_orig)
            h1 = .5 * mol.intor('cint1e_cg_irxp_sph', 3)
        pyscf.lib.chkfile.dump(self.chkfile, 'nmr/h1', h1)
        return h1

    def _vind(self, mo1):
        hyb = vxc.hybrid_coeff(self._scf.xc, spin=(mol.spin>0)+1)

        if abs(hyb) > 1e-10:
            mo_coeff = self._scf.mo_coeff
            mo_occ = self._scf.mo_occ
            dm1 = self.make_rdm1_1(mo1, mo_coeff, mo_occ)
            direct_scf_bak, self._scf.direct_scf = self._scf.direct_scf, False
            vj, vk = self._scf.get_jk(self.mol, dm1, hermi=2)
            v_ao = -.5 * hyb * vk
            self._scf.direct_scf = direct_scf_bak
            return rhf_nmr._mat_ao2mo(v_ao, mo_coeff, mo_occ)
        else:
            nocc = (self._scf.mo_occ>0).sum()
            nmo = self._scf.mo_coeff.shape[1]
            return numpy.zeros((3,nmo,nocc))


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import dft
    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None

    mol.atom.extend([
        [1   , (0. , 0. , .917)],
        ['F' , (0. , 0. , 0.)], ])
    mol.basis = {'H': '6-31g',
                 'F': '6-31g',}
    mol.build()

    mf = dft.RKS(mol)
    mf.xc = 'b3lyp'
    mf.kernel()
    nmr = NMR(mf)
    #nmr.cphf = True
    #nmr.gauge_orig = (0,0,0)
    msc = nmr.kernel() # _xx,_yy = 5440.33124, _zz = 483.072452
    print(msc)


