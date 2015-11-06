'''
Non-relativistic Restricted Kohn-Sham for periodic systems with k-point sampling

See Also:
    pyscf.pbc.dft.rks.py : Non-relativistic Restricted Kohn-Sham for periodic
                           systems at a single k-point
'''

import numpy as np
import pyscf.dft
import pyscf.pbc.scf
import pyscf.pbc.scf.khf
import pyscf.pbc.dft
from pyscf.lib import logger
from pyscf.pbc.dft import numint


class KRKS(pyscf.pbc.scf.khf.KRHF):
    '''RKS class adapted for PBCs with k-point sampling.
    '''
    def __init__(self, cell, kpts):
        pyscf.pbc.scf.khf.KRHF.__init__(self, cell, kpts)
        self._numint = numint._KNumInt(kpts) # use periodic images of AO in
                                             # numerical integration
        self.xc = 'LDA,VWN'
        self._ecoul = 0
        self._exc = 0

    def dump_flags(self):
        pyscf.pbc.scf.khf.KRHF.dump_flags(self)
        logger.info(self, 'XC functionals = %s', self.xc)
        self.grids.dump_flags()

    def get_veff(self, cell=None, dm=None, dm_last=None, vhf_last=0, hermi=1):
        '''
        Args:
             See pyscf.pbc.scf.khf.KRHF.get_veff

        Returns:
             vhf : (nkpts, nao, nao) ndarray
                Effective potential corresponding to input density matrix at
                each k-point
        '''
        if cell is None: cell = self.cell
        if dm is None: dm = self.make_rdm1()

        dm = np.array(dm, np.complex128) # e.g. if passed initial DM

        vhf = pyscf.dft.rks.get_veff_(self, cell, dm, dm_last, vhf_last,
                                      hermi)
        return vhf

    def energy_elec(self, dm_kpts=None, h1e_kpts=None, vhf=None):

        if h1e_kpts is None: h1e_kpts = self.get_hcore(self.cell, self.kpts)
        if dm_kpts is None: dm_kpts = self.make_rdm1()

        nkpts = len(dm_kpts)
        e1 = 0.
        for k in range(nkpts):
            e1 += 1./nkpts*np.einsum('ij,ji', h1e_kpts[k,:,:], dm_kpts[k,:,:]).real

        tot_e = e1 + self._ecoul + self._exc
        logger.debug(self, 'E1 = %s  Ecoul = %s  Exc = %s', e1, self._ecoul, self._exc)
        return tot_e, self._ecoul + self._exc

