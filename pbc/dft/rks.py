'''
Non-relativistic Restricted Kohn-Sham for periodic systems at a single k-point 

See Also:
    pyscf.pbc.dft.krks.py : Non-relativistic Restricted Kohn-Sham for periodic
                            systems with k-point sampling
'''

import numpy
import pyscf.dft
import pyscf.pbc.scf
from pyscf.lib import logger
from pyscf.pbc.dft import numint
import pyscf.pbc.tools as tools


class RKS(pyscf.pbc.scf.hf.RHF):
    '''RKS class adapted for PBCs. 
    
    This is a literal duplication of the molecular RKS class with some `mol`
    variables replaced by `cell`.

    '''
    def __init__(self, cell, kpt=None):
        pyscf.pbc.scf.hf.RHF.__init__(self, cell, kpt)
        self.xc = 'LDA,VWN'
        self._ecoul = 0
        self._exc = 0
#FIXME (Q): lazy create self._numint, since self.kpt might be changed
        # self.kpt is set in RHF.__init__()
        self._numint = numint._NumInt(self.kpt)
        self._keys = self._keys.union(['xc', 'grids'])

    def dump_flags(self):
        pyscf.pbc.scf.hf.RHF.dump_flags(self)
        logger.info(self, 'XC functionals = %s', self.xc)

    def get_veff(self, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        if cell is None: cell = self.cell
        if dm is None: dm = self.make_rdm1()
        return pyscf.dft.rks.get_veff_(self, cell, dm, dm_last, vhf_last, hermi)

    def energy_elec(self, dm, h1e=None, vhf=None):
        if h1e is None: h1e = pyscf.pbc.scf.hf.get_hcore(self, self.cell)
        return pyscf.dft.rks.energy_elec(self, dm, h1e)

    def get_veff_band(self, cell=None, dm=None, kpt=None):
        '''Get veff at a given (arbitrary) 'band' k-point.

        Returns:
            veff : (nao, nao) ndarray
        '''
        if cell is None: cell = self.cell
        if dm is None: dm = self.make_rdm1()
        # This would just return regular get_veff():
        if kpt is None: kpt = self.kpt

        coords, weights = self.grids.setup_grids_()
        ngs = len(weights)
        aoR = self._numint.eval_ao(cell, coords)
        rho = self._numint.eval_rho(cell, aoR, dm)

        # First we have to construct J using the density, not the dm
        coulG = tools.get_coulG(cell)
        rhoG = tools.fft(rho, cell.gs)
        vG = coulG*rhoG
        vR = tools.ifft(vG, cell.gs)

        # Don't use self._numint's eval_ao, because the kpt is wrong 
        aokR = pyscf.pbc.dft.numint.eval_ao(cell, coords, kpt=kpt)
        vj = (cell.vol/ngs) * numpy.dot(aokR.T.conj(), vR.reshape(-1,1)*aokR)

        # TODO: Fix sigma for GGAs
        sigma = numpy.zeros(ngs)
        x_id, c_id = pyscf.dft.vxc.parse_xc_name(self.xc)
        ec, vrho, vsigma = self._numint.eval_xc(x_id, c_id, rho, sigma)
        vxc = self._numint.eval_mat(cell, aokR, weights, rho, vrho) 

        return vj + vxc

    def get_band_fock_ovlp(self, cell=None, dm=None, band_kpt=None):
        '''Reconstruct Fock operator at a given (arbitrary) 'band' k-point.

        Returns:
            fock : (nao, nao) ndarray
            ovlp : (nao, nao) ndarray
        '''
        if cell is None: cell = self.cell
        if dm is None: dm = self.make_rdm1()
        # This would just return regular Fock and overlap matrices:
        if band_kpt is None: band_kpt = self.kpt

        fock = self.get_hcore(kpt=band_kpt) + self.get_veff_band(kpt=band_kpt)
        ovlp = self.get_ovlp(kpt=band_kpt)

        return fock, ovlp
