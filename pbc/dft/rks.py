'''
Non-relativistic Restricted Kohn-Sham
for periodic systems at a *single* k-point.

See Also:
    kscf.py : SCF tools for periodic systems with k-point *sampling*.
'''
import numpy
import pyscf.dft
import pyscf.pbc.scf
from pyscf.pbc.dft import numint

from pyscf.lib import logger

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
        self._numint = numint._NumInt(self.kpt) # use periodic images of AO in 
                                                # numerical integration
        self._keys = self._keys.union(['xc', 'grids'])

    def dump_flags(self):
        pyscf.pbc.scf.hf.RHF.dump_flags(self)
        logger.info(self, 'XC functionals = %s', self.xc)

    def get_veff(self, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        if cell is None: cell = self.cell
        if dm is None: dm = self.make_rdm1()
        
        # print "HACK VEFF"
        # return numpy.zeros_like(dm)
        return pyscf.dft.rks.get_veff_(self, cell, dm, dm_last, vhf_last, 
                                       hermi)

    def energy_elec(self, dm, h1e=None, vhf=None):
        if h1e is None: h1e = get_hcore(self, self.cell)
        return pyscf.dft.rks.energy_elec(self, dm, h1e)
    
