from pyscf.lib import logger
from pyscf import lib
import pyscf.cc
import pyscf.cc.ccsd
import pyscf.pbc.ao2mo

from pyscf.cc.uccsd import UCCSD as molCCSD
from pyscf.cc.uccsd import _ERIS

from pyscf.cc.rccsd import RCCSD as molRCCSD
from pyscf.cc.rccsd import _ERIS as _RERIS

#einsum = np.einsum
einsum = lib.einsum

class CCSD(molCCSD):

    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None):
        # ERIs with 2D-PBC have some divergent terms
        if mf.cell.dimension == 2:
            raise NotImplementedError
        molCCSD.__init__(self, mf, frozen, mo_coeff, mo_occ)

    def dump_flags(self):
        molCCSD.dump_flags(self)
        logger.info(self, '\n')
        logger.info(self, '******** PBC CC flags ********')

    def ao2mo(self, mo_coeff=None):
        return _ERIS(self, mo_coeff, ao2mofn=pyscf.pbc.ao2mo.general)

class RCCSD(molRCCSD):

    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None):
        if mf.cell.dimension == 2:
            raise NotImplementedError
        molRCCSD.__init__(self, mf, frozen, mo_coeff, mo_occ)

    def dump_flags(self):
        molRCCSD.dump_flags(self)
        logger.info(self, '\n')
        logger.info(self, '******** PBC CC flags ********')

    def ao2mo(self, mo_coeff=None):
        return _RERIS(self, mo_coeff, ao2mofn=pyscf.pbc.ao2mo.general)

