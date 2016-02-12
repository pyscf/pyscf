from pyscf.lib import logger
from pyscf.pbc import lib as pbclib
import pyscf.cc
import pyscf.cc.ccsd
import pyscf.pbc.ao2mo

from pyscf.cc.ccsd_eom import CCSD as molCCSD
from pyscf.cc.ccsd_eom import _ERIS

#einsum = np.einsum
einsum = pbclib.einsum

class CCSD(molCCSD):

    def __init__(self, mf, frozen=[], mo_energy=None, mo_coeff=None, mo_occ=None):
        molCCSD.__init__(self, mf, frozen, mo_energy, mo_coeff, mo_occ)

    def dump_flags(self):
        molCCSD.dump_flags(self)
        logger.info(self, '\n')
        logger.info(self, '******** PBC CC flags ********')

    def ao2mo(self, mo_coeff=None):
        return _ERIS(self, mo_coeff, ao2mofn=pyscf.pbc.ao2mo.general)

