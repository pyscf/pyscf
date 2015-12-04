'''
Non-relativistic Restricted Kohn-Sham for periodic systems at a single k-point 

See Also:
    pyscf.pbc.dft.krks.py : Non-relativistic Restricted Kohn-Sham for periodic
                            systems with k-point sampling
'''

import time
import numpy
import pyscf.dft
import pyscf.pbc.scf
from pyscf.lib import logger
from pyscf.pbc.dft import numint


def get_veff_(ks, cell, dm, dm_last=0, vhf_last=0, hermi=1, 
              kpt=None, kpt_band=None):
    '''Coulomb + XC functional

    .. note::
        This is a replica of pyscf.dft.rks.get_veff_ with kpts added.
        This function will change the ks object.

    Args:
        ks : an instance of :class:`RKS`
            XC functional are controlled by ks.xc attribute.  Attribute
            ks.grids might be initialized.  The ._exc and ._ecoul attributes
            will be updated after return.  Attributes ._dm_last, ._vj_last and
            ._vk_last might be changed if direct SCF method is applied.
        dm : ndarray or list of ndarrays
            A density matrix or a list of density matrices

    Kwargs:
        dm_last : ndarray or a list of ndarrays or 0
            The density matrix baseline.  If not 0, this function computes the
            increment of HF potential w.r.t. the reference HF potential matrix.
        vhf_last : ndarray or a list of ndarrays or 0
            The reference HF potential matrix.  If vhf_last is not given,
            the function will not call direct_scf and attacalites ._dm_last,
            ._vj_last and ._vk_last will not be updated.
        hermi : int
            Whether J, K matrix is hermitian

            | 0 : no hermitian or symmetric
            | 1 : hermitian
            | 2 : anti-hermitian

    Returns:
        matrix Veff = J + Vxc.  Veff can be a list matrices, if the input
        dm is a list of density matrices.
    '''
    t0 = (time.clock(), time.time())
    if ks.grids.coords is None:
        ks.grids.setup_grids_()
        t0 = logger.timer(ks, 'seting up grids', *t0)

    x_code, c_code = pyscf.dft.vxc.parse_xc_name(ks.xc)
    # Note: We always have _numint
    n, ks._exc, vx = ks._numint.nr_rks(cell, ks.grids, x_code, c_code, dm, 
                                       kpt=kpt, kpt_band=kpt_band)
    logger.debug(ks, 'nelec by numeric integration = %s', n)
    t0 = logger.timer(ks, 'vxc', *t0)

    hyb = pyscf.dft.vxc.hybrid_coeff(x_code, spin=(cell.spin>0)+1)

    if abs(hyb) < 1e-10:
        vj = ks.get_j(cell, dm, hermi, kpt=kpt, kpt_band=kpt_band)
    elif (ks._eri is not None or ks._is_mem_enough() or not ks.direct_scf):
        vj, vk = ks.get_jk(cell, dm, hermi, kpt=kpt, kpt_band=kpt_band)
    else:
        if (ks.direct_scf and isinstance(vhf_last, numpy.ndarray) and
            hasattr(ks, '_dm_last')):
            ddm = numpy.asarray(dm) - numpy.asarray(ks._dm_last)
            vj, vk = ks.get_jk(cell, ddm, hermi=hermi, kpt=kpt, kpt_band=kpt_band)
            vj += ks._vj_last
            vk += ks._vk_last
        else:
            vj, vk = ks.get_jk(cell, dm, hermi, kpt=kpt, kpt_band=kpt_band)
        ks._dm_last = dm
        ks._vj_last, ks._vk_last = vj, vk

    if abs(hyb) > 1e-10:
        if isinstance(dm, numpy.ndarray) and kpt_band is None:
            if dm.ndim == 2:
                ks._exc -= (numpy.einsum('ij,ji', dm, vk) * .5 * hyb*.5).real
            else:
                ks._exc -= ((1./len(dm)) * numpy.einsum('Kij,Kji', dm, vk) * .5 * hyb*.5).real

        vhf = vj - vk * (hyb * .5)
    else:
        vhf = vj

    if isinstance(dm, numpy.ndarray) and kpt_band is None:
        if dm.ndim == 2:
            ks._ecoul = (numpy.einsum('ij,ji', dm, vj) * .5).real
        else:
            ks._ecoul = ((1./len(dm)) * numpy.einsum('Kij,Kji', dm, vj) * .5).real

    return vhf + vx


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

    def get_veff(self, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1,
                 kpt=None, kpt_band=None):
        if cell is None: cell = self.cell
        if dm is None: dm = self.make_rdm1()
        if kpt is None: kpt = self.kpt
        return get_veff_(self, cell, dm, dm_last, vhf_last, hermi, kpt, kpt_band)

    def energy_elec(self, dm, h1e=None, vhf=None):
        if h1e is None: h1e = pyscf.pbc.scf.hf.get_hcore(self, self.cell)
        return pyscf.dft.rks.energy_elec(self, dm, h1e)

