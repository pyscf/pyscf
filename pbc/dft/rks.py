'''
Non-relativistic Restricted Kohn-Sham for periodic systems at a single k-point 

See Also:
    pyscf.pbc.dft.krks.py : Non-relativistic Restricted Kohn-Sham for periodic
                            systems with k-point sampling
'''

import time
import numpy
import pyscf.lib
import pyscf.dft
import pyscf.pbc.scf
from pyscf.lib import logger
from pyscf.pbc.dft import numint


def get_veff(ks, cell, dm, dm_last=0, vhf_last=0, hermi=1,
             kpt=None, kpt_band=None):
    '''Coulomb + XC functional

    .. note::
        This is a replica of pyscf.dft.rks.get_veff with kpts added.
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
        ks.grids.build()
        small_rho_cutoff = ks.small_rho_cutoff
        t0 = logger.timer(ks, 'setting up grids', *t0)
    else:
        small_rho_cutoff = 0

    hyb = ks._numint.hybrid_coeff(ks.xc, spin=(cell.spin>0)+1)
    n, ks._exc, vx = ks._numint.nr_rks(cell, ks.grids, ks.xc, dm, kpt=kpt,
                                       kpt_band=kpt_band)
    logger.debug(ks, 'nelec by numeric integration = %s', n)
    t0 = logger.timer(ks, 'vxc', *t0)


    if abs(hyb) < 1e-10:
        if (ks._eri is not None or not ks.direct_scf or
            not hasattr(ks, '_dm_last') or
            not isinstance(vhf_last, numpy.ndarray)):
            vhf = vj = ks.get_j(cell, dm, hermi, kpt=kpt, kpt_band=kpt_band)
        else:
            ddm = numpy.asarray(dm) - numpy.asarray(ks._dm_last)
            vj = ks.get_j(cell, ddm, hermi, kpt=kpt, kpt_band=kpt_band)
            vj += ks._vj_last
            ks._dm_last = dm
            vhf = ks._vj_last = vj
    else:
        if (ks._eri is not None or not ks.direct_scf or
            not hasattr(ks, '_dm_last') or
            not isinstance(vhf_last, numpy.ndarray)):
            vj, vk = ks.get_jk(cell, dm, hermi, kpt=kpt, kpt_band=kpt_band)
        else:
            ddm = numpy.asarray(dm) - numpy.asarray(ks._dm_last)
            vj, vk = ks.get_jk(cell, ddm, hermi, kpt=kpt, kpt_band=kpt_band)
            vj += ks._vj_last
            vk += ks._vk_last
            ks._dm_last = dm
            ks._vj_last, ks._vk_last = vj, vk

        if isinstance(dm, numpy.ndarray) and kpt_band is None:
            if dm.ndim == 2:
                ks._exc -= (numpy.einsum('ij,ji', dm, vk) * .5 * hyb*.5).real
            else:
                ks._exc -= ((1./len(dm)) * numpy.einsum('Kij,Kji', dm, vk) * .5 * hyb*.5).real

    if isinstance(dm, numpy.ndarray) and kpt_band is None:
        if dm.ndim == 2:
            ks._ecoul = (numpy.einsum('ij,ji', dm, vj) * .5).real
        else:
            ks._ecoul = ((1./len(dm)) * numpy.einsum('Kij,Kji', dm, vj) * .5).real

    if small_rho_cutoff > 1e-20:
        # Filter grids the first time setup grids
        idx = ks._numint.large_rho_indices(cell, dm, ks.grids, small_rho_cutoff)
        logger.debug(ks, 'Drop grids %d',
                     ks.grids.weights.size - numpy.count_nonzero(idx))
        ks.grids.coords  = numpy.asarray(ks.grids.coords [idx], order='C')
        ks.grids.weights = numpy.asarray(ks.grids.weights[idx], order='C')
        ks._numint.non0tab = None
    return vhf + vx


class RKS(pyscf.pbc.scf.hf.RHF):
    '''RKS class adapted for PBCs. 
    
    This is a literal duplication of the molecular RKS class with some `mol`
    variables replaced by `cell`.

    '''
    def __init__(self, cell, kpt=numpy.zeros(3)):
        pyscf.pbc.scf.hf.RHF.__init__(self, cell, kpt)
        self.xc = 'LDA,VWN'
        #self.grids = None # initialized in pbc.scf.hf.RHF
        self.small_rho_cutoff = 1e-7  # Use rho to filter grids
##################################################
# don't modify the following attributes, they are not input options
#FIXME (Q): lazy create self._numint, since self.kpt might be changed
        # self.kpt is set in RHF.__init__()
        self._ecoul = 0
        self._exc = 0
        self._numint = numint._NumInt(self.kpt)
        self._keys = self._keys.union(['xc', 'grids', 'small_rho_cutoff'])

    def dump_flags(self):
        pyscf.pbc.scf.hf.RHF.dump_flags(self)
        logger.info(self, 'XC functionals = %s', self.xc)

    @pyscf.lib.with_doc(get_veff.__doc__)
    def get_veff(self, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1,
                 kpt=None, kpt_band=None):
        if cell is None: cell = self.cell
        if dm is None: dm = self.make_rdm1()
        if kpt is None: kpt = self.kpt
        return get_veff(self, cell, dm, dm_last, vhf_last, hermi, kpt, kpt_band)

    def energy_elec(self, dm, h1e=None, vhf=None):
        if h1e is None: h1e = pyscf.pbc.scf.hf.get_hcore(self, self.cell)
        return pyscf.dft.rks.energy_elec(self, dm, h1e)

