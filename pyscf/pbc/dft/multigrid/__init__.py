from .multigrid import MultiGridFFTDF
from .multigrid import (
    multigrid_fftdf as multigrid_fftdf,
    nr_rks as nr_rks_v1,
    nr_rks_fxc as nr_rks_fxc,
    nr_rks_fxc_st as nr_rks_fxc_st,
    nr_uks as nr_uks,
    nr_uks_fxc as nr_uks_fxc
)

from .multigrid_pair import MultiGridFFTDF2
from .multigrid_pair import nr_rks as nr_rks_v2

def nr_rks(mydf, xc_code, dm_kpts, hermi=1, kpts=None,
           kpts_band=None, with_j=False, return_j=False, verbose=None):
    if isinstance(mydf, MultiGridFFTDF2):
        return nr_rks_v2(mydf, xc_code, dm_kpts, hermi=hermi, kpts=kpts,
                         kpts_band=kpts_band, with_j=with_j,
                         return_j=return_j, verbose=verbose) 
    elif isinstance(mydf, MultiGridFFTDF):
        return nr_rks_v1(mydf, xc_code, dm_kpts, hermi=hermi, kpts=kpts,
                         kpts_band=kpts_band, with_j=with_j,
                         return_j=return_j, verbose=verbose)
    else:
        raise TypeError("Wrong density fitting type for multigrid DFT.")
