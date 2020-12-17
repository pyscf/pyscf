import numpy as np


class ERIs:

    def __init__(self):
        pass






def ao2mo_pbc(mp, mo_coeff):

    mf = mp._scf
    ao2mofn = lambda mo_coeff : mf.with_df.ao2mo(mo_coeff, mf.kpt, compact=False)
    eris = mp._make_eris(mo_coeff, ao2mofn, mp.verbose)
    return eris

def _make_eris(mo_occ, mo_vir):


