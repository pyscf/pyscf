from pyscf.mp import mp2
from pyscf.mp import dfmp2

def MP2(mf):
    if hasattr(mf, 'with_df'):
        return dfmp2.MP2(mf)
    else:
        return mp2.MP2(mf)
