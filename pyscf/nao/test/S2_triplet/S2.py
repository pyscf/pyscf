
from pyscf.nao import gw as gw_c
import os


dname = os.getcwd()
gw = gw_c(label='S2', cd=dname, verbosity=3, niter_max_ev=70, rescf=True, magnetization=2)
gw.kernel_gw()
gw.report()
