import sys, tempfile, ctypes, time, numpy, h5py
from functools import reduce
from pyscf import lib
from pyscf.lib import logger
from pyscf.mcscf import mc_ao2mo
from pyscf.ao2mo import _ao2mo
from pyscf.ao2mo import outcore, r_outcore
from pyscf import ao2mo



# level = 1: ppaa, papa and vhf, jpc, kpc
# level = 2: ppaa, papa, vhf,  jpc=0, kpc=0
class _ERIS(object):
    def __init__(self, zcasscf, mo, method='incore', level=1):
        mol = zcasscf.mol
        nao, nmo = mo.shape
        ncore = zcasscf.ncore
        ncas = zcasscf.ncas
        nocc = ncore+ncas


        mem_incore, mem_outcore, mem_basic = mc_ao2mo._mem_usage(ncore, ncas, nmo)
        mem_now = lib.current_memory()[0]
        eri = zcasscf._scf._eri
        moc, moa = mo[:,:ncore], mo[:,ncore:nocc]
        if (method == 'incore' and eri is not None and
            (mem_incore+mem_now < zcasscf.max_memory*.9) or
            mol.incore_anyway):
            if eri is None:
                eri = mol.intor('int2e_spinor', aosym='s8')

            self.ppaa = ao2mo.kernel(eri, (mo, mo, moa, moa), 
                                     intor="int2e_spinor")
            #self.papa = ao2mo.kernel(eri, (mo, moa, mo, moa), 
            #intor="int2e_spinor")
            #self.j_pc, self.k_pc, self.ppaa, self.papa = \
                #mc_ao2mo.trans_e1_incore(eri, mo, ncore, ncas)
        else:
            import gc
            gc.collect()
            log = logger.Logger(zcasscf.stdout, zcasscf.verbose)
            self.feri = lib.H5TmpFile()
            max_memory = max(3000, zcasscf.max_memory*.9-mem_now)
            if max_memory < mem_basic:
                log.warn('Calculation needs %d MB memory, over CASSCF.max_memory (%d MB) limit',
                         (mem_basic+mem_now)/.9, zcasscf.max_memory)
            self.ppaa = ao2mo.kernel(mol, (mo, mo, moa, moa), 
                                     intor="int2e_spinor")
            self.papa = ao2mo.kernel(mol, (mo, moa, mo, moa), 
                                     intor="int2e_spinor")
            #self.ppcc = ao2mo.kernel(mol, (mo, mo, moc, moc), 
            #intor="int2e_spinor")
            #self.pcpc = ao2mo.kernel(mol, (mo, moc, mo, moc), 
            #intor="int2e_spinor")

        self.ppaa.shape = (nmo, nmo, ncas, ncas)            
        self.papa.shape = (nmo, ncas, nmo, ncas)            
        #self.ppcc.shape = (nmo, nmo, ncore, ncore)
        #self.pcpc.shape = (nmo, ncore, nmo, ncore)
