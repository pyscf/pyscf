from pyscf import gto, scf, mcscf

mol = gto.M(atom="N 0.0 0.0 0.0; N 1.0 0.0 0.0", verbose=4, basis='sto-3g')

hf = scf.HF(mol)
hf.kernel()

cas = mcscf.CASSCF(hf, 4, 2)
cas.max_cycle_macro = 1
cas.kernel()
