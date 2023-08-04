from pyscf import gto, scf, dft

mol = gto.M(atom="He 0.0 0.0 0.0", verbose=0, basis='6-31g')
mol2 = gto.M(atom="He 0.0 0.0 0.0", verbose=0, basis='6-31g', spin=2)

rhf = scf.RHF(mol).kernel()
print(f"RHF: {rhf}")
uhf = scf.UHF(mol2).kernel()
print(f"UHF: {uhf}")
rohf = scf.RHF(mol2).kernel()
print(f"ROHF: {rohf}")

