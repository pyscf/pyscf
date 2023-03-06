from pyscf import gto, scf
import multiprocessing



def nelec(mol):
    return mol.nelectron


with multiprocessing.Pool(2) as pool:
    print(pool.map(nelec, [gto.M(atom='H 0.0 0.0 0.0; H 1.0 0.0 0.0'), gto.M(atom='He 0.0 0.0 0.0; He 1.0 0.0 0.0')]))


