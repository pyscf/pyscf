from .mindo3 import RMINDO3, UMINDO3

def MINDO3(mol):
    if mol.spin == 0:
        return RMINDO3(mol)
    else:
        return UMINDO3(mol)
