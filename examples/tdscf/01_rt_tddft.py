import numpy as np
import sys, re
import pyscf
import pyscf.dft
from  pyscf import gto, tdscf
np.set_printoptions(linewidth=220, suppress = True,precision = 7)

def TestTDDFT():
    """
    Tests Basic Propagation Functionality. TDDFT
    """
    prm = '''
    Model	TDDFT
    Method	MMUT
    dt	0.02
    MaxIter	100
    ExDir	1.0
    EyDir	1.0
    EzDir	1.0
    FieldAmplitude	0.01
    FieldFreq	0.9202
    ApplyImpulse	1
    ApplyCw		0
    StatusEvery	10
    '''
    geom = """
    H 0. 0. 0.
    H 0. 0. 0.9
    H 2.0 0.  0
    H 2.0 0.9 0
    """
    output = re.sub("py","dat",sys.argv[0])
    mol = gto.Mole()
    mol.atom = geom
    mol.basis = 'sto-3g'
    mol.build()
    the_scf = pyscf.dft.RKS(mol)
    the_scf.xc='PBE0'
    print "Inital SCF finished. E=", the_scf.kernel()
    aprop = tdscf.tdscf(the_scf,prm,output)
    return
TestTDDFT()
