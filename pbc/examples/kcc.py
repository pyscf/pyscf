import numpy as np
from pyscf.pbc import cc as pbccc

def run_kccsd(mf):
    cc = pbccc.KCCSD(mf)
    cc.verbose = 7
    cc.ccsd()
    return cc

def run_krccsd(mf):
    cc = pbccc.KRCCSD(mf)
    cc.verbose = 7
    cc.ccsd()
    return cc

def run_ip_krccsd(cc, nroots=9):
    e,c = cc.ipccsd(nroots)
    return e,c

def run_ea_krccsd(cc, nroots=9):
    e,c = cc.eaccsd(nroots)
    return e,c


if __name__ == '__main__':
    import sys
    from helpers import get_ase_diamond_primitive, build_cell
    from scf import run_khf

    args = sys.argv[1:]
    if len(args) != 6:
        print 'usage: atom basis ke nkx nky nkz'
        sys.exit(1)
    atom = args[0]
    bas = args[1]
    ke = float(args[2])
    nmp = np.array([int(nk) for nk in args[3:6]])
  
    assert atom in ['C','Si']
    ase_atom = get_ase_diamond_primitive(atom=atom) 
    cell = build_cell(ase_atom, ke=ke, basis=bas, incore_anyway=True)

    nmp = nmp
    mf = run_khf(cell, nmp=nmp)
    
    cc = run_krccsd(mf)
    print "KRCCSD E =", cc.ecc

    print "%0.4f %0.8f %0.8f %0.8f"%(
            np.prod(nmp)**(1./3), mf.e_tot, cc.emp2, cc.ecc)

