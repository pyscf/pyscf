import numpy as np
from pyscf.pbc import cc as pbccc

def run_ccsd(mf):
    cc = pbccc.CCSD(mf)
    cc.verbose = 7
    cc.ccsd()
    return cc

def run_ipccsd(cc, nroots=4):
    e,c = cc.ipccsd(nroots)
    return e

def run_eaccsd(cc, nroots=4):
    e,c = cc.eaccsd(nroots)
    return e

def run_eeccsd(cc, nroots=4):
    e,c = cc.eeccsd(nroots)
    return e


if __name__ == '__main__':
    from helpers import get_ase_diamond_primitive, build_cell
    from scf import run_hf
    ase_atom = get_ase_diamond_primitive() 
    cell = build_cell(ase_atom)
    mf = run_hf(cell)
    
    cc = run_ccsd(mf)
    print "CCSD E =", cc.ecc

    eip = run_ipccsd(cc)
    print "IP-CCSD E =", eip
    eea = run_eaccsd(cc)
    print "EA-CCSD E =", eea
    print "Bandgap =", (eip[0]+eea[0])*27.211, "eV"

    eee = run_eeccsd(cc)
    print "EE-CCSD E ="
    # numpy 1.9.0 only
    es, ns = np.unique(eee.round(decimals=7), return_counts=True)
    for e,n in zip(es,ns):
        print e, "( x", n, ")"
    print "Opt gap =", eee[0]*27.211, "eV"

