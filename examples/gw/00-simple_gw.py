#!/usr/bin/env python

'''
A simple example to run a GW calculation 
'''

from pyscf import gto, dft, gw


def main():
    mol = gto.M(
        atom='H 0 0 0; F 0 0 1.1',
        basis='ccpvdz',
    )
    mf = dft.RKS(mol)
    mf.xc = 'pbe'
    mf.kernel()

    nocc = mol.nelectron // 2

    # By default, GW is done with analytic continuation.
    gw_obj = gw.GW(mf)
    # same as gw_obj = gw.GW(mf, freq_int='ac')
    try:
        gw_obj.kernel(orbs=range(nocc - 3, nocc + 3))
    except TypeError:
        # Newer GW-AC stores the orbital window on the object before kernel().
        gw_obj.orbs = range(nocc - 3, nocc + 3)
        gw_obj.kernel()
    print(gw_obj.mo_energy)


if __name__ == '__main__':
    main()
