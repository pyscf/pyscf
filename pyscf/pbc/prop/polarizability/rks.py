from pyscf.pbc.prop.polarizability.rhf import \
        (polarizability, hyper_polarizability, polarizability_with_freq,
         Polarizability)

if __name__ == "__main__":
    from pyscf.pbc import gto, scf
    cell = gto.Cell()
    cell.atom = """H  0.0 0.0 0.0
                   F  0.9 0.0 0.0
                """
    cell.basis = 'sto-3g'
    cell.a = [[2.82, 0, 0], [0, 2.82, 0], [0, 0, 2.82]]
    cell.dimension = 1
    cell.precision = 1e-10
    cell.build()

    kpts = cell.make_kpts([16,1,1])
    kmf = scf.KRKS(cell, kpts=kpts).density_fit()
    kmf.xc = "lda"
    kmf.kernel()

    #TODO implement the finite field version
    polar = Polarizability(kmf, kpts)
    dip = polar.dip_moment()
    print(dip)
    e2 = polar.polarizability()
    print(e2)
    e2 = polar.polarizability_with_freq(freq=0.)
    print(e2)
    e3 = polar.hyper_polarizability()
    print(e3)
