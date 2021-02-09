import numpy as np
#from pyscf import ao2mo
#from pyscf.pbc import gto, scf, mp, df
from pyscf.pbc import df
from pyscf.pbc.lib import kpts_helper
#from pyscf.pbc.agf2 import kragf2_ao2mo as kao2mo
#from . import ao2mo as kao2mo
import ao2mo as kao2mo


def make_ao_3c_eris(rhf):
    # Return the 3-center ERIs in AO basis, with dimensions:
    #  bra: (nk, nk, ngrids, nao**2)
    #  ket: (nk, nk, nk, ngrids, nao**2)

    duck_agf2 = lambda x: None
    duck_agf2.with_df = rhf.with_df
    duck_eris = lambda x: None
    duck_eris.kpts = rhf.kpts

    if isinstance(rhf.with_df, df.MDF):
        bra, ket = kao2mo._make_ao_eris_direct_mdf(duck_agf2, duck_eris)
    elif isinstance(rhf.with_df, df.GDF):                           
        bra, ket = kao2mo._make_ao_eris_direct_gdf(duck_agf2, duck_eris)
    elif isinstance(rhf.with_df, df.AFTDF):
        bra, ket = kao2mo._make_ao_eris_direct_aftdf(duck_agf2, duck_eris)
    elif isinstance(rhf.with_df, df.FFTDF):
        bra, ket = kao2mo._make_ao_eris_direct_fftdf(duck_agf2, duck_eris)
    else:
        raise ValueError('Unknown DF type %s' % type(rhf.with_df))

    return bra, ket


def make_mo_3c_eris(rhf, eri_ao_3c=None):
    # Return the 3-center ERIs in MO basis, with dimensions:
    #  bra: (nk, nk, ngrids, nmo**2)
    #  ket: (nk, nk, nk, ngrids, nmo**2)

    if eri_ao_3c is None:
        bra, ket = make_ao_3c_eris(rhf)
    else:
        bra, ket = eri_ao_3c
    mo_coeff = rhf.mo_coeff

    dtype = np.result_type(bra.dtype, ket.dtype, *[x.dtype for x in mo_coeff])
    naux = bra.shape[2]

    kpts = rhf.kpts
    nkpts = len(kpts)
    nmo = rhf.mo_occ[0].size

    kij = np.array([(ki,kj) for ki in kpts for kj in kpts])
    kis, kjs = kij[:,0], kij[:,1]
    q = kjs - kis
    ukpts, uidx, uinv = kpts_helper.unique(q)
    kconserv = kpts_helper.get_kconserv(cell, kpts)

    qij = np.zeros((nkpts, nkpts, naux, nmo**2), dtype=dtype)
    qkl = np.zeros((nkpts, nkpts, nkpts, naux, nmo**2), dtype=dtype)

    for uid in kao2mo.mpi_helper.nrange(len(ukpts)):
        q = ukpts[uid]
        adapted_ji = np.where(uinv == uid)[0]
        kjs = kij[:,1][adapted_ji]

        for ji, ji_idx in enumerate(adapted_ji):
            ki, kj = divmod(adapted_ji[ji], nkpts)
            ci = mo_coeff[ki].conj()
            cj = mo_coeff[kj].conj()
            qij[ki,kj] = kao2mo._fao2mo(bra[ki,kj], ci, cj, dtype, out=qij[ki,kj])

            for kk in range(nkpts):
                kl = kconserv[ki,kj,kk]
                ck = mo_coeff[kk]
                cl = mo_coeff[kl]
                qkl[ki,kj,kk] = kao2mo._fao2mo(ket[ki,kj,kk], ck, cl, dtype, out=qkl[ki,kj,kk])

    kao2mo.mpi_helper.barrier()
    kao2mo.mpi_helper.allreduce_safe_inplace(qij)
    kao2mo.mpi_helper.allreduce_safe_inplace(qkl)

    return qij, qkl


def make_mo_4c_eris(rhf):
    # Return the 4-center ERIs in MO basis, with dimensions:
    #  (nk, nk, nk, nmo, nmo, nmo, nmo)

    with_df = rhf.with_df
    mo_coeff = np.array(rhf.mo_coeff)
    dtype = np.result_type(*[x.dtype for x in mo_coeff])

    kpts = rhf.kpts
    nkpts = len(kpts)
    nmo = rhf.mo_occ[0].size
    npair = nmo * (nmo+1) // 2
    kconserv = kpts_helper.get_kconserv(cell, kpts)

    eri = np.empty((nkpts, nkpts, nkpts, nmo, nmo, nmo, nmo), dtype=dtype)

    for kpqr in kao2mo.mpi_helper.nrange(nkpts**3):
        kpq, kr = divmod(kpqr, nkpts)
        kp, kq = divmod(kpq, nkpts)
        ks = kconserv[kp,kq,kr]

        coeffs = mo_coeff[[kp,kq,kr,ks]]
        kijkl = kpts[[kp,kq,kr,ks]]

        eri_kpt = with_df.ao2mo(coeffs, kijkl, compact=False)
        eri_kpt = eri_kpt.reshape(nmo, nmo, nmo, nmo)

        if dtype in [np.float, np.float64]:
            eri_kpt = eri_kpt.real

        eri[kp,kq,kr] = eri_kpt / nkpts

    kao2mo.mpi_helper.barrier()
    kao2mo.mpi_helper.allreduce_safe_inplace(eri)

    return eri


if __name__ == "__main__":
    from pyscf.pbc import gto, scf
    from timeit import default_timer as timer

    cell = gto.C(
        atom='He 1 0 1; He 0 0 1',
        basis='6-31g',
        a=np.eye(3)*3,
        mesh=[15,15,15],
        verbose=0,
    )

    rhf = scf.KRHF(cell)
    rhf.kpts = cell.make_kpts([2,2,2])
    #rhf.density_fit()
    rhf.with_df = df.FFTDF(cell, rhf.kpts)
    rhf.run()

    # 3-center AO integrals:
    t0 = timer()
    Lpq, Lrs = make_ao_3c_eris(rhf)
    print("Time 3c AO: %.5f" % (timer()-t0))

    # 3-center MO integrals:
    t0 = timer()
    Lij, Lkl = make_mo_3c_eris(rhf, eri_ao_3c=(Lpq, Lrs))
    print("Time 3c MO: %.5f" % (timer()-t0))

    def unpack(x):
        n = int(np.round(x.shape[-1]**0.5))
        shape = list(x.shape[:-1]) + [n, n]
        return x.reshape(shape)

    # 4-center AO integrals (not optimised):
    t0 = timer()
    eri_ao_4c = np.einsum('abLpq,abcLrs->abcpqrs', unpack(Lpq).conj(), unpack(Lrs))
    print("Time 4c AO: %.5f" % (timer()-t0))

    # 4-center MO integrals (not optimised):
    t0 = timer()
    eri_mo_4c = np.einsum('abLij,abcLkl->abcijkl', unpack(Lij).conj(), unpack(Lkl))  
    print("Time 4c MO: %.5f" % (timer()-t0))

    # 4-center MO integrals (optimised):
    t0 = timer()
    eri_mo_4c_opt = make_mo_4c_eris(rhf)
    print("Time 4c MO (opt): %.5f" % (timer()-t0))

    # Sanity checks:
    t0 = timer()
    eri_ao_4c_check = (1.0/len(rhf.kpts))*rhf.with_df.ao2mo_7d([np.array([np.eye(cell.nao),]*len(rhf.kpts)),]*4)
    print("Time 4c AO (PySCF): %.5f" % (timer()-t0))
    t0 = timer()
    eri_mo_4c_check = (1.0/len(rhf.kpts))*rhf.with_df.ao2mo_7d([np.array(rhf.mo_coeff),]*4)
    print("Time 4c MO (PySCF): %.5f" % (timer()-t0))
    assert np.allclose(eri_ao_4c, eri_ao_4c_check)
    assert np.allclose(eri_mo_4c, eri_mo_4c_check)
    assert np.allclose(eri_mo_4c, eri_mo_4c_opt)
    print("All done")
