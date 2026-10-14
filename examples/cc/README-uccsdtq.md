# Unrestricted quadruples methods

Two methods extend the existing unrestricted CCSDT implementation:

- **UCCSDTQ** iterates the connected singles, doubles, triples, and quadruples
  equations. Its `e_tot` is the converged CCSDTQ total energy.
- **UCCSDT(Q)** evaluates the conventional noniterative quadruples correction
  to a converged CCSDT wavefunction, following Y. J. Bomble et al.,
  *J. Chem. Phys.* **123**, 054101 (2005), DOI: 10.1063/1.1950567.

## Usage

```python
from pyscf import cc, gto, scf

mol = gto.M(atom='Be 0 0 0; H 0 0 1.342', basis='sto-3g', spin=1)
mf = scf.UHF(mol).run(conv_tol=1e-12)
triples = cc.UCCSDT(mf, compact_tamps=True)
eris = triples.ao2mo()
triples.kernel(eris=eris)
assert mf.converged and triples.converged

e_bracket, e_paren = triples.ccsdt_q(eris=eris)
e_ccsdt_q = triples.e_tot + e_paren

quadruples = cc.UCCSDTQ(mf, compact_tamps=True)
quadruples.kernel(eris=eris)
assert quadruples.converged
e_ccsdtq = quadruples.e_tot
```

`ccsdt_q(tamps=None, eris=None)` returns `([Q], (Q))`. The second value already
includes `[Q]`; add only the second value to the CCSDT energy for CCSDT(Q).
It does not change the input amplitudes, integrals, `e_corr`, or `e_tot`.
Reuse the original AO-to-MO integrals, not the T1-dressed intermediates.

`cc.CCSDTQ(mf)` dispatches UHF references to UCCSDTQ and converts ROHF
references to UHF. Explicit `cc.UCCSDTQ(mf)` also accepts an RHF reference
by conversion. Existing frozen-orbital specifications are supported,
including different alpha and beta frozen lists.

See `67-uccsdt_q.py` for both perturbative storage layouts and
`71-uccsdtq.py` for starting UCCSDTQ from converged CCSDT amplitudes.
These minimal-basis examples demonstrate the interfaces.

## Storage and supported inputs

With `compact_tamps=True`, same-spin occupied and virtual groups in T3 and
T4 are stored as increasing combinations. T4 has five spin blocks:
`aaaa`, `aaab`, `aabb`, `bbba`, and `bbbb`. With `compact_tamps=False`,
the high-memory implementation stores full T3 and T4 arrays.
The direct and blocked quadruples backends use the same equations;
`quadruples_backend='auto'` selects between them. T3 contractions use the
existing PySCF UCCSDT implementation. T4 is excluded from DIIS by default;
`do_diis_max_t=True` includes it.

Both methods require real double-precision amplitudes and integrals.
UCCSDT(Q) requires canonical UHF orbitals. Iterative UCCSDTQ includes
off-diagonal Fock terms and supports real noncanonical orbitals, including
ROHF-to-UHF references. Complex orbitals and GHF references are unsupported.
Iterative damping is not implemented for UCCSDTQ.

The methods operate on full four-index MO integrals through the existing
UCCSDT integral transformation. There is no dedicated density-fitted,
out-of-core, or GPU quadruples algorithm. A density-fitted SCF reference
can reconstruct full integral tensors through the inherited transformation;
it does not remove their memory cost. Missing cached AO integrals without
density fitting raise an error for multi-electron systems. The one-electron
zero-integral special case is retained.

The C/OpenMP kernels build as part of `libccsdt`; no additional Python
dependency is required. `max_memory` controls quadruples workspace checks
and blocking, but is not an operating-system memory limit.

## Numerical validation

The accompanying tests compare residuals and corrections with independent
determinant-space and spin-orbital references. They cover all five spin
sectors, nonzero T3/T4, frozen orbitals, compact/full layouts, direct/blocked
backends, and multiple OpenMP thread counts. Small systems whose allowed
excitations terminate at rank four are checked against FCI; closed-shell
limits are checked against the restricted methods. These are numerical
implementation checks, not external molecular benchmark results.
