#!/usr/bin/env python

"""Validate internal CMS-PDFT diabatization derivatives for three-state water.

Compare optimized response and nuclear-derivative routines with their reference
implementations. These are intermediate derivatives of the diabatization
objective, not complete CMS-PDFT nuclear gradients. For state gradients see
19-multi_state_mcpdft_grad.py.
"""

import numpy as np
from pyscf import gto, scf, mcscf, lib
from pyscf.grad.cmspdft import (
    diab_response, diab_response_o0, diab_grad, diab_grad_o0,
)

mol = gto.M(
    atom='''O  0.00000000   0.08111156   0.00000000
            H  0.78620605   0.66349738   0.00000000
            H -0.78620605   0.66349738   0.00000000''',
    basis='6-31g', symmetry=False, verbose=4)
mf = scf.RHF(mol).run()
mc = mcscf.CASSCF(mf, 4, 4)
# Use the standard determinant solver with a singlet spin penalty, avoiding
# the optional CSF solver used in the original module demonstration.
mc.fix_spin_(ss=0)
mc = mc.state_average([1. / 3] * 3).run()
assert mc.converged
ci = np.asarray(mc.ci)
mc_grad = mc.nuc_grad_method()
eris = mc.ao2mo(mc.mo_coeff)

# Three states have three independent pair rotations. Use reproducible
# coefficients for the response contracted with these interstate rotations.
rng = np.random.default_rng(37)
Lis = np.pi * (rng.random(3) - 0.5)


def response_components(response):
    orbital, ci_response = mc_grad.unpack_uniq_var(response)
    ci_response = np.asarray(ci_response)
    # Resolve changes within the reference-state span (interstate rotations)
    # separately from CI changes orthogonal to all reference states.
    interstate = lib.einsum('pab,qab->pq', ci_response, ci.conj())
    orthogonal_ci = ci_response - lib.einsum('pq,qab->pab', interstate, ci)
    return orbital, orthogonal_ci, interstate


response = diab_response(mc_grad, Lis, mo=mc.mo_coeff, ci=mc.ci, eris=eris)
reference = diab_response_o0(mc_grad, Lis, mo=mc.mo_coeff, ci=mc.ci, eris=eris)
orbital, ci_response, interstate = response_components(response)
orbital_ref, ci_ref, interstate_ref = response_components(reference)
print('Orbital response difference norm:', np.linalg.norm(orbital - orbital_ref))
print('Orthogonal CI response difference norm:', np.linalg.norm(ci_response - ci_ref))
print('Interstate response difference norm:', np.linalg.norm(interstate - interstate_ref))

# Supply the RHF gradient explicitly for the state-averaged CASSCF reference.
mf_grad = mf.nuc_grad_method()
nuclear = diab_grad(mc_grad, Lis, mo=mc.mo_coeff, ci=mc.ci, eris=eris,
                    mf_grad=mf_grad)
nuclear_ref = diab_grad_o0(mc_grad, Lis, mo=mc.mo_coeff, ci=mc.ci, eris=eris,
                           mf_grad=mf_grad)
print('Nuclear derivative difference norm:', np.linalg.norm(nuclear - nuclear_ref))
print('Optimized nuclear derivative:\n', nuclear)
print('Reference nuclear derivative:\n', nuclear_ref)
