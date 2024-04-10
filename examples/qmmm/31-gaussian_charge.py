"""
QMMM calculation with the MM charges being Gaussian distributions
"""

from pyscf import gto, scf, qmmm, grad

mol = gto.M(
    verbose = 3,
    atom = '''O       -1.464   0.099   0.300
              H       -1.956   0.624  -0.340
              H       -1.797  -0.799   0.206''',
    basis = '631G')

mm_coords = [(1.369, 0.146,-0.395),  # O
              (1.894, 0.486, 0.335), # H
              (0.451, 0.165,-0.083)] # H
mm_charges = [-1.040, 0.520, 0.520]
mm_radii = [0.63, 0.32, 0.32] # radii of Gaussians


mf = qmmm.mm_charge(scf.RHF(mol), mm_coords, mm_charges, mm_radii)
e_hf = mf.kernel() # -76.00028338468692

# QM grad
g_hf = qmmm.mm_charge_grad(grad.RHF(mf), mm_coords, mm_charges, mm_radii)
g_hf_qm = g_hf.kernel()
print('Nuclear gradients of QM atoms:')
print(g_hf_qm)

# MM grad
# NOTE For post-HF methods, the density matrix should
# include the orbital response. See more details
# in examples/qmmm/30-force_on_mm_particles.py
g_hf_mm_h1 = g_hf.grad_hcore_mm(mf.make_rdm1())
g_hf_mm_nuc = g_hf.grad_nuc_mm()
print('Nuclear gradients of MM atoms:')
print(g_hf_mm_h1 + g_hf_mm_nuc)
