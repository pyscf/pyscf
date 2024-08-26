from pyscf import gto, scf, mcscf, lib

# NAC signs are really, really hard to nail down.
# There are arbitrary signs associated with
# 1. The MO coefficients
# 2. The CI vectors
# 3. Almost any kind of post-processing (natural-orbital analysis, etc.)
# 4. Developer convention on whether the bra index or ket index is 1st
# It MIGHT help comparison to OpenMolcas if you load a rasscf.h5 file
# I TRIED to choose the same convention for #4 as OpenMolcas.
mol = gto.M (atom='Li 0 0 0;H 1.5 0 0', basis='sto-3g',
             output='LiH_sa2casscf22_sto3g.log', verbose=lib.logger.INFO)

mf = scf.RHF (mol).run ()
mc = mcscf.CASSCF (mf, 2, 2)
mc.fix_spin_(ss=0, shift=1)
mc = mc.state_average ([0.5,0.5]).run (conv_tol=1e-10)

mc_nacs = mc.nac_method()

# 1. <1|d0/dR>
#    Equivalent OpenMolcas input:
#    ```
#    &ALASKA
#    NAC=1 2
#    ```
nac = mc_nacs.kernel (state=(0,1))
print ("\nNAC <1|d0/dR>:\n", nac)
print ("Notice that according to the NACs printed above, rigidly moving the")
print ("molecule along the bond axis changes the electronic wave function, which")
print ("is obviously unphysical. This broken translational symmetry is due to the")
print ("'CSF contribution'. Omitting the CSF contribution corresponds to using the")
print ("'electron-translation factors' of Fatehi and Subotnik and is requested by")
print ("passing 'use_etfs=True'.")

# 2. <1|d0/dR> w/ ETFs (i.e., w/out CSF contribution)
#    Equivalent OpenMolcas input:
#    ```
#    &ALASKA
#    NAC=1 2
#    NOCSF
#    ```
nac = mc_nacs.kernel (state=(0,1), use_etfs=True)
print ("\nNAC <1|d0/dR> w/ ETFs:\n", nac)
print ("These NACs are much more well-behaved: moving the molecule rigidly around")
print ("in space doesn't induce any change to the electronic wave function.")

# 3. <0|d1/dR>
#    Equivalent OpenMolcas input:
#    ```
#    &ALASKA
#    NAC=2 1
#    ```
nac = mc_nacs.kernel (state=(1,0))
print ("\nThe NACs are antisymmetric with respect to state transposition.")
print ("NAC <0|d1/dR>:\n", nac)

# 4. <0|d1/dR> w/ ETFs
#    Equivalent OpenMolcas input:
#    ```
#    &ALASKA
#    NAC=2 1
#    NOCSF
#    ```
nac = mc_nacs.kernel (state=(1,0), use_etfs=True)
print ("NAC <0|d1/dR> w/ ETFs:\n", nac)

# 5. <1|d0/dR>*(E1-E0) = <0|d1/dR>*(E0-E1)
#    I'm not aware of any OpenMolcas equivalent for this, but all the information
#    should obviously be in the output file, as long as you aren't right at a CI.
nac_01 = mc_nacs.kernel (state=(0,1), mult_ediff=True)
nac_10 = mc_nacs.kernel (state=(1,0), mult_ediff=True)
print ("\nNACs diverge at conical intersections (CI). The important question")
print ("is how quickly it diverges. You can get at this by calculating NACs")
print ("multiplied by the energy difference using the keyword 'mult_ediff=True'.")
print ("This yields a quantity which is symmetric wrt state interchange and is")
print ("finite at a CI.")
print ("NAC <1|d0/dR>*(E1-E0):\n", nac_01)
print ("NAC <0|d1/dR>*(E0-E1):\n", nac_10)

# 6. <1|d0/dR>*(E1-E0) w/ETFs = <0|d1/dR>*(E0-E1) w/ETFs = <0|dH/dR|1>
#    This is the quantity one uses to optimize MECIs
v01 = mc_nacs.kernel (state=(0,1), use_etfs=True, mult_ediff=True)
v10 = mc_nacs.kernel (state=(1,0), use_etfs=True, mult_ediff=True)
print ("\nUsing both 'use_etfs=True' and 'mult_ediff=True' corresponds to the")
print ("derivative of the off-diagonal element of the potential matrix. This")
print ("tells you one of the two components of the branching plane at the CI.")
print ("<1|d0/dR>*(E1-E0) w/ ETFs = <1|dH/dR|0>:\n", v01)
print ("<0|d1/dR>*(E0-E1) w/ ETFs = <0|dH/dR|1>:\n", v10)

