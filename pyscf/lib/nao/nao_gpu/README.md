GPU Routines for NAO iterative Procedure
========================================

2017-10-30

Description
-----------

This modules are cuda GPU routines to speed up the iterative part 
of the calculations (routine apply_rf0 in tddft_iter.py). The installation
is automatic if the CUDA library is detected while running CMake.
The flags for the compilation can be change in the CMakeLists.txt file present
in nao module. The speed up for the iterative procedure can be quite consequent
depending the hardware. We performed tests on a Intel Xeon CPU at 2.40 GHz (16 threads)
with a Geforce GTX 1050Ti, the speed up using the GPU for a silver cluster of 561 atoms
(+ 1 layer of ghost atoms, so a total of 923 atoms) and for 101 frequencies reached 
9.24 faster compared to mkl/Blas routines.

Bug report
----------
Koval Peter <koval.peter@gmail.com>

Marc Barbry <marc.barbry@mailoo.org>

