# README #

### What is this repository for? ###

* This module is for parallel mole/pbc coupled cluster calculations. All integrals are stored in memory as distributed ctf tensors(wrapper by symtensor). Functions includes

* * mole rccsd/uccsd/gccsd
* * kpoint rccsd/uccsd/gccsd
* * eomip, eomea for all mole/kpoint ccsd class
* * parallel GDF

### What are the prerequisites? ###

* * CTF: https://github.com/cyclops-community/ctf
* * Symtensor: https://github.com/yangcal/symtensor
* * mpi4py: https://bitbucket.org/mpi4py/

### Who do I talk to? ###

* Questions about the ctfcc/symtensor module: younggao1994@gmail.com
* Questions about compiling CTF: solomon2@illinois.edu
