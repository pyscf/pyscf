from __future__ import division
import subprocess
import ase

print(ase.data.chemical_symbols)
for pseudo,pseudo_min in zip(["LDA", "GGA"], ["lda", "gga"]):
    for sym in ase.data.chemical_symbols:
        cmd = "wget https://departments.icmab.es/leem/siesta/Databases/Pseudopotentials/Pseudos_" + pseudo + "_Abinit/" + sym + "_html/" + sym + ".psf"
        print(cmd)
        subprocess.call(cmd, shell=True)
        try:
            cmd = "mv " + sym + ".psf " + sym + "." + pseudo_min + ".psf"
            print(cmd)
            subprocess.call(cmd, shell = True)
        except:
            print("no file " + sym + ".psf")
