from __future__ import division
import numpy as np
import ase.io as io
import subprocess

def write_geofdf(atoms, fname="geo.fdf"):
    from ase.data import chemical_symbols

    f=open(fname, "w")
    f.write("NumberOfAtoms   {0}\n".format(len(atoms)))
    f.write("NumberOfSpecies {0}\n".format(len(set(atoms.numbers))))
    f.write("\n")
    f.write('%block ChemicalSpeciesLabel\n')
    species_label = {}
    for i, nb in enumerate(set(atoms.numbers)):
        species_label[chemical_symbols[nb]] = i+1
        f.write('  {0}  {1}  '.format(i+1, nb)+ chemical_symbols[nb]+'\n')
    f.write("%endblock ChemicalSpeciesLabel\n")
    f.write("\n")

    f.write("AtomicCoordinatesFormat  Ang\n")
    f.write("%block AtomicCoordinatesAndAtomicSpecies\n")
    for ia, atm in enumerate(atoms):
        pos = atm.position
        f.write("{0:.6f}  {1:.6f}  {2:.6f}  {3} {4}  ".format(pos[0], pos[1], pos[2],
            species_label[atm.symbol], ia+1) + atm.symbol + "\n")

    f.write("%endblock AtomicCoordinatesAndAtomicSpecies")

    f.close()

xyz_range = np.arange(0, 5000, 25)

for i, xyz in enumerate(xyz_range):
    if xyz < 10:
        num = "00000{0}".format(xyz)
    elif xyz < 100:
        num = "0000{0}".format(xyz)
    elif xyz < 1000:
        num = "000{0}".format(xyz)
    elif xyz < 10000:
        num = "00{0}".format(xyz)
    else:
        raise ValueError("xyz too large?? {0}".format(xyz))

    path = "calc_" + num
    atoms = io.read("x"+num, format="xyz")
    subprocess.call("mkdir " + path, shell=True)
    write_geofdf(atoms, fname=path+"/geo.fdf")
    subprocess.call("cp siesta_C60.fdf " + path, shell=True)
    subprocess.call("cp C.psf " + path, shell=True)
