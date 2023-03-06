from pyscf import gto, scf, dft
import numpy


def getGeometry(name):
    geometryfile = 'geometries.geo'
    content = None
    with open(geometryfile) as gf:
        content = gf.readlines()

    currentGeometry = False
    atom_str = ""
    charge = 0
    spin = 0
    for line in content:
        if not currentGeometry and line.strip() == "# " + name:
            currentGeometry = True
            continue
        elif currentGeometry and "#" in line:
            return atom_str, charge, spin
        line = line.strip()

        if currentGeometry:
            if len(line.split(" ")) == 2:
                l = line.split(" ")
                charge = int(l[0])
                spin = int(l[1])-1
            else:
                if len(atom_str) > 0:
                    atom_str += "; "
                atom_str += line
    return atom_str, charge, spin


def getBasis(path):

    basis = {}

    with open(path) as f:
        content = f.readlines()

        tag = ""

        for line in content:
            if '#' in line:
                tag = line.replace('#', '').strip()
                basis[tag] = ''
                continue
            basis[tag] += line
                
    return basis

def main(name):

    atom_str, charge, spin = getGeometry(name)
    print("Atom str:")
    print(atom_str)
    spin = 0
    basis = getBasis("6-31gs.bas")
    
    mol1 = gto.M(atom=atom_str, charge=charge, spin=spin, verbose=4, basis=basis)
    
    rks = dft.RKS(mol1, xc='cam-b3lyp')

    diis_m3 = scf.DIIS_M3(rks, 256, initScattering=0.3, trustScaleRange=(0.07, 0.2, 1))
    diis_m3.kernel(bufferSize=15)



if __name__ == '__main__':
    main("Azapentacene")
