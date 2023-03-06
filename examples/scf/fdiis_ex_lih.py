from pyscf import gto, scf
import numpy



def main():

    # LiH for different bond lengths, single point calculations

    minbond = 0.1
    maxbond = 10.0
    samples = 50

    savefile = 'lih-profile.txt'

    bondlengths = numpy.linspace(minbond, maxbond, samples)

    for bond in bondlengths:

        atom_str = 'Li 0 0 0; H 0 0 ' + str(bond)
        mol = gto.M(atom=atom_str, basis='6-31+g*')

        hf = scf.HF(mol)
        hf.diis = scf.EDIIS()
        hf.max_cycle = 250

        hf_result = hf.kernel()

        m3 = scf.M3SOSCF(hf, 25, trustScaleRange=(0.005, 1.0, 6))
        m3_result = m3.converge()

        with open(savefile, 'a') as f:
            writestring = "Dist: " + str(round(bond, 5)) + " A,  HF: " + str(round(hf_result, 10)) + " au (Converged? " + str(hf.converged) + "), M3: "
            writestring += str(round(m3_result[1], 10)) + " au (Converged? " + str(m3_result[0]) + ")\n"

            print(writestring)

            f.write(writestring)



if __name__ == '__main__':
    main()
