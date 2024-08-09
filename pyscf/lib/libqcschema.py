import json
import numpy as np
from pyscf.lib.parameters import BOHR

def load_qcschema_json( file_name ):
    # load qcschema output json file
    data = None
    with open(file_name,'r') as f:
        data = json.load(f)
    return data

def load_qcschema_go_final_json( file_name ):
    # load qcschema GO output json file 
    # and return last 'trajectory' point's entries
    # (this is the optimized molecule)
    data = None
    temp = None
    with open(file_name,'r') as f:
        temp = json.load(f)
    data = temp["trajectory"][-1]
    return data

def combine_syms_coords(syms,coords,to_Angstrom = False, xyz=False):
    # QCSchema molecules are split into symbols and coordinates,
    #  combine them here.
    # Optionally convert from Bohr to Angstrom.
    # Returns: either a geometry string (default) or
    #  optionally it can return an xyz format string.
    if(to_Angstrom):
        # convert Bohr to Angstrom
        coords = coords*BOHR
        
    NAtoms = len(syms)
    geo = np.reshape(coords, (NAtoms,3))

    # Concatenate the symbols and coordinates along the second axis
    combined = np.concatenate([syms[:, np.newaxis], geo], axis=1)

    # Convert the combined array to a string with spaces as separators
    output = np.array2string(combined, separator=' ', max_line_width=np.inf, threshold=np.inf)

    # Remove the brackets and quotes from the output string
    PySCF_atoms = output.replace('[', '').replace(']', '').replace("'", '')

    # Return as string or return as xyz-format string (i.e. top is NAtoms,blankline)
    if(xyz):
        xyz = f'{NAtoms}\n\n'
        PySCF_atoms = xyz+PySCF_atoms
    
    return PySCF_atoms

def load_qcschema_molecule(qcschema_dict, to_Angstrom = False, xyz=False):
    # Load QCSchema molecule.
    # Optionally convert geometry to angstrom
    # Returns: either a geometry string (default) or
    #  optionally it can return an xyz format string.

    syms = np.array(qcschema_dict["molecule"]["symbols"])
    geo = np.array(qcschema_dict["molecule"]["geometry"])

    # combine together the symbols and coordinates
    PySCF_atoms = combine_syms_coords(syms,geo,to_Angstrom,xyz)

    return PySCF_atoms

def load_qcschema_final_molecule(qcschema_dict, to_Angstrom = False, xyz=False):
    # Load final molecule in QCSchema. 
    # In GO job this is the optimized geometry.
    # Optionally convert geometry to angstrom
    # Returns: either a geometry string (default) or
    #  optionally it can return an xyz format string.

    syms = np.array(qcschema_dict["final_molecule"]["symbols"])
    geo = np.array(qcschema_dict["final_molecule"]["geometry"])

    # combine together the symbols and coordinates
    PySCF_atoms = combine_syms_coords(syms,geo,to_Angstrom,xyz)

    return PySCF_atoms

def load_qcschema_hessian(qcschema_dict):
    # qcschema_dict: pass in dict that has the qcschema output json loaded into it

    # load qcschema hessian
    qc_h = []
    qc_h = qcschema_dict["return_result"]

    # Get Number of atoms N
    syms = np.array(qcschema_dict["molecule"]["symbols"])
    NAtom = len(syms)

    # reshape from (3N)**2 array to (N,N,3,3)
    hessian = np.array(qc_h).reshape(NAtom,NAtom,3,3)
    return hessian

def load_qcschema_mol_scf(qcschema_dict,save_chk=False,chkfile="output.chk",to_Angstrom=False):
    # uses qcschema dict and re-creates PySCF mol and scf info 
    # returns mol and scf and optionally saves to chkfile
    # qcschema_dict: dict containing the qcschema output loaded into it
    # save_chk: whether to save chkfile or not
    # chkfile: the name of the chkfile we may create
    try:
        import pyscf
        from pyscf.lib.chkfile import load_chkfile_key, load
        from pyscf.lib.chkfile import dump_chkfile_key, dump, save
        from pyscf.lib.chkfile import load_mol, save_mol
    except ImportError:
        raise ImportError(
            "Missing optional 'pyscf' dependencies. \
            To install run: pip install pyscf"
        )
    
    # Accelerated DFT service return scf_occupations_a only for R, so occ is 1 or 0.
    # Need to double if RHF/RKS/ROHF
    method = qcschema_dict["keywords"]["scf"]["method"]
    if(method == 'rks' or method == 'roks' or method == 'rhf' or method == 'rohf'):
        OccFactor = 2.0
        have_beta = False
    elif(method == 'uks' or method == 'uhf'):
        OccFactor = 1.0
        have_beta = True
    elif(method == 'gks' or method == 'ghf'):
        OccFactor = 1.0
        have_beta = False

    # Need to reshape MO coefficients for PySCF shape.
    # NOTE: assumes NMO=NAO which isn't the case if linear dependencies etc. 
    nmo = qcschema_dict["properties"]["calcinfo_nbasis"]
    nao = nmo

    ## chkfile info ##
    # Get the 4 things that PySCF chkfile wants
    # ...remembering to reshape coeffs and scale occupancies.
    e_tot = float( qcschema_dict["properties"]["return_energy"] )
    mo_coeff = np.reshape(qcschema_dict["wavefunction"]["scf_orbitals_a"],(nao,nmo))
    mo_occ = np.array( qcschema_dict["wavefunction"]["scf_occupations_a"] )*OccFactor
    mo_energy = np.array( qcschema_dict["wavefunction"]["scf_eigenvalues_a"] )
    if(have_beta): 
        # for each useful piece of info we need to combine alpha and beta into 2d array, with alpha first
        # MO occupations
        mo_occ_beta = qcschema_dict["wavefunction"]["scf_occupations_b"]
        mo_occ = np.vstack( (mo_occ, mo_occ_beta) )
        # MO coefficients
        mo_coeff_beta = np.reshape(qcschema_dict["wavefunction"]["scf_orbitals_b"],(nao,nmo))
        mo_coeff = np.vstack( (mo_coeff,mo_coeff_beta))
        mo_coeff = np.reshape(mo_coeff,(2,nao,nmo))
        # MO energies
        mo_energy_beta = np.array( qcschema_dict["wavefunction"]["scf_eigenvalues_b"] )
        mo_energy = np.vstack( (mo_energy, mo_energy_beta) )
        # etot obviously doesn't need manipulation

    # Convert to dictionary for PySCF
    scf_dic = {'e_tot'    : e_tot,
               'mo_energy': mo_energy,
               'mo_occ'   : mo_occ,
               'mo_coeff' : mo_coeff}

    ## Mol info: ##
    PySCF_charge = int( qcschema_dict["molecule"]["molecular_charge"] )
    # PySCF 'spin' is number of unpaired electrons, it will be mult-1
    PySCF_spin = int( qcschema_dict["molecule"]["molecular_multiplicity"] - 1 )
    PySCF_basis = str( qcschema_dict["model"]["basis"] )

    # Cartesian/Pure basis
    PySCF_cart = bool( qcschema_dict["keywords"]["basisSet"]["cartesian"] )

    # Get molecular structure.
    # QCSchema has separate atom symbols and coordinates
    syms = np.array(qcschema_dict["molecule"]["symbols"])
    geo = np.array(qcschema_dict["molecule"]["geometry"])
    PySCF_atoms = load_qcschema_molecule(qcschema_dict, to_Angstrom,False)

    # Unit Bohr or Angstrom. QCSchema default is Bohr but can change here.
    if(to_Angstrom):
        units='A'
    else:
        units='B'

    ## Create mol and save to chkfile ##
    mol = pyscf.gto.Mole(atom=PySCF_atoms,basis=PySCF_basis,ecp=PySCF_basis,charge=PySCF_charge,spin=PySCF_spin,cart=PySCF_cart,unit=units)

    ## Save scf info data into chk ##
    if(save_chk):
        save(chkfile, 'scf', scf_dic)
        save_mol(mol,chkfile)

    return scf_dic, mol

import pyscf
def recreate_scf_obj(qcschema_dict,save_chk=False,chkfile="",to_Angstrom=False):
    # Create Pyscf Molecule
    scf_dict, mol = load_qcschema_mol_scf(qcschema_dict,save_chk,chkfile,to_Angstrom)
    mol.build()

    # Create DFT (or HF) object
    method =  qcschema_dict["keywords"]["scf"]["method"]
    if(method =='rks'):
        ks = pyscf.dft.RKS(mol)
    elif(method =='uks'): 
        ks = pyscf.dft.UKS(mol)
    elif(method =='rhf'):
        ks = pyscf.hf.RKS(mol)
    elif(method =='uhf'):
        ks = pyscf.hf.UHF(mol)
    else:
        print("cannot determine method..exit")
        return

    #temp set functional...could get it from the json
    if(method == 'rks' or method == 'uks'):
        functional = qcschema_dict["keywords"]["xcFunctional"]["name"]
        #functional = "b3lyp"
        ks.xc = functional

    # Load 4 key pieces of info we got from json into DFT object
    ks.mo_coeff = scf_dict["mo_coeff"]
    ks.mo_energy = scf_dict["mo_energy"]
    ks.mo_occ = scf_dict["mo_occ"]
    ks.e_tot = scf_dict["e_tot"]
    return mol, ks
