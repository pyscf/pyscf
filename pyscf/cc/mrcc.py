import os
import subprocess
from pyscf import tools

class MRCCInterface:
    @staticmethod
    def generate_integrals(mf, filename='fort.55', **kwargs):
        """
        Generate the integrals file (fort.55) from the SCF calculation.
        
        Parameters:
        - mf: The mean-field object from the SCF calculation.
        - filename: The name of the output integrals file (default is 'fort.55').
        - kwargs: Additional keyword arguments for from_chkfile_uhf.
        """
        # Ensure the filename is 'fort.55'
        if filename != 'fort.55':
            print("Warning: The filename for the integrals file must be 'fort.55'. It will be set to 'fort.55'.")
            filename = 'fort.55'
        
        # Generate the integrals file
        tools.fcidump.from_chkfile_uhf(filename, mf.chkfile, **kwargs)
    
    @staticmethod # The method is not implemented for this version, currently the user shall provide the MINP file
    def create_mrcc_input(options, filename='MINP'):
        # Create the MRCC input file based on provided options
        with open(filename, 'w') as f:
            f.write(f"MRCC input file\n")
            # Add options to the input file
            for key, value in options.items():
                f.write(f"{key} = {value}\n")

    @staticmethod
    def run_mrcc(mf, fort_file='fort.55', mrcc_input_file='MINP', tol=1e-18, float_format='% 0.20E', **kwargs):
        """
        Run the MRCC calculation.
        
        Parameters:
        - mf: The mean-field object from the SCF calculation.
        - fort_file: The name of the integrals file (default is 'fort.55').
        - mrcc_input_file: The name of the MRCC input file (default is 'MINP').
        - tol: Tolerance for integral generation (default is 1e-18).
        - float_format: Format for floating-point numbers (default is '% 0.20E').
        - kwargs: Additional keyword arguments for generate_integrals.
        """
        # Check if the fort.55 file exists
        if not os.path.isfile(fort_file):
            print(f"'{fort_file}' does not exist. Generating integrals file...")
        # Generate the integrals file
            MRCCInterface.generate_integrals(mf, fort_file, tol=tol, float_format=float_format, **kwargs)
        else:
            print(f"'{fort_file}' already exists. Proceeding with MRCC calculation.")
        
        # Check if the MINP input file exists
        if not os.path.isfile(mrcc_input_file):
            print(f"Error: The MRCC input file '{mrcc_input_file}' does not exist.")
            return

        # Call the MRCC executable with the specified files
        try:
            with open("mrcc-output", "w") as mrcc_output_file:
                subprocess.run(['dmrcc', mrcc_input_file], stdout=mrcc_output_file, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running MRCC: {e}")

# Example usage
# Assuming the user has already run the following to generate fort.55:
# pyscf.tools.fcidump.from_chkfile('fort.55', name+'.chk', tol=1e-18, float_format='% 0.20E', molpro_orbsym=False)
'''
from cc.mrcc import MRCCInterface

# Define MRCC options for the MINP file
mrcc_options = {
    'CC_METHOD': 'CCSD',  # Example option, adjust as needed
    'MAX_ITER': 100,
    'CONVERGENCE': 1e-8,
    # Add other necessary options here
}

# Run MRCC with specified files
MRCCInterface.run_mrcc(mf, fort_file='fort.55', mrcc_input_file='MINP', tol=1e-18, float_format='% 0.20E', molpro_orbsym=False)
'''