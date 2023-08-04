.. _nao_examples_qmd_C60:

nao.examples.qmd_C60 --- NAO Examples: QMD C60
**********************************************
A demonstration of the Nao module used to calculate the polarizability
of C60 at rooms temperature using Quantum Molecular Dynamic (QMD).
The relaxation and the ground state calculations are done using the Siesta
program. This example will calculate the geometries of 5000 steps of C60 using Siesta
and the polarizability will be obtained by calculating 200 steps. This example MUST be
done on a server because it will perform 200 calculations. This example has been done
for the `Torque <https://wiki.archlinux.org/index.php/TORQUE>`_ resources manager and will 
need to be adapted to your case before to run.

To start the calculations, we first need to generate the 5000 geometries using Molecular
Dynamic (MD) with temperature controlled by means of a Nosé thermostat. For this we use
the following `siesta <siesta_C60_thermal.fdf>`_ input file,

.. literalinclude:: siesta_C60_thermal.fdf

The file containing the position of the atoms can be downloaded with the following
`link <C60_org.fdf>`_.

You run this calculation with the normal siesta command (don't forget the *C.psf* file)::
  siesta < siesta_C60_thermal.fdf > siesta.out
or with a bash script if you run on a server.
Once, this first calculation finish you will be left with a siesta.ANI file containing the
geometry at each time step. We will first generate the xyz files corresponding to the
geometries for comomdity,

.. literalinclude:: create_qmmd.sh

This will create a folder xyz containing the 5000 generated geometries. We need then to 
generate the fdf geometries file for siesta::
  cd cyz
  python xyz2fdf.py

`xyz2fdf.py <xyz2fdf.py>`_ is a small python script that will generate the geometries for 200 calculations
from the 5000 previous geometries because 5000 is a bit too much. The script look like,

.. literalinclude:: xyz2fdf.py 

The script will generate the siesta input files in different trajectory, don't forget to add in
the xyz directory the `siesta input <siesta_C60.fdf>`_

.. literalinclude:: siesta_C60.fdf 

After running the xyz2fdf.py script you should have 200 folders containing the 3 input files,
  * siesta_C60.fdf
  * geo.fdf
  * C.psf

I advise you to create a new folder, and to copy all the calculations input into this new folder::
 
  cd ..
  mkdir calculations
  cp -r xyz/calc_* calculations
  cd calculations

Now we will need script to perform the siesta and nao calculations. Here, I will perform in total 50
jobs, each running 4 calculations. Each jobs will be using 6 cpus. I'm assuming that you are using 
the Torque managing system. We will use two python script to perform the calculations,
  * `submit_jobs.py <submit_jobs.py>`_
  * `calc_polarizability.py <calc_polarizability.py>`_

The first one is the script that you call in order to submit the 200 jobs and it call the second 
script *calc_polarizability.py*. This second script will perform the siesta and nao calculations
for each configurations.

submit_jobs.py:

.. literalinclude:: submit_jobs.py

calc_polarizability.py

.. literalinclude:: calc_polarizability.py 
