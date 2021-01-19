# You can run build.sh with bash -x -e. The -x makes it echo each command that is run
# and the -e makes it exit whenever a command in the script returns nonzero exit status.
set -x -e

cd ../pyscf/lib
mkdir build
cd build
cmake ..
make
cd ../../conda

$PYTHON setup.py install --single-version-externally-managed --record record.txt
