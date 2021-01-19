# You can run build.sh with bash -x -e. The -x makes it echo each command that is run
# and the -e makes it exit whenever a command in the script returns nonzero exit status.
set -x -e

conda install numpy

$PYTHON -m pip install . -vv
