#!/bin/bash

set -ex
set -o pipefail

if [ ! -f setup.py ]; then
  echo "setup.py must exist in the directory that is being packaged and published."
  exit 1
fi

if [ ! -f conda/meta.yaml ]; then
  echo "meta.yaml must exist in the directory that is being packaged and published."
  exit 1
fi

# Build for Linux
conda build --output-folder . conda

export ANACONDA_API_TOKEN=$ANACONDATOKEN
anaconda upload linux-64/*.tar.bz2
