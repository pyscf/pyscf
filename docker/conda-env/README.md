Docker image to build conda release

## How to build
docker build -t pyscf/pyscf-conda-env:latest .
docker push pyscf/pyscf-conda-env:latest
