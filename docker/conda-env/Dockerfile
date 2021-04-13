FROM continuumio/miniconda3:4.7.10

RUN conda install -y anaconda-client conda-build

COPY build-conda.sh /build-conda.sh
CMD ["/build-conda.sh"]
