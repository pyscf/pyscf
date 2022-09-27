FROM continuumio/miniconda3:4.7.10

RUN conda install -y anaconda-client conda-build

COPY build-conda.sh /build-conda.sh
CMD ["/build-conda.sh"]

RUN cp /opt/conda/lib/python3.7/_sysconfigdata_x86_64_conda_cos6_linux_gnu.py /opt/conda/lib/python3.7/_sysconfigdata_x86_64_conda_linux_gnu.py
RUN conda config --set anaconda_upload yes
