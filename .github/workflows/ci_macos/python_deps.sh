#!/usr/bin/env bash
python -m pip install --upgrade pip
pip install "numpy!=1.16,!=1.17" scipy h5py pytest pytest-cov pytest-timer
