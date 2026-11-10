#!/usr/bin/env bash

set -e

yum install -y epel-release
yum-config-manager --enable epel
yum install -y openblas-devel gcc cmake curl
