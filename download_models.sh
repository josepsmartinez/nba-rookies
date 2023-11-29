#!/bin/sh

INSTALL_DIR="${HOME}.insightface/models"
DOWNLOAD_DIR="/tmp/"

model_url="https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
install_path="${DOWNLOAD_DIR}/buffalo_l.zip"

mkdir -p ${INSTALL_DIR} &&\
wget ${model_url} -O ${install_path} &&\
unzip ${install_path} -d ${INSTALL_DIR}/buffalo_l