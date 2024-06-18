#!/bin/bash
# Copyright 2024 Brno University of Technology, National Institute of Informatics (authors: Lin Zhang)
# Licensed under the MIT license.
#1. env
MAIN_DIR="/path/to/EENDEDA_VIB" #TODO
WORK_DIR=`pwd`
SAVE_DIR=${WORK_DIR}/exp

#2. initializes the environment:
source ~/.bashrc
conda activate EEND_GPU

#When you run
#1. general attribute: model_dir, around +++
#2. assign the specific attribute around ---
#2. 

cd ${WORK_DIR}
set -x

############configuration file for adaotation data.

init_model=s1_wz0.0001_nz1_we0.0001_ne1_pitmu_avgprob+kldmean_invKLD
init_epochs=0 #Epoch(s) of the trained model. Can be a single bumber or a range like 10-20
init_model_conf=${SAVE_DIR}/${init_model}/train.yaml #Keep other parameters same as the initial trained model. 

adap_data_name="adap" #specify the yaml file.

con_args="" #additional parameters
OUT_DIR=${SAVE_DIR}/${init_model}_A${init_epochs}_${adap_data_name}


#######################RUN--
mkdir -p ${OUT_DIR}

bash ${MAIN_DIR}/adap.sh $con_args \
     --init_model ${init_model} --init_epochs ${init_epochs} \
     --init_model_conf ${init_model_conf} \
     --adap_data_name ${adap_data_name}\
     --OUT_DIR ${OUT_DIR} \
     --MAIN_DIR ${MAIN_DIR} \
     --WORK_DIR ${WORK_DIR} \
	>${OUT_DIR}/adap.out 2> ${OUT_DIR}/adap.err || exit 1

