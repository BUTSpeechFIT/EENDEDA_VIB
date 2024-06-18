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


set -x
cd ${WORK_DIR}

###########3#configuration file for adaotation data.
#++++

init_model=s1_wz0.0001_nz1_we0.0001_ne1_pitmu_avgprob+kldmean_invKLD_A0_adap
init_epochs=0 #Epoch(s) of the trained model. Can be a single bumber or a range like 10-20
init_model_conf=${SAVE_DIR}/${init_model}/adap.yaml 

ft_yaml_name="ft"
con_args="" #additional parameters
OUT_DIR=${SAVE_DIR}/${init_model}_FT${init_epochs}_${ft_yaml_name}

#######################RUN--
mkdir -p ${OUT_DIR}

bash ${MAIN_DIR}/adap.sh $con_args \
     --init_model ${init_model} --init_epochs ${init_epochs} \
     --init_model_conf ${init_model_conf} \
     --ft_data_name ${ft_yaml_name}\
     --OUT_DIR ${OUT_DIR} \
     --MAIN_DIR ${MAIN_DIR} \
     --WORK_DIR ${WORK_DIR} \
	>${OUT_DIR}/ft.out 2> ${OUT_DIR}/ft.err || exit 1

