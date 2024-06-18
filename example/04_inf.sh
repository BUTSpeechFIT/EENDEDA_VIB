#!/bin/bash

# Copyright 2024 Brno University of Technology, National Institute of Informatics (authors: Lin Zhang)
# Licensed under the MIT license.

#1. env
MAIN_DIR="/path/to/EENDEDA_VIB" #TODO
WORK_DIR=`pwd`
SAVE_DIR=${WORK_DIR}/exp

#2. initializes the environment:
source ~/.bashrc
conda activate EEND_CPU

#When you run
#1. general attribute: model_dir, around +++
#2. assign the specific attribute around ---
#2. 

set -x
cd ${WORK_DIR}


stage=0
#0: infer.py: inference to generate rttm
#1: calculate DER
#2: visulize and record results.
name=s1_wz0.0001_nz1_we0.0001_ne1_pitmu_avgprob+kldmean_invKLD_A0_adap_FT0_ft
epochs=0
inf_yaml=callhome_part1_2spk

INF_GT_RTTM=/path/to/roundturth/rttm #TODO

# config for inference.
# use mu for inference: use_mu_* = True
# use multiple sampling: use_mu_* = False, inf_num_* > 0
inf_num_zsamples=0 
inf_num_esamples=0 
use_mu_z=True 
use_mu_e=True 

infer_type=avgprob
temp=

#######################
OUT_DIR=${SAVE_DIR}/${name}

if [[ "${use_mu_z}" == "True" ]]; then
     inf_num_esamples=0
     con_args+=' --use_mu_z True'
else
     con_args+=" --inf_num_zsamples ${inf_num_zsamples}"
fi
if [[ "${use_mu_e}" == "True" ]]; then
     inf_num_esamples=0
     con_args+=' --use_mu_e True'
else
     con_args+=" --inf_num_esamples ${inf_num_esamples}"
fi

if [[ -n ${temp} ]]; then
	con_args+=" --temp ${temp}"
fi


#######################RUN--

bash ${MAIN_DIR}/infer.sh $con_args \
   --stage ${stage} \
   --inf_name ${inf_yaml} \
   --INF_GT_RTTM ${INF_GT_RTTM} \
   --epochs ${epochs} \
   --OUT_DIR ${OUT_DIR} \
   --infer_type ${infer_type} \
   --MAIN_DIR ${MAIN_DIR} \
   --WORK_DIR ${WORK_DIR} \
   >${OUT_DIR}/infer.out 2> ${OUT_DIR}/infer.err || exit 1 

