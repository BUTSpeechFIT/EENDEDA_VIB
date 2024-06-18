#!/bin/bash

# Copyright 2024 Brno University of Technology, National Institute of Informatics (authors: Lin Zhang)
# Licensed under the MIT license.
#set -x

#1. env
MAIN_DIR="/path/to/EENDEDA_VIB" #TODO
WORK_DIR=`pwd`
SAVE_DIR=${WORK_DIR}/exp

#2. initializes the environment:
source ~/.bashrc
conda activate EEND_GPU

cd ${WORK_DIR}

#3. config VIB-related parameters

zkld_loss_weight=0.0001  # weight for attractor's KLD
num_zsamples=1           # sampling number for attractor 
ekld_loss_weight=0.0001  # weight for emb's KLD
num_esamples=1           # sampling number for emb 
permutation_type=mu      # sample #mu 
average_type=prob   
use_mu_z=False 
use_mu_e=False 
KLD_cal=mean
zKLD_invalid=True
seed=1

    # When we set 0 to the weight for KLD, it make no sense to sampling.
if [ $zkld_loss_weight -eq 0  ] && [ $ekld_loss_weight -eq 0 ]; then
    num_zsamples=0 
    num_esamples=0 
    use_mu_z=True 
    use_mu_e=True 
fi


con_args="--zkld_loss_weight ${zkld_loss_weight} --ekld_loss_weight ${ekld_loss_weight}
          --permutation_type ${permutation_type} --average_type ${average_type} "


# assign specific name based on config.
if [[ ${average_type}  == "llr"  ]]; then
    name+=wz${zkld_loss_weight// /_}_we${ekld_loss_weight// /_}_avg${average_type}
else
    if [[ "${use_mu_z}" == "True" ]]; then
         num_zsamples=0
         con_args+=' --use_mu_z True'
    else
         con_args+=" --num_zsamples ${num_zsamples}"
    fi
    if [[ "${use_mu_e}" == "True" ]]; then
         num_esamples=0
         con_args+=' --use_mu_e True'
    else
         con_args+=" --num_esamples ${num_esamples}"
    fi
    name+=wz${zkld_loss_weight// /_}_nz${num_zsamples}_we${ekld_loss_weight// /_}_ne${num_esamples}_pit${permutation_type}_avg${average_type}
fi



if [[ ! -z $early_stop_sep ]]; then
    con_args+=" --early_stop_step ${early_stop_step}"
fi

if [[ ! -z "${KLD_cal}" ]]; then
    con_args+=" --KLD_cal ${KLD_cal}"
    name+="+kld${KLD_cal}"
fi

if [[ ! -z "${seed}" ]]; then
    con_args+=" --seed ${seed}"
    name=s${seed}_${name}
fi

if [[ "${zKLD_invalid}" == "True" ]]; then
    con_args+=" --zKLD-invalid ${zKLD_invalid}"
    name+="_invKLD"
fi

OUT_DIR=${SAVE_DIR}/${name}
mkdir -p ${OUT_DIR}


bash ${MAIN_DIR}/train.sh \
	${con_args} \
	--train_conf_name train \
	--OUT_DIR ${OUT_DIR} \
	--MAIN_DIR ${MAIN_DIR} \
	--WORK_DIR ${WORK_DIR}
        	




