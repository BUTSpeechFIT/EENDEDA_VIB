#!/bin/bash

# Copyright 2024 Brno University of Technology, National Institute of Informatics (authors: Lin Zhang)
# Licensed under the MIT license.

MAIN_DIR=
WORK_DIR=
OUT_DIR=

kld_weight_type=none
zkld_loss_weight=
ekld_loss_weight=
num_zsamples=
num_esamples=
average_type=
permutation_type=
KLD_cal=sum
use_mu_z=
use_mu_e=
temp=
seed=
zKLD_invalid=

train_conf_name=train

con_args=

. parse_options.sh



cd ${WORK_DIR}

conf_file=${WORK_DIR}/${train_conf_name}.yaml
conf_name=`basename "${conf_file}"`
cp ${conf_file} ${OUT_DIR}/

#modify conf yaml
echo "###variational parameters for train
average_type: ${average_type}
zkld_loss_weight: ${zkld_loss_weight} 
ekld_loss_weight: ${ekld_loss_weight} 
kld_weight_type: ${kld_weight_type}
permutation_type: ${permutation_type}
output_path: ${OUT_DIR}" >> ${OUT_DIR}/${conf_name}

if [[ "$use_mu_z" == "True"  ]]; then
   echo "use_mu_z: True" >> ${OUT_DIR}/${conf_name}
else
   echo "num_zsamples: ${num_zsamples}" >> ${OUT_DIR}/${conf_name}
fi

if [[ "$use_mu_e" == "True"  ]]; then
   echo "use_mu_e: True" >> ${OUT_DIR}/${conf_name}
else
   echo "num_esamples: ${num_esamples}" >> ${OUT_DIR}/${conf_name}
fi

if [[ ! -z $temp ]]; then
    echo "temp: ${temp}" >> ${OUT_DIR}/${conf_name}
fi

if [[ ! -z $early_stop_sep ]]; then
    echo "early_stop_step: ${early_stop_step}" >> ${OUT_DIR}/${conf_name}
fi

if [[ ! -z $KLD_cal ]]; then
    echo "KLD_cal: ${KLD_cal}" >> ${OUT_DIR}/${conf_name}
fi

if [[ ! -z $zKLD_invalid ]]; then
    echo "zKLD_invalid: ${zKLD_invalid}" >> ${OUT_DIR}/${conf_name}
fi

if [[ ! -z $seed ]]; then
    sed -i "s/seed: 3/seed: $seed/g" ${OUT_DIR}/${conf_name}
fi

python ${MAIN_DIR}/eendedavib/train.py -c ${OUT_DIR}/${conf_name}

