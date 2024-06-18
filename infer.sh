#!/bin/bash

# Copyright 2024 Brno University of Technology, National Institute of Informatics (authors: Lin Zhang)
# Licensed under the MIT license.

source ~/.bashrc

set -x

stage=
inf_name=  #data name
epochs=
MAIN_DIR=
WORK_DIR=
OUT_DIR=
temp=

#inf related conf.
INF_GT_RTTM=
plot_output=False
median=11
spk_qty=-1
spk_thr=0.5
thr=0.5
COLLARs=0.25

inf_num_zsamples=
inf_num_esamples=
use_mu_z=
use_mu_e=
infer_type=sample 



. parse_options.sh
# find the exp_name.
exp_name=$(dirname "${OUT_DIR}")
exp_name=${exp_name##*/}
main_exp_name=${exp_name} #to save a group of results.


#for inference output
show_metric=der

cd ${WORK_DIR}

inf_conf_name=infer_${inf_name}.yaml 
inf_conf_file=${WORK_DIR}/${inf_conf_name}

MODEL_DIR=${OUT_DIR}/models/ 
RTTM_DIR=${OUT_DIR}/${inf_name} 


inf_dir=spk_qty${spk_qty}_spk_qty_thr${thr}
if [[ "${use_mu_z}" == "True" ]]; then
    app_name="nzmu"		
elif [ ${inf_num_zsamples} -ge 1 ]; then
    app_name=nz${inf_num_zsamples}
else
    echo "ERROR"
    exit 1      
fi

if [[ "${use_mu_e}" == "True" ]]; then
    app_name+="_nemu"		
elif [ ${inf_num_esamples} -ge 1 ]; then
    app_name+=_ne${inf_num_esamples}
else
    echo "ERROR"
    exit 1      
fi

app_name=${app_name}_t${infer_type}  

sysdir=${RTTM_DIR}/epochs$epochs/${app_name}/timeshuffleTrue/${inf_dir}/detection_thr${spk_thr}/median${median}/

if [ $stage -le 0 ]; then
  # copy infer yaml	
  # cat conf to the file.
  mkdir -p ${sysdir}
    cp ${inf_conf_file} ${sysdir}/ 
  
echo "###variational parameters for inference" >> ${sysdir}/${inf_conf_name}
echo "epochs: ${epochs}
models_path: ${MODEL_DIR}
rttms_dir: ${RTTM_DIR}
estimate_spk_qty: ${spk_qty}
estimate_spk_qty_thr: ${thr}  
threshold: ${spk_thr} 
ref_rttms_dir: ${INF_GT_RTTM}  
infer_type: ${infer_type} 
plot_output: ${plot_output}
median_window_length: ${median}" >> ${sysdir}/${inf_conf_name}

  if [[ -n $temp ]]; then
    echo "temp: ${temp}" >> ${sysdir}/${inf_conf_name}
  fi
  
  if [[ "$use_mu_z" == "True"  ]]; then
     echo "use_mu_z: True" >> ${sysdir}/${inf_conf_name}
  else
     echo "num_zsamples: ${inf_num_zsamples}" >> ${sysdir}/${inf_conf_name}
  fi
  
  if [[ "$use_mu_e" == "True"  ]]; then
     echo "use_mu_e: True" >> ${sysdir}/${inf_conf_name}
  else
     echo "num_esamples: ${inf_num_zsamples}" >> ${sysdir}/${inf_conf_name}
  fi
  
  python  ${MAIN_DIR}/eendedavib/infer.py \
	    -c ${sysdir}/${inf_conf_name} 
fi

