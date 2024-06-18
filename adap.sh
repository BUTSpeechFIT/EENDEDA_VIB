#!/bin/bash

# Copyright 2024 Brno University of Technology, National Institute of Informatics (authors: Lin Zhang)
# Licensed under the MIT license.

#use the same conf with train.

set -x

MAIN_DIR=
WORK_DIR=
OUT_DIR=
init_model=
init_epochs=
init_model_conf=
adap_data_name=adap
ft_data_name=

con_args=

. parse_options.sh


cd ${WORK_DIR}

init_model_path=${WORK_DIR}/exp/${init_model}/models

if [[ ! -z ${ft_data_name}  ]]; then
    conf_path=${WORK_DIR}/${ft_data_name}.yaml
    tag=finetune
elif [[ ! -z ${adap_data_name}  ]]; then
    conf_path=${WORK_DIR}/${adap_data_name}.yaml
    tag=adap
fi
conf_name=`basename "${conf_path}"`
cp ${conf_path} ${OUT_DIR}/


#init_conf

#modify conf yaml
echo "###variational parameters for ${tag}
init_model_path: ${init_model_path}
init_epochs: ${init_epochs}
output_path: ${OUT_DIR}" >> ${OUT_DIR}/${conf_name}

#modify conf yaml based on the init_model
awk 'BEGIN{flag=0}
{if(flag>0){
	if($0!~/output_path/ && $0!~/init/){print}
	}
 if($0~/variational/){flag=1} 
}' ${init_model_conf} >> ${OUT_DIR}/${conf_name}

python ${MAIN_DIR}/eendedavib/train.py -c ${OUT_DIR}/${conf_name}
