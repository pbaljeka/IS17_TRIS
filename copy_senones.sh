#!/bin/bash

UTILSDIR='/home/pbaljeka/TRIS_Exps3/utils/'
VOXDIR='/home/pbaljeka/TRIS_Exps3/cmu_us_slt/festival/'
model_name='slt_tris_norm_AD_1'
mkdir -p $VOXDIR/${model_name}
mkdir -p $VOXDIR/un_norm_${model_name}
for i in `cat ${UTILSDIR}/senone`; 
do
cp $VOXDIR/norm_nn_tris_trees/${i}_.mean ${VOXDIR}/predicted_norm_nn_tris_trees/${model_name}/
cp $VOXDIR/norm_nn_tris_trees/${i}_.std ${VOXDIR}/predicted_norm_nn_tris_trees/${model_name}/
done
#SPECIFIC TO THIS BECAUSE SLT HAS ZH_3 as senone
cp $VOXDIR/nn_tris_trees/zh_3_.mean ${VOXDIR}/predicted_norm_nn_tris_trees/un_norm_${model_name}/
cp $VOXDIR/nn_tris_trees/zh_3_.std ${VOXDIR}/predicted_norm_nn_tris_trees/un_norm_${model_name}/

