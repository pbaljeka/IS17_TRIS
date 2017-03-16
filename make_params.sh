#!/bin/bash
TREEDIR='/home/pbaljeka/TRIS_Exps3/cmu_us_slt/festival/trees/'
VOICEDIR='/home/pbaljeka/TRIS_Exps3/cmu_us_slt/'
UTILSDIR='/home/pbaljeka/TRIS_Exps3/utils/'
voxname='cmu_us_slt'

rm -f  *_st
rm -f cmu_us_slt_mcep.rawparams
rm -f patch
python make_rp_td.py
sed -n 8949p $UTILSDIR/cmu_us_slt_mcep.rawparams>patch
sed -i '8949d' cmu_us_slt_mcep.rawparams
sed -i 8948rpatch cmu_us_slt_mcep.rawparams

cat ${UTILSDIR}/cmu_us_slt_mcep.rawparams|cut -d ' ' -f1,2 >f0_st
cat ${UTILSDIR}/cmu_us_slt_mcep.rawparams|cut -d ' ' -f103,104 >v0_st
#cat cmu_us_sltcg_mcep.rawparams|cut -d ' ' -f3-102 >mcep_st
#cat cmu_us_slt_mcep.rawparams|cut -d ' ' -f3-102 >mcep_st
rm -f ${TREEDIR}/${voxname}_mcep.*params
#paste -d ' ' f0_st mcep_st v0_st > ${TREEDIR}/${voxname}_mcep.rawparams
cp cmu_us_slt_mcep.rawparams ${TREEDIR}/${voxname}_mcep.rawparams
cat ${TREEDIR}/${voxname}_mcep.rawparams|awk '{print NF}'
#cat ${TREEDIR}/${voxname}_mcep.rawparams |sed 's+^ ++g'|sed 's+  + +g'
$ESTDIR/bin/ch_track -itype ascii -otype est_binary -s 0.005 -o ${TREEDIR}/${voxname}_mcep.params ${TREEDIR}/${voxname}_mcep.rawparams
rm -rf ${VOICEDIR}/test/cgp_114
rm -f ${VOICEDIR}/mcd-base.114
cd ${VOICEDIR}
$FESTVOXDIR/src/clustergen/cg_test resynth cgp_114 ${VOICEDIR}/etc/txt.done.data.test >mcd-base.114
