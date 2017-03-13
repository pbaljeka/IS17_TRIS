#!/bin/bash

UTILSDIR='/home/pbaljeka/TRIS_Exps3/utils/'
VOXDIR='/home/pbaljeka/TRIS_Exps3/cmu_us_slt/festival/'
#cat $VOXDIR/coeffs/*.tris > $UTILSDIR/slt.tris
#copy phonenames, statenames
cp $VOXDIR/clunits/statenames $UTILSDIR/
cp $VOXDIR/clunits/phonenames $UTILSDIR/

#make states
cat $UTILSDIR/statenames|sed 's+___+\n+g' >$UTILSDIR/states

#make question list
cat $VOXDIR/clunits/mcep.desc|grep "("|sed 's+)++g'|sed 's+(++g'|awk '{print $1}'|sed 's+occurid++g'|sed '/^\s*$/d'>$UTILSDIR/question_list

#make nodenames, senones
cat $UTILSDIR/slt.tris |awk '{print $1}'|grep -v "L"|sort|uniq >$UTILSDIR/nodenames
cat $UTILSDIR/slt.tris |grep "L"|awk '{print $2, $1}'|sort|uniq|sed 's+L++g' |sort -n> $UTILSDIR/senone_numbers
cat $UTILSDIR/senone_numbers|awk '{print $2}' >$UTILSDIR/senones

#make train,test, val list - utterancewise
cat ${VOXDIR}../etc/txt.done.data.train.train|awk '{print $2}' >$UTILSDIR/train_list
cat ${VOXDIR}../etc/txt.done.data.train.test|awk '{print $2}' >$UTILSDIR/val_list
cat ${VOXDIR}../etc/txt.done.data.test|awk '{print $2}' >$UTILSDIR/test_list
cat $UTILSDIR/train_list $UTILSDIR/val_list $UTILSDIR/test_list >$UTILSDIR/all_list
#make train, list of all nodes
cat $UTILSDIR/senones $UTILSDIR/nodenames > $UTILSDIR/allnodes 
##
mkdir -p $VOXDIR/tris/
for i in `cat $UTILSDIR/nodenames`;
do
    cat $UTILSDIR/slt.tris|awk '$1=="'$i'"' >$VOXDIR/tris/${i}.tris
done
for i in `cat $UTILSDIR/senones`;
do
    f=${i}L
    cat $UTILSDIR/slt.tris|awk '$1=="'$f'"' >$VOXDIR/tris/${i}.tris
done

