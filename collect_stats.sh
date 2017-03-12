#!/bin/bash

UTILSDIR='/home/pbaljeka/TRIS_Exps3/utils'
#Collect unique questions
mkdir -p festival/tris/
for i in `cat etc/non_terminal_questions`;
do
    cat festival/coeffs/$VOICENAME.tris|awk '$1=="'$i'"' >festival/tris/${i}.tris
done
mkdir -p festival/tris_terminals/
for i in `cat etc/terminal_questions`;
do
    cat festival/coeffs/$VOICENAME.tris|awk '$1=="'$i'"' >festival/tris_terminals/${i}.tris
done

