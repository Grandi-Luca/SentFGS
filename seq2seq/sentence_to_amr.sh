#!/bin/bash

FOLDER_SPRING=/home/grandi/project/spring #
PATH_AMR_SENTS=amrs_graph/ #
PATH_MODEL=${FOLDER_SPRING}/AMR3.parsing.pt

if ! [ $# -eq 0 ]
then
  if ! [ -d ${PATH_AMR_SENTS} ]
  then
    mkdir ${PATH_AMR_SENTS}
  fi
  for file in $@
  do
    python3 -u ${FOLDER_SPRING}/predict_amrs_from_plaintext.py \
      --checkpoint ${PATH_MODEL} \
      --texts $file \
      --penman-linearization \
      --use-pointer-tokens > ${PATH_AMR_SENTS}$(basename -- $file .txt).amr.txt

    # echo `cat ${PATH_AMR_SENTS}$(basename -- $file .txt).amr.txt` >> /home/grandi/project/test_amrs.amr.txt
  done
fi

echo ${PATH_AMR_SENTS} >&1
