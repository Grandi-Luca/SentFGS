#!/bin/bash

FOLDER_SPRING=/home/grandi/project/server/spring #
PATH_AMR_SENTS=amrs_graph/ #
PATH_MODEL=${FOLDER_SPRING}/AMR3.parsing.pt

if ! [ $# -eq 0 ]
then
  if ! [ -d ${PATH_AMR_SENTS} ]
  then
    mkdir ${PATH_AMR_SENTS}
  fi
  file=$1
  output_file=$2
  if ! [ -z "$output_file" ]
  then
    python3 -u ${FOLDER_SPRING}/predict_amrs_from_plaintext.py \
      --checkpoint ${PATH_MODEL} \
      --texts $file \
      --penman-linearization \
      --use-pointer-tokens \
      --output-file-name ${PATH_AMR_SENTS}${output_file}.amr.txt
    
    echo ${PATH_AMR_SENTS}${output_file}.amr.txt >&1
  
  else
    python3 -u ${FOLDER_SPRING}/predict_amrs_from_plaintext.py \
      --checkpoint ${PATH_MODEL} \
      --texts $file \
      --penman-linearization \
      --use-pointer-tokens \
      --output-file-name ${PATH_AMR_SENTS}$(basename -- $file .txt).amr.txt

    echo ${PATH_AMR_SENTS}${output_file}.amr.txt >&1

  fi
fi

