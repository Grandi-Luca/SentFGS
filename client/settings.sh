#!/bin/bash

FOLDER_SPRING=/home/grandi/project/server/spring #
PATH_MODEL=${FOLDER_SPRING}/AMR3.parsing.pt
PATH_AMR_SENTS=/home/grandi/project/graphs
PATH_DOC_SENTS=/home/grandi/project/documents

if ! [ -d ${PATH_AMR_SENTS} ]
then
  mkdir ${PATH_AMR_SENTS}
fi

if ! [ -d ${PATH_DOC_SENTS} ]
then
  mkdir ${PATH_DOC_SENTS}
fi

python3 -u init_model_inside_server.py \
  --checkpoint ${PATH_MODEL} \
  --penman-linearization \
  --use-pointer-tokens \
  --amrs-folder ${PATH_AMR_SENTS}

python3 -u test_generation.py