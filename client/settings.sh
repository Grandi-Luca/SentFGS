#!/bin/bash

FOLDER_SPRING=../server/spring
PATH_MODEL=${FOLDER_SPRING}/checkpoints
PATH_AMR_SENTS=../graphs
PATH_DOC_SENTS=../documents

if ! [ -d ${PATH_AMR_SENTS} ]
then
  mkdir ${PATH_AMR_SENTS}
fi

if ! [ -d ${PATH_DOC_SENTS} ]
then
  mkdir ${PATH_DOC_SENTS}
fi

if ! [ -d ${PATH_MODEL} ]
then
  mkdir ${PATH_MODEL}
fi

python3 -u init_model_inside_server.py \
  --checkpoint ${PATH_MODEL}/AMR3.parsing.pt \
  --penman-linearization \
  --use-pointer-tokens \
  --amrs-folder ${PATH_AMR_SENTS}

# run the experiments
python3 -u test_generation.py ${PATH_AMR_SENTS} ${PATH_DOC_SENTS}