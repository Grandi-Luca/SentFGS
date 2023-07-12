#!/bin/bash

PROJECT_DIR="/home/grandi/project"     # change project dir here

DataPath=$PROJECT_DIR/documents/run_1  # change data path if multiple tasks are runnging in parallel
OutputDir=$PROJECT_DIR/graphs/run_1    # change output path if multiple tasks are runnging in parallel

export PARSING_MODEL="spring"          # change parsing model based on which image is active
export PORT=$1

if [ ! -d ${OutputDir} ];then
  mkdir -p ${OutputDir}
fi

if [ ! -d ${DataPath} ];then
  mkdir -p ${DataPath}
fi

if [[ "$PARSING_MODEL" == "amrbart" ]]
  then
    RootDir="${PROJECT_DIR}/server/AMRBART/fine-tune"
    BasePath=$PROJECT_DIR/server/AMRBART

    ModelCache=$BasePath/.cache
    DataCache=$DataPath/.cache/dump-amrparsing

    export HF_DATASETS_CACHE=$DataCache

    if [ ! -d ${DataCache} ];then
      mkdir -p ${DataCache}
    fi

    python3 -u init_model_inside_server.py \
      --config_file "$PROJECT_DIR/server/AMRBart_config.json" \
      --data_dir $DataPath \
      --output_dir $OutputDir \
      --model_cache_dir $ModelCache \
      --data_cache_dir $DataCache

elif [[ "$PARSING_MODEL" == "spring" ]]
  then
    FOLDER_SPRING=$PROJECT_DIR/server/spring
    PATH_MODEL=${FOLDER_SPRING}/checkpoints/2_2_0.7602.pt
    CONFIG_PATH=${FOLDER_SPRING}/checkpoints/config.json

    python3 -u init_model_inside_server.py \
      --checkpoint ${PATH_MODEL} \
      --penman-linearization \
      --use-pointer-tokens \
      --amrs-folder ${OutputDir} \
      --config_path ${CONFIG_PATH}
      # --model "facebook/bart-large"

elif [[ "$PARSING_MODEL" == "amrlib" ]]
  then
    PATH_MODEL="${PROJECT_DIR}/server/AMRLIB/checkpoints/model_parse_xfm_bart_large-v0_1_0"
    
    python3 -u init_model_inside_server.py \
      --output_dir $OutputDir \
      --model_dir $PATH_MODEL
fi

python3 -u test_functionality.py \
  --data_dir $DataPath \
  --output_dir $OutputDir \
  # --amrlib-model-path "${PROJECT_DIR}/server/AMRLIB/checkpoints/model_parse_xfm_bart_large-v0_1_0"