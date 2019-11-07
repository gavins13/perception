#!/bin/bash

export TF_CPP_MIN_LOG_LEVEL=0
export TF_CPP_MIN_VLOG_LEVEL=0 # usually = 1 is okay for debugging

for i in "$@"
do
case $i in
    -g=*|--gpu=*)
    export CUDA_VISIBLE_DEVICES="${i#*=}"
    shift # past argument=value
    ;;
    -n=*|--name=*)
    EXP_NAME="${i#*=}"
    shift # past argument=value
    ;;
    -d=*|--desc=*)
    EXP_DESC="${i#*=}"
    shift # past argument=value
    ;;
    -t=*|--tf_roll_version=*)
    TF_ROLL_VERSION="${i#*=}"
    shift # past argument=value
    ;;
    -w=*|--tf_while_loop_version=*)
    TF_WHILE_LOOP_VERSION="${i#*=}"
    shift # past argument=value
    ;;
    -o=*|--old=*)
    OLD="${i#*=}"
    shift # past argument=value
    ;; 
    --no-email)
    NO_EMAIL=YES
    shift # past argument with no value
    ;;
    -d=*|--debug=*)
    export TF_CPP_MIN_VLOG_LEVEL="${i#*=}"
    shift # past argument=value
    ;;
    -u=*|--undersampling_rate=*)
    MERUP_UNDERSAMPLING_RATE="${i#*=}"
    shift # past argument=value
    ;;
    -p=*|--eval_folder_prefix=*)
    EVAL_FOLDER_PREFIX="${i#*=}"
    shift # past argument=value
    ;;
    --type=*)
    TYPE="${i#*=}"
    shift # past argument=value
    ;;
    --experiment_id=*)
    EXPERIMENT_ID="${i#*=}"
    shift # past argument=value
    ;;
    # --default)
    #DEFAULT=YES
    #shift # past argument with no value
    #;;
    *)
          # unknown option
    ;;
esac
done

if [ "$OLD" = '' ]; then
  #export PYTHONPATH=
  export CUDA_PATH=/vol/cuda/10.0.130-cudnn7.6.4.38
  export CUDA_INC_PATH=/vol/cuda/10.0.130-cudnn7.6.4.38
  export LD_LIBRARY_PATH=/vol/cuda/10.0.130-cudnn7.6.4.38/lib64:/vol/cuda/10.0.130-cudnn7.6.4.38:$LD_LIBRARY_PATH
else
  folder=9.0.176
  folder2=8.0.61-cudnn.7.0.2
  export PATH=$PATH:/vol/cuda/$folder:/vol/cuda/$folder/lib64:/vol/cuda/$folder2:/vol/cuda/$folder2/lib64
  export LIBRARY_PATH=$LIBRARY_PATH:/vol/cuda/$folder:/vol/cuda/$folder/lib64:/vol/cuda/$folder2:/vol/cuda/$folder2/lib64
  export LD_LIBRARY_PATH=:/vol/cuda/$folder:/vol/cuda/$folder/lib64:/vol/cuda/$folder2:/vol/cuda/$folder2/lib64
  export CPATH=$CPATH:/vol/cuda/$folder:/vol/cuda/$folder/lib64:/vol/cuda/$folder2:/vol/cuda/$folder2/lib64
  echo $LD_LIBRARY_PATH
fi

#python3 -m memory_profiler experiment.py

if [ "$MERUP_UNDERSAMPLING_RATE" = '' ]; then
  MERUP_UNDERSAMPLING_RATE=''
else
  MERUP_UNDERSAMPLING_RATE='undersampling_factor='$MERUP_UNDERSAMPLING_RATE
fi


if [ "$EVAL_FOLDER_PREFIX" = '' ]; then
  EVAL_FOLDER_PREFIX=''
else
  EVAL_FOLDER_PREFIX='eval_folder_prefix='$EVAL_FOLDER_PREFIX
fi


if [ "$EXPERIMENT_ID" = '' ]; then
  EXPERIMENT_ID=''
else
  EXPERIMENT_ID='experiment_id='$EXPERIMENT_ID
fi


if [ "$TYPE" = '' ]; then
  TYPE=''
else
  TYPE='type='$TYPE
fi


if [ "$CUDA_VISIBLE_DEVICES" = '' ]; then
  export CUDA_VISIBLE_DEVICES=3
fi
echo "GPU BEING USED  = ${CUDA_VISIBLE_DEVICES}"
if [ "$1" = 'memory' ]; then
  echo "memory testing"
  #python3 -m memory_profiler experiment.py
else
  echo "no memory testing"
  echo "TF_ROLL_VERSION"
  echo $TF_ROLL_VERSION
  python3 experiment.py tf_roll_version=$TF_ROLL_VERSION tf_while_loop_version=$TF_WHILE_LOOP_VERSION $MERUP_UNDERSAMPLING_RATE $EVAL_FOLDER_PREFIX $TYPE $EXPERIMENT_ID
  #python3 -m pip uninstall tensorflow tensorflow-gpu
  #pip install tensorflow==1.15 tensorflow-gpu==1.15 --no-cache-dir
fi
if [ "$NO_EMAIL" = '' ]; then
  curl -s "https://www.doc.ic.ac.uk/~kgs13/sendemail.php?name="$EXP_NAME"&desc="$EXP_DESC
  echo "Email Sent."
fi
