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
    -p=*|--port=*)
    export PORT="${i#*=}"
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
  export CUDA_PATH=/vol/biomedic/users/kgs13/Software/cuda_envs/9.2.148-cudnn7.2.1/cuda
  export CUDA_INC_PATH=/vol/biomedic/users/kgs13/Software/cuda_envs/9.2.148-cudnn7.2.1/cuda/include
  export LD_LIBRARY_PATH=/vol/biomedic/users/kgs13/Software/cuda_envs/9.2.148-cudnn7.2.1/cuda/lib64:/vol/biomedic/users/kgs13/Software/cuda_envs/9.2.148-cudnn7.2.1/cuda/:$LD_LIBRARY_PATH
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




if [ "$CUDA_VISIBLE_DEVICES" = '' ]; then
  export CUDA_VISIBLE_DEVICES=
fi

if [ "$PORT" = '' ]; then
  export PORT=8889
fi

echo "GPU BEING USED  = ${CUDA_VISIBLE_DEVICES}"

echo "no memory testing"
echo "TF_ROLL_VERSION"
echo $TF_ROLL_VERSION

cd /
jupyter-notebook --port=$PORT --ip=146.169.26.137
#python3 experiment.py tf_roll_version=$TF_ROLL_VERSION tf_while_loop_version=$TF_WHILE_LOOP_VERSION
