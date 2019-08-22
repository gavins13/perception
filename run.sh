folder=9.0.176
folder2=8.0.61-cudnn.7.0.2
export PATH=$PATH:/vol/cuda/$folder:/vol/cuda/$folder/lib64:/vol/cuda/$folder2:/vol/cuda/$folder2/lib64
export LIBRARY_PATH=$LIBRARY_PATH:/vol/cuda/$folder:/vol/cuda/$folder/lib64:/vol/cuda/$folder2:/vol/cuda/$folder2/lib64
export LD_LIBRARY_PATH=:/vol/cuda/$folder:/vol/cuda/$folder/lib64:/vol/cuda/$folder2:/vol/cuda/$folder2/lib64
export CPATH=$CPATH:/vol/cuda/$folder:/vol/cuda/$folder/lib64:/vol/cuda/$folder2:/vol/cuda/$folder2/lib64
echo $LD_LIBRARY_PATH
export TF_CPP_MIN_LOG_LEVEL=0
export TF_CPP_MIN_VLOG_LEVEL=0 # usually = 1 is okay for debugging
#python3 -m memory_profiler experiment.py
python3 experiment.py
