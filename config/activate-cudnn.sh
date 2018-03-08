export CUDNN_ROOT=/usr/local/cuda   #  set conda-env specific cudnn path, e.g.: /home/<your-user-name>/libs/cudnn 
export LD_LIBRARY_PATH_CUDNN_BACKUP="$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH=$CUDNN_ROOT/lib64:$LD_LIBRARY_PATH
export CPATH_CUDNN_BACKUP="$CPATH"
export CPATH=$CUDNN_ROOT/include:$CPATH
export LIBRARY_PATH_CUDNN_BACKUP="$LIBRARY_PATH"
export LIBRARY_PATH=$CUDNN_ROOT/lib64:$LIBRARY_PATH
