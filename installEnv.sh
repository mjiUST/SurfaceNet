#!/bin/bash -evx

conda_env_name=SurfaceNet
cuda_root=/usr/local/cuda

# set -e # To exit the script as soon as one of the commands failed
cd ~/Downloads

# install miniconda
if which conda >/dev/null; then
    echo "conda is available, skip the conda installation."
else
    echo "conda command is not available, preparing to install Miniconda. Please accept it"
    wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
    bash Miniconda2-latest-Linux-x86_64.sh
    . ~/.bashrc    # to enable the conda command
fi

# config .bashrc, can change the CUDA_ROOT path
echo 'export CUDA_ROOT=$cuda_root' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_ROOT/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export CPATH=$CUDA_ROOT/include:$CPATH' >> ~/.bashrc
echo 'export LIBRARY_PATH=$CUDA_ROOT/lib64:$LIBRARY_PATH' >> ~/.bashrc
. ~/.bashrc    # source before `source activate conda_env`

# create new environment
if conda info --env | grep -w "$conda_env_name" >/dev/null; then
    echo "The '$conda_env_name' conda env exists."    # but the env will be visible even thoug the installation was terminated. In this case you should delete this conda env first.
else    # if the env does not exist.
    conda create -n $conda_env_name python=2.7 anaconda    # install anaconda with the majority of the depandencies, (very large)
fi
. activate $conda_env_name      # seperate environment. Can use command `which python` to check the path of the local python/packeges

# install packeges
conda install theano pygpu --yes    # http://www.deeplearning.net/software/theano/install_ubuntu.html#requirements-installation-through-conda-recommended
pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip    # http://lasagne.readthedocs.io/en/latest/user/installation.html#bleeding-edge-version
pip install plyfile progressbar

# config .theanorc,
if ls ~/.theanorc >/dev/null; then cp ~/.theanorc ~/.theanorc_backup; echo ".theanorc_backup was backed up"; fi
echo -e "[global] \nfloatX=float32 \ndevice=cuda \noptimizer=fast_run \n \nallow_gc=True \ngpuarray.preallocate=-1 \n \nnvcc.fastmath=True \n \n[cuda] \nroot=/usr/local/cuda" > ~/.theanorc

# test lasagne:
echo "Try to import lasagne. If there is no error, congratulations!"
python -c "import lasagne"

