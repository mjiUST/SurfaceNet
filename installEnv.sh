#!/bin/bash -evx

conda_env_name=SurfaceNet

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

cd -  # cd back to SurfaceNet work dir

# create new environment
if conda info --env | grep -w "$conda_env_name" >/dev/null; then
    echo "The '$conda_env_name' conda env exists."    # but the env will be visible even thoug the installation was terminated. In this case you should delete this conda env first.
else    # if the env does not exist.
    conda create -n $conda_env_name --file config/conda_list_explicit.txt    # clone conda packages, (very large)
fi
# assume the miniconda path is ~/miniconda2/
mkdir -p ~/miniconda2/envs/$conda_env_name/etc/conda/activate.d/
cp ./config/act* ~/miniconda2/envs/$conda_env_name/etc/conda/activate.d/   # before copy *PLEASE* change the CUDA/CUDNN path in the first line of these files accordingly
mkdir -p ~/miniconda2/envs/$conda_env_name/etc/conda/deactivate.d/
cp ./config/de* ~/miniconda2/envs/$conda_env_name/etc/conda/deactivate.d/
. activate $conda_env_name      # seperate environment. Can use command `which python` to check the path of the local python/packages

# install pip packages
# pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip    
pip install git+git://github.com/Lasagne/Lasagne.git@7992faa
pip install ipdb plyfile progressbar

# config .theanorc,
if ls ~/.theanorc >/dev/null; then cp ~/.theanorc ~/.theanorc_SurfaceNet_backup; echo ".theanorc_SurfaceNet_backup was backed up"; fi
echo -e "[global] \nfloatX=float32 \ndevice=cuda \noptimizer=fast_run \n \nallow_gc=True \ngpuarray.preallocate=-1 \n \nnvcc.fastmath=True \n" > ~/.theanorc

# test lasagne:
echo "Try to import lasagne. If there is no error, congratulations!"
python -c "import lasagne"

