# SurfaceNet

M. Ji, J. Gall, H. Zheng, Y. Liu, and L. Fang. [SurfaceNet: An End-to-end 3D Neural Network for Multiview Stereopsis](https://www.researchgate.net/publication/318920947_SurfaceNet_An_End-to-end_3D_Neural_Network_for_Multiview_Stereopsis). ICCV, 2017

The [poster pdf](https://www.researchgate.net/publication/321126305_ICCV2017_SurfaceNet_poster) is also available.

![SurfaceNet experiment results](figures/experiment.png?raw=true "SurfaceNet experiment results")
![SurfaceNet pipeline](figures/pipeline.png?raw=true "SurfaceNet pipeline")

## How to run

1. install [Nvidia driver 375 + cuda 8.0 + cudnn v5.1](https://github.com/mjiUST/driver_cuda_cudnn)
2. install the conda environment by: `bash installEnv.sh`
    * DON'T WORRY, conda will generate an isolated environment for SurfaceNet with python2.7, anaconda, theano, ... etc. That means all your libraries / packeges' version will not be affacted, at the same time the `~/.bashrc` file will not be changed.
    * before you run, PLEASE change the CUDA/CUDNN path in the files: 
        - `./config/activate-cuda.sh` change the 1st line to your cuda path, e.g.: `export CUDA_ROOT=/usr/local/cuda`
        - `./config/activate-cudnn.sh` change the 1st line to your cudnn path, e.g.: `export CUDNN_ROOT=/home/<your-user-name>/libs/cudnn`
3. download the network model to the folder "./inputs/SurfaceNet_models" from the Dropbox [folder](https://www.dropbox.com/sh/8xs0u57ikj4qfvr/AADRQFQyJfG3WfH7ZvpcWmMKa?dl=0)
4. if the conda environment has been installed, one can activate it by: `. activate SurfaceNet`; deactivate it by: `. deactivate`.
5. in terminal run: `python main.py` 

## Evaluation results

Some evaluation results are uploaded, including '.ply' files and the detailed number of Table 3. This could be helpful if you want to compare with this work.

## License

SurfaceNet is released under the MIT License (refer to the LICENSE file for details).

## Citing SurfaceNet

If you find SurfaceNet useful in your research, please consider citing:

    @inproceedings{ji2017surfacenet,
      title={SurfaceNet: An End-To-End 3D Neural Network for Multiview Stereopsis},
      author={Ji, Mengqi and Gall, Juergen and Zheng, Haitian and Liu, Yebin and Fang, Lu},
      booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
      pages={2307--2315},
      year={2017}
    }
