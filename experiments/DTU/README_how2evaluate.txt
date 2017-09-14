### how to evaluate the DTU models ###

1. download the DTU MVS dataset from: http://roboimagedata.compute.dtu.dk/?page_id=36

2. use the Matlab evaluation code in the dataset
    - run the code 'DTU_home_folder/SampleSet/Matlab evaluation code/BaseEvalMain_web.m' to evaluate the methods in the folder 'DTU_home_folder/SampleSet/Matlab evaluation code/MVS Data/Points/'
    - in order to evaluate a single '.ply' point cloud data of the Nth model, can use the piece of code 'eval_ply.m' in this folder. Just input the model index & input '.ply' file & output '.mat' file path. The mean/median of accuracy/completeness are store in the variable 'eval_acc_compl'.

3. as mentioned in the Table 3 of the SurfaceNet paper, 22 models are selected for evaluation. 
    - The numbers are stored in the file 'evaluate_DTU_22models.txt', which can be imported to Microsoft Excel for better visualization and further calculation.
    - Some reconstruction results are shared in the Dropbox shared folder:
        * https://www.dropbox.com/sh/8xs0u57ikj4qfvr/AADRQFQyJfG3WfH7ZvpcWmMKa?dl=0
        * only 2 good cases are selected: "s=32,adapt\beta=6,\gamma=80%" & "s=64,\tau=0.7,\gamma=80%"


If you have any questions, feel free to contact Mengqi (mji_at_connect.ust.hk) or post issues.
