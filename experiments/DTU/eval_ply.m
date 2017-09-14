function eval_acc_compl = eval_ply(cSet, DataInName, EvalName)
% cSet: model index
% DataInName: input '.ply' file name
% EvalName: save the evaluation into this '.mat' file
Mesh = plyread(DataInName);
Qdata=[Mesh.vertex.x Mesh.vertex.y Mesh.vertex.z]';
[dataPath,resultsPath]=getPaths();
dst=0.2;    %Min dist between points when reducing


BaseEval=PointCompareMain(cSet,Qdata,dst,dataPath);

save(EvalName,'BaseEval');

% eval_acc_compl = [mean(BaseEval.Ddata), median(BaseEval.Ddata), mean(BaseEval.Dstl), median(BaseEval.Dstl)]
eval_acc_compl = [mean(BaseEval.Ddata .* BaseEval.DataInMask), median(BaseEval.Ddata .* BaseEval.DataInMask), mean(BaseEval.Dstl .* BaseEval.StlAbovePlane), median(BaseEval.Dstl .* BaseEval.StlAbovePlane)]

