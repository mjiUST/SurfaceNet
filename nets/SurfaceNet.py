import lasagne
from lasagne.layers.dnn import Conv3DDNNLayer, Pool3DDNNLayer
from lasagne.layers import ElemwiseSumLayer, ReshapeLayer, SliceLayer, ConcatLayer, batch_norm, DenseLayer, NonlinearityLayer, PadLayer
from lasagne.regularization import regularize_layer_params, l2
from layers import ChannelPool_max, ChannelPool_argmaxWeight, ChannelPool_weightedAverage, Bilinear_3DInterpolation, DilatedConv3DLayer
import theano.tensor as T
import theano
from theano.ifelse import ifelse
import numpy as np
import params   # TODO: remove params
import pickle


#############
# 1 view pair
#############

def __1viewPair_SurfaceNet__(input_var_5D, input_var_shape = (None,3*2)+(64,)*3,\
        N_predicts_perGroup = 6):
    """
    from the 5D input (N_cubePair, 2rgb, h, w, d) of the colored cubePairs 
    to predicts occupancy probability map (N_cubePair, 1, h, w, d)
    """
    input_var = input_var_5D
    net={}
    net["input"] = lasagne.layers.InputLayer(input_var_shape, input_var)
    input_chunk_len = input_var.shape[0] / N_predicts_perGroup

    conv_nonlinearity = lasagne.nonlinearities.rectify
    nonlinearity_sigmoid = lasagne.nonlinearities.sigmoid

    #---------------------------    
    net["conv1_1"] = batch_norm(Conv3DDNNLayer(net["input"],32,(3,3,3),nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))
    net["conv1_2"] = batch_norm(Conv3DDNNLayer(net["conv1_1"],32,(3,3,3),nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))
    net["conv1_3"] = batch_norm(Conv3DDNNLayer(net["conv1_2"],32,(3,3,3),nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))

    net["pool1"] = Pool3DDNNLayer(net["conv1_3"], (2,2,2), stride=2)
    net["side_op1"] = batch_norm(Conv3DDNNLayer(net["conv1_3"],16,(1,1,1),nonlinearity=nonlinearity_sigmoid,untie_biases=False,pad='same'))
    net["side_op1_deconv"] = net["side_op1"]

    #---------------------------
    net["conv2_1"] = batch_norm(Conv3DDNNLayer(net["pool1"],80,(3,3,3),nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))
    net["conv2_2"] = batch_norm(Conv3DDNNLayer(net["conv2_1"],80,(3,3,3),nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))
    net["conv2_3"] = batch_norm(Conv3DDNNLayer(net["conv2_2"],80,(3,3,3),nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))

    net["pool2"] = Pool3DDNNLayer(net["conv2_3"], (2,2,2), stride=2)  
    net["side_op2"] = batch_norm(Conv3DDNNLayer(net["conv2_3"],16,(1,1,1),nonlinearity=nonlinearity_sigmoid,untie_biases=False,pad='same'))
    net["side_op2_deconv"] = Bilinear_3DInterpolation(net["side_op2"], upscale_factor=2, untie_biases=False, nonlinearity=None, pad='same')
                                                    
    #---------------------------
    net["conv3_1"] = batch_norm(Conv3DDNNLayer(net["pool2"],160,(3,3,3),nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))
    net["conv3_2"] = batch_norm(Conv3DDNNLayer(net["conv3_1"],160,(3,3,3),nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))
    net["conv3_3"] = batch_norm(Conv3DDNNLayer(net["conv3_2"],160,(3,3,3),nonlinearity=conv_nonlinearity,untie_biases=False,pad='same') )

    ##pool3 = Pool3DDNNLayer(conv3_3, (2,2,2), stride=2)  
    net["side_op3"] = batch_norm(Conv3DDNNLayer(net["conv3_3"],16,(1,1,1),nonlinearity=nonlinearity_sigmoid,untie_biases=False,pad='same'))
    net["side_op3_deconv"] = Bilinear_3DInterpolation(net["side_op3"], upscale_factor=4, untie_biases=False, nonlinearity=None, pad='same')
    
    #---------------------------
    net["conv3_3_pad"] = PadLayer(net["conv3_3"], width=2, val=0, batch_ndim=2)
    net["conv4_1"] = batch_norm(DilatedConv3DLayer(net["conv3_3_pad"],300,(3,3,3),dilation=(2,2,2),nonlinearity=conv_nonlinearity,untie_biases=False))
    net["conv4_1_pad"] = PadLayer(net["conv4_1"], width=2, val=0, batch_ndim=2)
    net["conv4_2"] = batch_norm(DilatedConv3DLayer(net["conv4_1_pad"],300,(3,3,3),dilation=(2,2,2),nonlinearity=conv_nonlinearity,untie_biases=False))
    net["conv4_2_pad"] = PadLayer(net["conv4_2"], width=2, val=0, batch_ndim=2)
    net["conv4_3"] = batch_norm(DilatedConv3DLayer(net["conv4_2_pad"],300,(3,3,3),dilation=(2,2,2),nonlinearity=conv_nonlinearity,untie_biases=False) )
    net["conv4_3_pad"] = PadLayer(net["conv4_3"], width=0, val=0, batch_ndim=2)
    net["side_op4"] = batch_norm(DilatedConv3DLayer(net["conv4_3_pad"],16,(1,1,1),dilation=(2,2,2),nonlinearity=nonlinearity_sigmoid,untie_biases=False))
    net["side_op4_deconv"] = Bilinear_3DInterpolation(net["side_op4"], upscale_factor=4, untie_biases=False, nonlinearity=None, pad='same')
                                
    #---------------------------
    net["fuse_side_outputs"] = ConcatLayer([net["side_op1_deconv"],net["side_op2_deconv"],net["side_op3_deconv"],net["side_op4_deconv"]], axis=1)
    net["merge_conv"] = batch_norm(Conv3DDNNLayer(net["fuse_side_outputs"],100,(3,3,3),nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))
    net["merge_conv"] = batch_norm(Conv3DDNNLayer(net["merge_conv"],100,(3,3,3),nonlinearity=conv_nonlinearity,untie_biases=False,pad='same'))
    net["merge_conv3"] = batch_norm(Conv3DDNNLayer(net["merge_conv"],1,(1,1,1),nonlinearity=nonlinearity_sigmoid,untie_biases=False,pad='same')) # linear output for regression
    net["output_SurfaceNet"] = net["merge_conv3"]
    return net



##############
# n view pairs
##############

def __relativeWeight_net__(feature_input_var, D_viewPairFeature, num_hidden_units, N_predicts_perGroup):
    """
    Because the softmax weights reflect relative importance. Each value will not be meaningful without comparing with other members in each group. 

    from the feature input (N_cubePair, D_viewPairFeature) for each cube pair 
    to predict the importance/weight output (N_cubePair, 1) for surface fusion
    """

    net = {}
    net["feature_input"] = lasagne.layers.InputLayer((None,D_viewPairFeature), feature_input_var)
    net["feature_fc1"] = batch_norm(DenseLayer(net["feature_input"], num_units=num_hidden_units, nonlinearity=lasagne.nonlinearities.sigmoid))
    net["feature_linear1"] = DenseLayer(net["feature_fc1"], num_units=1, nonlinearity=None)
    net["feature_reshape"] = ReshapeLayer(net["feature_linear1"], shape=(-1, N_predicts_perGroup))
    net["feature_softmax"] = NonlinearityLayer(net["feature_reshape"], nonlinearity=lasagne.nonlinearities.softmax)

    net["output_softmaxWeights"] = net["feature_softmax"]
    return net


def __weightedAverage_net__(input_var, feature_input_var, input_cube_size, N_viewPairs4inference, \
             D_viewPairFeature, num_hidden_units, with_weight=True):
    """
    Because no matter train / val / test, the __weightedAverage_net__ (SurfaceNet + __relativeWeight_net__) will be build and the trained model will be loaded.
    Latter, when SurfaceNet_fn_xxx, only need to feed in the defined __weightedAverage_net__.

    Check
    ===========
    >> import theano.tensor as T
    >> import nets as n
    >> import lasagne
    >> tensor5D = T.TensorType('float32', (False,)*5)
    >> input_var = tensor5D('X')
    >> similFeature_var = T.matrix('similFeature')
    >> net = n.__weightedAverage_net__(input_var, similFeature_var,32,3,128,100,True)
    >> param_volum = len(lasagne.layers.get_all_params(net['output_SurfaceNet']))
    >> param_simil = len(lasagne.layers.get_all_params(net['feature_softmax']))
    >> param_fuse = len(lasagne.layers.get_all_params(net['output_fusionNet']))
    >> param_fuse == param_volum + param_simil
    """

    net = __1viewPair_SurfaceNet__(input_var_5D = input_var, input_var_shape = (None,3*2)+(input_cube_size,)*3, \
            N_predicts_perGroup = N_viewPairs4inference)
    net["output_SurfaceNet_reshape"] = ReshapeLayer(net["output_SurfaceNet"], shape=(-1, N_viewPairs4inference)+(input_cube_size,)*3)
    if with_weight:
        softmaxWeights_net = __relativeWeight_net__(feature_input_var, D_viewPairFeature,\
                num_hidden_units, N_viewPairs4inference)
        net.update(softmaxWeights_net)
        #output_softmaxWeights_var= lasagne.layers.get_output(net["output_softmaxWeights"])
        ###output_SurfaceNet_channelPool = ChannelPool_argmaxWeight(output_SurfaceNet_reshape, average_weight_tensor)
        net["output_SurfaceNet_channelPool"] = ChannelPool_weightedAverage([net["output_SurfaceNet_reshape"], net["output_softmaxWeights"]])
    
    else:
        net["output_SurfaceNet_channelPool"] = ChannelPool_max(net["output_SurfaceNet_reshape"])

 

    net["output_fusionNet"] = net["output_SurfaceNet_channelPool"] ##output_SurfaceNet_reshape_channelPool / conv1_3
    # print "output shape:", net["output_fusionNet"].output_shape
    return net


#----------------------------------------------------------------------        
def __updates__(net, cost, layer_range_tuple_2_update, default_lr, update_algorithm='nesterov_momentum'):
    """
    learning rate for finetuning

    Parameters
    ----------
    net: dict of layers
    cost: cost function
    layer_range_tuple_2_update: ('layerName1','layerName2')
            only the params within the range ('layerName1','layerName2'] will be updated
            DON'T update the 'layerName1's params
    default_lr: default lr
    update_algorithm: 'sgd' / 'nesterov_momentum'
    
    Returns
    -------
    updates: for train_fn

    Notes
    -------
    If multiple range of layers will be updated.
    Just updates_old.update(updates_new), because it is OrderedDict.
    """
    if len(layer_range_tuple_2_update) != 2:
        raise ValueError("2 element tuple is desired for layer_range_tuple_2_update, rather than {}".format(len(layer_range_tuple_2_update)))

    # params_Layer0 = [w,b], where w/b are theano tensor variable (have its own ID)
    # params_Layer1 = [w,b,w1,b1]
    # params_trainable = [w1,b1]
    params_untrainable = lasagne.layers.get_all_params(net[layer_range_tuple_2_update[0]], trainable=True)
    params_trainable = [p for p in lasagne.layers.get_all_params(net[layer_range_tuple_2_update[1]], trainable=True)\
            if not p in params_untrainable]

    print("\nonly update the weights in the range ({},{}]".format(layer_range_tuple_2_update[0], layer_range_tuple_2_update[1]))
    print("the weights to be updated: {}".format(params_trainable))
            
    if update_algorithm in 'nesterov_momentum':
        layer_updates = lasagne.updates.nesterov_momentum(cost, params_trainable, learning_rate=default_lr, momentum=0.9)
    elif update_algorithm in 'sgd; stochastic gradient descent':
        layer_updates = lasagne.updates.sgd(cost, params_trainable, learning_rate=default_lr)    
    else:
        raise ValueError("the update_algorithm {} is not found".format(update_algorithm))

    return layer_updates
    




def __weighted_mult_binary_crossentropy__(prediction, target, w_for_1):
    return -(w_for_1 * target * T.log(prediction) + (1.0-w_for_1)*(1.0 - target) * T.log(1.0 - prediction))

def __weighted_MSE__(prediction, target, w_for_1):
    w_for_0 = 1.0 - w_for_1
    w_for_p = lambda p: w_for_0 + p * (w_for_1 - w_for_0) 
    return T.sqrt(T.sqr(prediction - target)) * w_for_p(target)

def __weighted_accuracy__(prediction, target):
    """
    calculate the positive/negative acc
    acc = (acc_pos + acc_neg) / 2. # equally weighted
    >>> gt = T.matrix('gt')
    >>> pred = T.matrix('pred')
    >>> acc = __weighted_accuracy__(pred, gt)
    >>> f = theano.function([pred, gt], acc)
    >>> pred_np = np.array([[0.1,0],[0.9,1]]).astype(np.float32)
    >>> gt_np = np.array([[0,0],[0,0]]).astype(np.float32)
    >>> f(pred_np, gt_np).sum()
    0.5
    """
    pos = (target > 0).nonzero()
    neg = T.eq(target, 0).nonzero()

    accuracy_neg = lasagne.objectives.binary_accuracy(prediction[neg], target[neg])
    # when the cube is empty, (target > 0).sum() = 0
    # accuracy_pos = accuracy_neg if (target > 0).sum() == 0 else lasagne.objectives.binary_accuracy(prediction[pos], target[pos]) 
    accuracy_pos = ifelse(T.eq((target > 0).sum(), 0), accuracy_neg, lasagne.objectives.binary_accuracy(prediction[pos], target[pos]) )

    return (T.mean(accuracy_pos) + T.mean(accuracy_neg))/2.0


def SurfaceNet_fn_trainVal(N_viewPairs4inference, default_lr, input_cube_size, D_viewPairFeature, \
            num_hidden_units, CHANNEL_MEAN, return_train_fn=True, return_val_fn=True, with_weight=True):

    """
    This function only defines the train_fn and the val_fn while training process.
    There are 2 training process:
    1. only train the SurfaceNet without weight
    2. train the softmaxWeight with(out) finetuning the SurfaceNet

    For the val_fn when only have validation, refer to the [TODO].

    ===================
    >> SurfaceNet_fn_trainVal(with_weight = True)
    >> SurfaceNet_fn_trainVal(with_weight = False)
    """
    train_fn = None
    val_fn = None


    tensor5D = T.TensorType('float32', (False,)*5)
    input_var = tensor5D('X')
    output_var = tensor5D('Y')
    similFeature_var = T.matrix('similFeature')

    net = __weightedAverage_net__(input_var, similFeature_var, input_cube_size, N_viewPairs4inference,\
            D_viewPairFeature, num_hidden_units, with_weight)
    if return_val_fn:
        pred_fuse_val = lasagne.layers.get_output(net["output_fusionNet"], deterministic=True)
        # accuracy_val = lasagne.objectives.binary_accuracy(pred_fuse_val, output_var) # in case soft_gt
        accuracy_val = __weighted_accuracy__(pred_fuse_val, output_var)

        # fuseNet_val_fn = theano.function([input_var, output_var], [accuracy_val,pred_fuse_val])

        val_fn_input_var_list = [input_var, similFeature_var, output_var] if with_weight\
                else [input_var, output_var]
        val_fn_output_var_list = [accuracy_val,pred_fuse_val] if with_weight\
                else [accuracy_val,pred_fuse_val]
        val_fn = theano.function(val_fn_input_var_list, val_fn_output_var_list)
    
    if return_train_fn:
        pred_fuse = lasagne.layers.get_output(net["output_fusionNet"])
        output_softmaxWeights_var= lasagne.layers.get_output(net["output_softmaxWeights"]) if with_weight \
                else None

        #loss = __weighted_MSE__(pred_fuse, output_var, w_for_1 = 0.98) \
        loss = __weighted_mult_binary_crossentropy__(pred_fuse, output_var, w_for_1 = 0.96) \
            + regularize_layer_params(net["output_fusionNet"],l2) * 1e-4 \

        aggregated_loss = lasagne.objectives.aggregate(loss)

        if not params.__layer_range_tuple_2_update is None: 
            updates = __updates__(net=net, cost=aggregated_loss, layer_range_tuple_2_update=params.__layer_range_tuple_2_update, \
                    default_lr=default_lr, update_algorithm='nesterov_momentum') 
        else:
            params = lasagne.layers.get_all_params(net["output_fusionNet"], trainable=True)
            updates = lasagne.updates.nesterov_momentum(aggregated_loss, params, learning_rate=params.__lr)   


        # accuracy = lasagne.objectives.binary_accuracy(pred_fuse, output_var) # in case soft_gt
        accuracy = __weighted_accuracy__(pred_fuse, output_var)

        train_fn_input_var_list = [input_var, similFeature_var, output_var] if with_weight \
                else [input_var, output_var]
        train_fn_output_var_list = [loss,accuracy, pred_fuse, output_softmaxWeights_var] if with_weight \
                else [loss,accuracy, pred_fuse]

        train_fn = theano.function(train_fn_input_var_list, train_fn_output_var_list, updates=updates)
    return net, train_fn, val_fn


def __SurfaceNet_fn_inference__(N_viewPairs4inference, input_cube_size, D_viewPairFeature, num_hidden_units, \
            with_weight=True, with_groundTruth = True, return_unfused_predict = False):
    """
    this function difines 2 net_fns, which could be used in the test phase:
    1. viewPair_relativeImpt_fn: calculate softmax weight given feature input
    2. nViewPair_SurfaceNet_fn: ouput a prediction based on the colored cube pairs with(out) weighted average. (based on whether the softmax weight is available)

    when with_weight=True/False, N_viewPairs4inference=1:
        viewPair_relativeImpt_fn != None, because it will be used to find the argmax softmax weight. 

        In this case, the prediction would be the output of the SurfaceNet(don't need to reshape anymore). 
        So that the tensor n_samples_perGroup_var, 
            which is only used to reshape from (N_group*N_sample_perGroup,1) to (N_group, N_sample_perGroup), 
            will not be treated as input.
    
    return_unfused_predict: True: also return the unfused predictions of all the view pairs.
            This return unfused predictions could be used for color fusion. 
    ==============
    >> python -c "import nets; nets.__SurfaceNet_fn_inference__(with_weight=True, N_viewPairs4inference=1)"
    >> __SurfaceNet_fn_inference__(with_weight=False, N_viewPairs4inference=1)
    >> __SurfaceNet_fn_inference__(with_weight=True, N_viewPairs4inference=2)
    >> __SurfaceNet_fn_inference__(with_weight=False, N_viewPairs4inference=2)
    """
    viewPair_relativeImpt_fn = None
    tensor5D = T.TensorType('float32', (False,)*5)
    input_var = tensor5D('X')
    output_var = tensor5D('Y')
    similFeature_var = T.matrix('similFeature')

    # This tensor is only used to reshape from (N_group*N_sample_perGroup,1) to (N_group, N_sample_perGroup)
    # so will not be used when N_viewPairs4inference == 1
    n_samples_perGroup_var = T.iscalar('n_samples_perGroup') # when setted as arg of theano.function, use the 'n_samples_perGroup' to pass value 

    net = __weightedAverage_net__(input_var, similFeature_var, input_cube_size, n_samples_perGroup_var,
                 D_viewPairFeature, num_hidden_units, with_weight)

    ##### the viewPair_relativeImpt_fn
    if with_weight == True:
        output_softmaxWeights_var= lasagne.layers.get_output(net["output_softmaxWeights"], deterministic=True)

        viewPair_relativeImpt_fn = theano.function([similFeature_var, theano.In(n_samples_perGroup_var, value=N_viewPairs4inference)], \
                output_softmaxWeights_var)
             
    ##### the nViewPair_SurfaceNet_fn
    if N_viewPairs4inference >= 2:
        if with_weight == True:
            similWeight_var = T.matrix('similWeight')
            
            similWeight_input_layer = lasagne.layers.InputLayer((None,N_viewPairs4inference), similWeight_var)
            net["output_SurfaceNet_channelPool_givenWeight"] = ChannelPool_weightedAverage([net["output_SurfaceNet_reshape"], similWeight_input_layer])
            net["output_fusionNet"] = net["output_SurfaceNet_channelPool_givenWeight"]
        else:
            net["output_SurfaceNet_channelPool"] = ChannelPool_max(net["output_SurfaceNet_reshape"])
            net["output_fusionNet"] = net["output_SurfaceNet_channelPool"]

        output_fusionNet_var, unfused_predictions_var = lasagne.layers.get_output([net["output_fusionNet"], net["output_SurfaceNet_reshape"]], \
                deterministic=True)
    elif N_viewPairs4inference == 1: # if only use 1 colored Cube pair, we don't need weight any more.
        with_weight = False # IMPORTANT, in this case, the vars related to weights will be ignored
        output_fusionNet_var = lasagne.layers.get_output(net["output_SurfaceNet"], deterministic=True) 
        unfused_predictions_var = output_fusionNet_var

    if with_groundTruth:
        # accuracy_val_givenWeight = lasagne.objectives.binary_accuracy(output_fusionNet_var, output_var) # in case of soft_gt
        accuracy_val_givenWeight = __weighted_accuracy__(output_fusionNet_var, output_var)


    # *********************
    fuseNet_fn_input_var_list = [input_var, similWeight_var] if with_weight \
            else [input_var] 
    # in the reconstruction procedure, we don't have ground truth
    fuseNet_fn_input_var_list += [output_var] if with_groundTruth else []
    # Tensor n_samples_perGroup_var is only used to reshape from (N_group*N_sample_perGroup,1) to (N_group, N_sample_perGroup)
    # so only used when N_viewPairs4inference >= 2
    fuseNet_fn_input_var_list += [theano.In(n_samples_perGroup_var, value=N_viewPairs4inference)] if N_viewPairs4inference >= 2 \
            else [] 
    # *********************    
    if return_unfused_predict:
        fuseNet_fn_output_var_list = [accuracy_val_givenWeight, output_fusionNet_var, unfused_predictions_var] if with_groundTruth \
                else [output_fusionNet_var, unfused_predictions_var]
    else:
        fuseNet_fn_output_var_list = [accuracy_val_givenWeight, output_fusionNet_var] if with_groundTruth \
                else output_fusionNet_var
    
    # *********************
    nViewPair_SurfaceNet_fn = theano.function(fuseNet_fn_input_var_list, fuseNet_fn_output_var_list)
    return net, viewPair_relativeImpt_fn, nViewPair_SurfaceNet_fn

def SurfaceNet_inference(N_viewPairs4inference, model_file, layerNameList_2_load):
    """
    return the SurfaceNet functions used for inference, and load the model weights.
    """

    # define DL functions: SurfaceNet
    net, viewPair_relativeImpt_fn, nViewPair_SurfaceNet_fn = __SurfaceNet_fn_inference__(with_weight=True, with_groundTruth=False, \
            input_cube_size = params.__cube_D, N_viewPairs4inference = N_viewPairs4inference, \
            D_viewPairFeature = params.__D_viewPairFeature, num_hidden_units = params.__similNet_hidden_dim,\
            return_unfused_predict = True)

    # load the pretrained model
    layerList_2_load = [net[_layerName] for _layerName in layerNameList_2_load]
    with open(model_file, 'r') as f:
        file_data = pickle.load(f)
    lasagne.layers.set_all_param_values(layerList_2_load, file_data)
    print ('loaded SurfaceNet model: {}'.format(model_file))
    return viewPair_relativeImpt_fn, nViewPair_SurfaceNet_fn





if __name__ == '__main__':
    import doctest
    doctest.testmod()

