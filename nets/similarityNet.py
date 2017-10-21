"""
define network structures    
"""
import lasagne
#if lasagne.utils.theano.config.device == 'cpu':
if lasagne.utils.theano.sandbox.cuda.dnn_available(): # when cuDNN available
    from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer 
else:
    from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import InputLayer, DenseLayer, SliceLayer, ReshapeLayer, ConcatLayer, batch_norm,FlattenLayer
from lasagne.nonlinearities import tanh, sigmoid,rectify
import theano
import theano.tensor as T
from layers import CropFeatureMapCenterLayer, L2NormLayer, DistanceLayer
import params
import pickle

###################
# network structure
###################

def __input_var_TO_embedding_layer__(input_var, imgPatch_hw_size):
    net = {}
    # subtracted MEAN in the preprocess stage.
    net['input'] = InputLayer((None, 3, imgPatch_hw_size[0], imgPatch_hw_size[1]), \
	      input_var=input_var)
    net['conv1_1'] = ConvLayer(net['input'], 64, 3, pad=1)
    net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3, pad=1)
    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    net['conv2_1'] = ConvLayer(net['pool1'], 128, 3, pad=1)
    net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3, pad=1)
    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    net['conv3_1'] = ConvLayer(net['pool2'], 256, 3, pad=1)
    net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3, pad=1)
    net['conv3_3'] = ConvLayer(net['conv3_2'], 256, 3, pad=1)
    net['pool3'] = PoolLayer(net['conv3_3'], 2)
    net['conv4_1'] = ConvLayer(net['pool3'], 512, 3, pad=1)
    net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, pad=1)
    net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3, pad=1)
    net['pool4'] = PoolLayer(net['conv4_3'], 2)
    net['conv5_1'] = ConvLayer(net['pool4'], 512, 3, pad=1)
    net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3, pad=1)
    net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3, pad=1)
    net['pool5'] = PoolLayer(net['conv5_3'], 2) # keep the output layer in lower dim
    # focus more on the center of the embedding maps 
    net['concat'] = ConcatLayer([FlattenLayer(net['pool5'], 2), 
                                CropFeatureMapCenterLayer(net['pool1'], cropCenter_r=1),
                                CropFeatureMapCenterLayer(net['pool2'], cropCenter_r=1),
                                CropFeatureMapCenterLayer(net['pool3'], cropCenter_r=1),
                                CropFeatureMapCenterLayer(net['pool4'], cropCenter_r=1)
                                ], axis=1)
    net['flat1'] = FlattenLayer(net['concat'], 2)
    net['L2_norm'] = L2NormLayer(net['flat1'])    
    net['embedding'] = DenseLayer(net['L2_norm'], num_units=params.__D_imgPatchEmbedding, nonlinearity=None)
    return net

def __embedding_layer_TO_similarity_layer__(embedding_layer, tripletInput=True):
    net = {}
    if tripletInput:
        net['reshape'] = ReshapeLayer(embedding_layer, (-1,3,[1]))
        net['triplet_anchor'] = SliceLayer(net['reshape'], indices=0, axis=1) # in order to keep the dim, use slice(0,1) == array[0:1,...]
        net['triplet_pos'] = SliceLayer(net['reshape'], indices=1, axis=1)
        net['triplet_neg'] = SliceLayer(net['reshape'], indices=2, axis=1)
        net['euclid_pos'] = DistanceLayer([net['triplet_anchor'], net['triplet_pos']], Lp=2, axis=1, keepdims=True)
        net['euclid_neg'] = DistanceLayer([net['triplet_anchor'], net['triplet_neg']], Lp=2, axis=1, keepdims=True)
        net['euclid_dist'] = ConcatLayer([net['euclid_pos'], net['euclid_neg']],axis=0)
    else:
        net['reshape'] = ReshapeLayer(embedding_layer, (-1,2,[1]))
        net['pair_1'] = SliceLayer(net['reshape'], indices=0, axis=1)
        net['pair_2'] = SliceLayer(net['reshape'], indices=1, axis=1)
        net['euclid_dist'] = DistanceLayer([net['pair_1'], net['pair_2']], Lp=2, axis=1, keepdims=True)
    # input-->output (shape 1-->1), logistic regression
    net['similarity'] = DenseLayer(net['euclid_dist'], num_units=1, nonlinearity=sigmoid)
    return net
 
def __similarityNet__(input_var, imgPatch_hw_size, tripletInput=True):
    """
    how to check the correctness. (how to check the equality of two lasagne nets)
    Can use lasagne.layers.count_params/.get_all_layers/params
    """
    net = {}
    net_embedding = __input_var_TO_embedding_layer__(input_var, imgPatch_hw_size)
    net.update(net_embedding) 

    net_similarity = __embedding_layer_TO_similarity_layer__(net['embedding'], tripletInput = tripletInput)
    net.update(net_similarity)   # dict.update the same 'key' will be 'updated'/replaced
    return net


##################
# theano functions
##################

def __cost_triplet__(diff_pos_Euclid, diff_neg_Euclid, alpha):
    # dist = triplet_alpha - (diff_neg_sq_sum - diff_pos_sq_sum) #(diff_neg_sq_sum - diff_pos_sq_sum) / (diff_neg_Euclid - diff_pos_Euclid)
    dist = 1 - (diff_neg_Euclid/(diff_pos_Euclid + alpha))
    dist_thresh = dist*(dist>0)
    return dist_thresh.sum()

def __similarity_acc_cost__(predict_var, similarity_cost_ON = False):
    N_pos_sample = predict_var.shape[0] / 2
    target_var = T.concatenate([T.zeros((N_pos_sample,1)),T.ones((N_pos_sample,1))], axis=0)

    acc = lasagne.objectives.binary_accuracy(predict_var, target_var)
    cost = lasagne.objectives.binary_crossentropy(predict_var, target_var).sum() if similarity_cost_ON else None
    return acc, cost   

def __updates__(net, cost, layer_range_tuple_2_update, default_lr, update_algorithm='nesterov_momentum'):
    """
    learning rate for finetuning

    Parameters
    ----------
    net: dict of layers
    cost: cost function
    layer_range_tuple_2_update: ('layerName1','layerName2'), or list of tuple [(l1,l2),(l3,l4)]
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

    params_trainable_all = []
    if isinstance(layer_range_tuple_2_update[0], tuple):
        layer_range_tuple_2_update_iter = layer_range_tuple_2_update
    else:
        layer_range_tuple_2_update_iter = [layer_range_tuple_2_update]
    for layer_range_tuple_2_update in layer_range_tuple_2_update_iter:
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

        params_trainable_all += params_trainable
            
    if update_algorithm in 'nesterov_momentum':
        layer_updates = lasagne.updates.nesterov_momentum(cost, params_trainable_all, learning_rate=default_lr, momentum=0.9)
    elif update_algorithm in 'sgd; stochastic gradient descent':
        layer_updates = lasagne.updates.sgd(cost, params_trainable_all, learning_rate=default_lr)    
    else:
        raise ValueError("the update_algorithm {} is not found".format(update_algorithm))

    return layer_updates



def similarityNet_fn_train_val(imgPatch_hw_size, return_train_fn=True, return_val_fn=True):
    train_fn = None
    val_fn = None
    input_var = T.tensor4('inputs')
    net_train_val = __similarityNet__(input_var, imgPatch_hw_size, tripletInput=True)

    if return_val_fn:
        predict_var_val = lasagne.layers.get_output(net_train_val['similarity'], deterministic=True)
        similarity_acc_val, _ = __similarity_acc_cost__(predict_var_val, similarity_cost_ON=False)
        val_fn = theano.function([input_var], [similarity_acc_val]) #similarity_acc_val

    if return_train_fn:
        diff_pos_Euclid, diff_neg_Euclid, predict_var = lasagne.layers.get_output(\
                                                [net_train_val['euclid_pos'],\
                                                net_train_val['euclid_neg'],\
                                                net_train_val['similarity']], deterministic=False)

        similarity_acc, similarity_cost = __similarity_acc_cost__(predict_var, similarity_cost_ON=True)
        tripletCost = __cost_triplet__(diff_pos_Euclid, diff_neg_Euclid, alpha = params.__triplet_alpha)

        ############## cost with regularization ##############
        weight_l2_penalty = lasagne.regularization.regularize_network_params(net_train_val['similarity'], lasagne.regularization.l2) * params.__weight_decay
        cost = tripletCost + weight_l2_penalty + similarity_cost

        updates = __updates__(net=net_train_val, cost = cost, \
                layer_range_tuple_2_update=[('pool5','similarity')], default_lr=params.__DEFAULT_LR)
        train_fn = theano.function([input_var], [cost,similarity_acc,diff_pos_Euclid, diff_neg_Euclid, predict_var], updates=updates)    

    return net_train_val, train_fn, val_fn



def similarityNet_fn_patchPair_2_embedding(imgPatch_hw_size):
    """
    Used for training
    to get the embedding+similarity output with input of patch pairs
    The returned layer is used to load the trained similarityNet model.
    """
    input_var = T.tensor4('inputs')
    net_fuse = __similarityNet__(input_var, imgPatch_hw_size, tripletInput=False)

    embedding_var, similarity = lasagne.layers.get_output([net_fuse['embedding'], \
            net_fuse['similarity']], deterministic=True)

    fn_fuse = theano.function([input_var], [embedding_var, similarity])  
    return net_fuse['similarity'], fn_fuse

def similarityNet_fn_patch_2_embedding_2_similarity(imgPatch_hw_size):
    """
    Used for viewPair selection and weighted average
    the case where the 2 functions: get_patch_embedding and the calc_similarity_from_embeddings_of_patchPair are seperately used.
    Q: How to reload the trained model to these two seperated nets?
    A: Could use the set_all_param_values(layer_list,...)
    """
    patch_var = T.tensor4('patch')
    net_embedding = __input_var_TO_embedding_layer__(patch_var, imgPatch_hw_size)
    patch_embedding_var = lasagne.layers.get_output(net_embedding['embedding'], deterministic=True)
    patch2embedding_fn = theano.function([patch_var], patch_embedding_var)  

    embeddingPair_var = T.matrix('embeddingPair') 
    net_embeddingPair2simil = __embedding_layer_TO_similarity_layer__(InputLayer((None,params.__D_imgPatchEmbedding), input_var=embeddingPair_var), tripletInput=False)
    embeddingPair_similarity_var = lasagne.layers.get_output(net_embeddingPair2simil['similarity'], deterministic=True)
    embeddingPair2simil_fn = theano.function([embeddingPair_var], embeddingPair_similarity_var)  

    return net_embedding['embedding'], net_embeddingPair2simil['similarity'], patch2embedding_fn, embeddingPair2simil_fn


def similarityNet_inference(model_file, imgPatch_hw_size):
    """
    return the similarityNet functions used for inference, and load the model weights.
    """

    # define DL functions: similarityNet
    net_embeddingLayer, net_embeddingPair2similarityLayer, patch2embedding_fn, embeddingPair2simil_fn = \
            similarityNet_fn_patch_2_embedding_2_similarity(imgPatch_hw_size)
    # load weights  TODO
    with open(model_file) as f:
        modelWeights = pickle.load(f)
    lasagne.layers.set_all_param_values([net_embeddingLayer, net_embeddingPair2similarityLayer], modelWeights) #[similNet_outputLayer]
    print('loaded similarityNet model: {}'.format(model_file))
    return patch2embedding_fn, embeddingPair2simil_fn 


