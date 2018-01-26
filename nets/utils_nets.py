import os
import sys
import pickle
import lasagne
sys.path.append("./utils")
import utils

def save_entire_model(model, save_folder, filename):
    """Pickels the parameters within a Lasagne model."""
    model_data = lasagne.layers.get_all_param_values(model)
    utils.mkdirs_ifNotExist(save_folder)
    filePath = os.path.join(save_folder, filename)
    with open(filePath, 'wb') as f:
        pickle.dump(model_data, f)
        print("Save model to file: {}".format(filePath))
    return filePath

