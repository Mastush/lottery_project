import numpy as np


def get_model_weights(model):
    model_weights = []
    for layer in model.layers:
        if len(layer.get_weights()) > 0:
            model_weights.append(layer.get_weights())
        else:
            model_weights.append(None)
    return model_weights

def weights_to_vector(weights_list):
    vector = None
    for entry in weights_list:
        if entry is not None:
            for part in entry:
                vector = part.flatten() if vector is None else np.concatenate((vector, part.flatten()))
    return vector
