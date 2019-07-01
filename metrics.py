from keras.metrics import categorical_accuracy


def get_metrics():
    return [categorical_accuracy]