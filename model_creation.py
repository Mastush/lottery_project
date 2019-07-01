from keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, MaxPooling2D, Activation, Dropout, BatchNormalization
from keras.models import Model, Sequential
from keras.losses import categorical_crossentropy
import constants


def get_convolutional_model(input_shape, num_of_classes, num_of_conv_layers, num_of_channels, kernel_size, stride,
                            num_of_fc_layers, fc_size, regularizer, regularization_coeffs, conv_activation,
                            fc_activation, last_activation, optimizer, loss_function, metrics, use_max_pooling=False,
                            padding='same', dropout=0 ,batch_norm=False):
    """
    Creates an compiles a convolutional neural network with the given specifications.
    :param input_shape: (tuple) input shape
    :param num_of_classes: (int) number of classes for the multi-class classification task
    :param num_of_conv_layers: (int) amount of convolution layers wanted
    :param num_of_channels: (int or list of ints) represents the number of convolution channels in each
                             convolutional layer
    :param kernel_size: (int or list of ints) represents the kernel size in each convolutional layer
    :param num_of_fc_layers: (int) the wanted number of fully-connected layers. Should be >= 1
    :param fc_size: (int or list of ints) represents the number of neurons in each fully connected layer
    :param regularizer: (regularizer) An instantiatable regularizer class
    :param regularization_coeffs: (number or list of numbers) represents the amount of penalty that will arise from
                                   regularization in each layer
    :param conv_activation: (activation function) the activation function to be activated on the outputs of the
                             convolutional layers
    :param fc_activation: (activation function) the activation function to be activated on the outputs of the
                           fully connected layers
    :param last_activation: (activation function) the activation function to be activated on the outputs of the
                             last (output) layer
    :param optimizer: (keras optimizer) the wanted optimizer for training
    :param loss_function: (keras loss function) the wanted loss function fot training
    :param metrics: (list) a list of metrics for the model
    :param use_max_pooling: (boolean) a boolean variable that indicates whether or not max pooling is to be used
    :param padding: (string or list of strings) the method of padding for the convolutional layers
    :param dropout: TODO: delete or fill
    :param batch_norm: TODO: delete or fill
    :return: A compiled model with the given specifications
    """
    # make vectors out of single numbers if such are given
    if not (isinstance(num_of_channels, tuple) or isinstance(num_of_channels, list)):
        num_of_channels = [num_of_channels for _ in range(num_of_conv_layers)]
    if not (isinstance(kernel_size, tuple) or isinstance(kernel_size, list)):
        kernel_size = [kernel_size for _ in range(num_of_conv_layers)]
    if not (isinstance(stride, tuple) or isinstance(stride, list)):
        stride = [stride for _ in range(num_of_conv_layers)]
    if not (isinstance(padding, tuple) or isinstance(padding, list)):
        padding = [padding for _ in range(num_of_conv_layers)]
    if not (isinstance(fc_size, tuple) or isinstance(fc_size, list)):
        fc_size = [fc_size for _ in range(num_of_fc_layers - 1)]

    total_num_of_layers = num_of_conv_layers + num_of_fc_layers

    if not (isinstance(regularization_coeffs, tuple) or isinstance(regularization_coeffs, list)):
        regularization_coeffs = [regularization_coeffs for _ in range(total_num_of_layers)]
    if not (isinstance(dropout, tuple) or isinstance(dropout, list)):
        dropout = [dropout for _ in range(total_num_of_layers)]

    # check number of coeffs
    if len(regularization_coeffs) > total_num_of_layers:
        raise ValueError("There are too many elements in the regularization coefficients vector!")
    elif len(regularization_coeffs) < total_num_of_layers:
        raise ValueError("There are not enough values in the regularization coefficients vector!")

    # make regularizers
    if regularizer is None:
        regularizers = [None for _ in range(num_of_conv_layers + num_of_fc_layers)]
    else:
        regularizers = [regularizer(regularization_coeffs[i]) for i in range(num_of_conv_layers + num_of_fc_layers)]

    # ----- construct the network ----- #
    input_layer = Input(input_shape)
    last_layer = input_layer

    # conv layers
    for i in range(num_of_conv_layers):
        if batch_norm:
            last_layer = BatchNormalization()(last_layer)
        last_layer = Conv2D(num_of_channels[i], kernel_size[i], strides=stride[i], padding=padding[i],
                            activation=conv_activation, kernel_initializer='glorot_normal',
                            kernel_regularizer=regularizers[i])(last_layer)
        if dropout[i] > 0:
            last_layer = Dropout(dropout[i])(last_layer)
        if use_max_pooling:
            last_layer = MaxPool2D()(last_layer)

    last_layer = Flatten()(last_layer)

    # fc layers
    for j in range(num_of_fc_layers - 1):
        if batch_norm:
            last_layer = BatchNormalization()(last_layer)
        last_layer = Dense(fc_size[j], activation=fc_activation,
                           kernel_regularizer=regularizers[num_of_conv_layers + j])(last_layer)
        if dropout[num_of_conv_layers + j] > 0:
            last_layer = Dropout(dropout[i])(last_layer)

    if batch_norm:
        last_layer = BatchNormalization()(last_layer)
    output_layer = Dense(num_of_classes, activation=last_activation,
                         kernel_regularizer=regularizers[-1])(last_layer)

    # finish up
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)
    return model


