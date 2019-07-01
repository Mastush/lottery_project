import numpy as np
from abc import ABC, abstractmethod
from keras.initializers import glorot_normal


class ModelReinitializer(ABC):
    def __init__(self, model, weight_ranker, reinitializer, layer_by_layer=True, mask=None, percentage=None):
        self._model = model
        self._ranker = weight_ranker
        self._percentage = percentage  # TODO: support dynamic percentage?
        self._reinitializer = reinitializer
        self._layer_by_layer = layer_by_layer
        self._mask = mask  # TODO: for setting renitialized weights in stone?

    @abstractmethod
    def reinitialize(self, percentage=None):
        pass



class DiffModelReinitializer(ModelReinitializer):
    def __init__(self, model, weight_ranker, reinitializer, layer_by_layer=True, mask=None, percentage=None):
        super().__init__(model, weight_ranker, reinitializer, layer_by_layer, mask, percentage)
        self._initial_weights = self._get_initial_weights()

    def _get_initial_weights(self):
        initial_weights = []
        for layer in self._model.layers:
            if len(layer.get_weights()) > 0:
                initial_weights.append(layer.get_weights())
            else:
                initial_weights.append(None)
        return initial_weights

    def reinitialize(self, percentage=None):
        if percentage is None:
            percentage = self._percentage
        if self._layer_by_layer:
            model_layers = self._model.get_layers()
            for i in range(len(model_layers)):
                layer = model_layers[i]
                if len(layer.get_weights()) > 0:
                    kernel, bias = layer.get_weights()
                    diff_kernels, diff_biases = np.abs(kernel - self._initial_weights[i][0]), \
                                                np.abs(bias - self._initial_weights[i][1])
                    new_weights = self._reinitializer.apply(kernel, bias, diff_kernels, diff_biases, percentage)
                    layer.set_weights(new_weights)
        else:
            pass  # TODO: implement


class Reinitializer(ABC):
    def __init__(self, ranker_func, kernel_initializer, bias_initializer):
        """
        :param ranker_func: Ranks a vector via some
        :param kernel_initializer:
        :param bias_initializer:
        """
        self._ranker = ranker_func
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer

    @abstractmethod
    def apply(self):
        pass


class DiffReinitializer(Reinitializer):
    def __init__(self, ranker_func, kernel_initializer, bias_initializer):
        super().__init__(ranker_func, kernel_initializer, bias_initializer)

    def apply(self, kernel, bias, diff_kernels, diff_biases, percentage):
        diff_weights = np.concatenate((diff_kernels.flatten(), diff_biases.flatten()))

        where_to_reinitialize_mask = self._ranker(diff_weights)  # TODO: implement
        kernel_mask = where_to_reinitialize_mask[:kernel.size].reshape(kernel.shape)
        bias_mask = where_to_reinitialize_mask[kernel.size:].reshape(bias.shape)

        initialized_kernel = self._kernel_initializer(kernel.shape)
        initialized_bias = self._kernel_initializer(bias.shape)

        new_kernel = kernel + (initialized_kernel * kernel_mask)
        new_bias = bias + (initialized_bias * bias_mask)

        return new_kernel, new_bias


