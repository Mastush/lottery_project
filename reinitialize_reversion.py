import numpy as np
from abc import ABC, abstractmethod
from ranking import *
import keras.backend as K
import utils


class ModelReinitializer(ABC):
    def __init__(self, model, reinitializer, layer_by_layer=True, mask=None, percentage=None):
        self._model = model
        self._percentage = percentage  # TODO: support dynamic percentage?
        self._reinitializer = reinitializer
        self._layer_by_layer = layer_by_layer
        self._mask = mask  # TODO: for setting renitialized weights in stone?

    @abstractmethod
    def reinitialize(self, percentage=None):
        pass


class DiffModelReinitializer(ModelReinitializer):
    def __init__(self, model, reinitializer, mask=None, percentage=None):
        super().__init__(model, reinitializer, mask, percentage)
        self._reference_weights = utils.get_model_weights(self._model)

    def reinitialize(self, percentage=None):
        if percentage is None:
            percentage = self._percentage
        self._reinitializer.apply_to_model(self._model, self._reference_weights, percentage)
        self._reference_weights = utils.get_model_weights(self._model)


class MagnitudeModelReinitializer(ModelReinitializer):  # TODO: implement and try
    def __init__(self, model, reinitializer, mask=None, percentage=None):
        super().__init__(model, reinitializer, mask, percentage)
        self._reference_weights = utils.get_model_weights(self._model)

    def reinitialize(self, percentage=None):
        if percentage is None:
            percentage = self._percentage
        self._reinitializer.apply_to_model(self._model, self._reference_weights, percentage)
        self._reference_weights = utils.get_model_weights(self._model)


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
    def apply_to_model(self):  # TODO: move some abstraction here after POC
        pass


class DiffReinitializer(Reinitializer):
    def __init__(self, ranker, kernel_initializer, bias_initializer):
        super().__init__(ranker, kernel_initializer, bias_initializer)

    def apply_to_model(self, model, reference_weights, percentage):
        reference_weights_vector = utils.weights_to_vector(reference_weights)
        current_weights_vector = utils.weights_to_vector(utils.get_model_weights(model))
        diff_vector = np.abs(reference_weights_vector - current_weights_vector)
        mask_vector = self._ranker(diff_vector, percentage)

        last_idx = 0
        layer_masks = []
        reinitialized_layers = []
        for i in range(len(model.layers)):
            layer = model.layers[i]
            if len(layer.get_weights()) > 0:
                kernel, bias = layer.get_weights()[0], layer.get_weights()[1]
                kernel_mask = mask_vector[last_idx:last_idx + kernel.size].reshape(kernel.shape)
                last_idx += kernel.size
                bias_mask = mask_vector[last_idx:last_idx + bias.size].reshape(bias.shape)
                last_idx += bias.size

                kernel_reinitialized = self._kernel_initializer(kernel.shape)
                bias_reinitialized = self._bias_initializer(bias.shape)

                new_kernel = np.copy(reference_weights[i][0])
                new_bias = np.copy(reference_weights[i][1])

                new_kernel[kernel_mask] = kernel_reinitialized[kernel_mask]
                new_bias[bias_mask] = bias_reinitialized[bias_mask]

                layer.set_weights((new_kernel, new_bias))

                layer_masks.append([kernel_mask, bias_mask])
                reinitialized_layers.append([kernel_reinitialized, bias_reinitialized])
            else:
                layer_masks.append(None)
                reinitialized_layers.append(None)


class MagnitudeReinitializer(Reinitializer):
    def __init__(self, kernel_initializer, bias_initializer, ranker=magnitude_ranking):
        super().__init__(ranker, kernel_initializer, bias_initializer)

    def apply_to_model(self, model, reference_weights, percentage):
        weights_vector = utils.weights_to_vector(utils.get_model_weights(model))
        mask_vector = self._ranker(weights_vector, percentage)

        last_idx = 0
        layer_masks = []
        reinitialized_layers = []
        for i in range(len(model.layers)):
            layer = model.layers[i]
            if len(layer.get_weights()) > 0:
                kernel, bias = layer.get_weights()[0], layer.get_weights()[1]
                kernel_mask = mask_vector[last_idx:last_idx + kernel.size].reshape(kernel.shape)
                last_idx += kernel.size
                bias_mask = mask_vector[last_idx:last_idx + bias.size].reshape(bias.shape)
                last_idx += bias.size

                kernel_reinitialized = self._kernel_initializer(kernel.shape)
                bias_reinitialized = self._bias_initializer(bias.shape)

                new_kernel = np.copy(reference_weights[i][0])
                new_bias = np.copy(reference_weights[i][1])

                new_kernel[kernel_mask] = kernel_reinitialized[kernel_mask]
                new_bias[bias_mask] = bias_reinitialized[bias_mask]

                layer.set_weights((new_kernel, new_bias))

                layer_masks.append([kernel_mask, bias_mask])
                reinitialized_layers.append([kernel_reinitialized, bias_reinitialized])
            else:
                layer_masks.append(None)
                reinitialized_layers.append(None)


class LayerReinitializer(ABC):
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
    def apply_to_layer(self):  # TODO: move some abstraction here after POC
        pass


class DiffLayerReinitializer(LayerReinitializer):
    def __init__(self, kernel_initializer, bias_initializer):
        super().__init__(diff_ranking_over, kernel_initializer, bias_initializer)

    def apply_to_layer(self, kernel, bias, diff_kernels, diff_biases, percentage):
        where_to_reinitialize_mask = self._ranker(diff_kernels, diff_biases, percentage)  # TODO: implement
        kernel_mask = where_to_reinitialize_mask[:kernel.size].reshape(kernel.shape)
        bias_mask = where_to_reinitialize_mask[kernel.size:].reshape(bias.shape)

        initialized_kernel = self._kernel_initializer(kernel.shape)
        initialized_bias = self._bias_initializer(bias.shape)

        new_kernel = np.copy(kernel)
        new_bias = np.copy(bias)
        new_kernel[kernel_mask] = initialized_kernel[kernel_mask]
        new_bias[bias_mask] = initialized_bias[bias_mask]

        return new_kernel, new_bias


class KernelInitializer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, shape):
        pass


class KerasKernelInitializerWrapper(KernelInitializer):  # TODO: test this
    def __init__(self, keras_initializer):
        self._keras_initializer = keras_initializer()
        super().__init__()

    def __call__(self, shape):
        return K.eval(self._keras_initializer(shape))


class BiasInitializer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, shape):
        pass


class ZeroBiasInitializer(BiasInitializer):
    def __init__(self):
        super().__init__()

    def __call__(self, shape):
        return np.zeros(shape)