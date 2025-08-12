import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond as tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
import qkeras as qk
import numpy as np

@tf.keras.utils.register_keras_serializable()
class FullyQDense(qk.QDense):
    
    def __init__(self,
                 units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer="he_normal",
                 bias_initializer="zeros",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 data_constraint=None,
                 kernel_quantizer=None,
                 bias_quantizer=None,
                 data_quantizer=None,
                 kernel_range=None,
                 bias_range=None,
                 **kwargs):

        super(FullyQDense, self).__init__(units, activation=activation, use_bias=use_bias,
                                          kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                          kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                                          activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
                                          bias_constraint=bias_constraint, kernel_quantizer=kernel_quantizer,
                                          bias_quantizer=bias_quantizer, kernel_range=kernel_range,
                                          bias_range=bias_range, **kwargs)

        self.data_quantizer = data_quantizer
        self.data_quantizer_internal = qk.get_quantizer(data_quantizer)
        self.quantizers.append(self.data_quantizer_internal)
        self.data_constraint = qk.get_constraint(self.data_quantizer_internal, data_constraint) if data_quantizer \
            else None

    def call(self, inputs):
        if self.kernel_quantizer:
            quantized_kernel = self.kernel_quantizer_internal(self.kernel)
        else:
            quantized_kernel = self.kernel
        output = tf.keras.backend.dot(inputs, quantized_kernel)
        if self.data_quantizer:
            output = self.data_quantizer_internal(output)
        if self.use_bias:
            if self.bias_quantizer:
                quantized_bias = self.bias_quantizer_internal(self.bias)
            else:
                quantized_bias = self.bias
            output = tf.keras.backend.bias_add(output, quantized_bias,
                                               data_format="channels_last")
            if self.data_quantizer:
                output = self.data_constraint(output)
                # output = self.data_quantizer_internal(output)

        if self.activation is not None:
            # The activation function should handle quantization by itself, e.g. by specifying the quantized version
            # of the function. Thus, no explicit quantization of the activation output is done here.
            output = self.activation(output)
        return output

    def get_config(self):
        config = {
            "data_quantizer": tf.keras.constraints.serialize(self.data_quantizer_internal),
            "data_constraint": tf.keras.constraints.serialize(self.data_constraint)
        }
        base_config = super(FullyQDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_quantization_config(self):
        config = {
            "data_quantizer":
                str(self.data_quantizer_internal),
        }
        base_config = super(FullyQDense, self).get_quantization_config()
        return dict(list(base_config.items()) + list(config.items()))

@tf.keras.utils.register_keras_serializable()
class FullyQConv2D(qk.QConv2D):

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding="valid",
                 data_format="channels_last",
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer="he_normal",
                 bias_initializer="zeros",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 data_constraint=None,
                 kernel_range=None,
                 bias_range=None,
                 kernel_quantizer=None,
                 bias_quantizer=None,
                 data_quantizer=None,
                 **kwargs):

        super(FullyQConv2D, self).__init__(filters, kernel_size, strides=strides, padding=padding,
                                           data_format=data_format, dilation_rate=dilation_rate, activation=activation,
                                           use_bias=use_bias, kernel_initializer=kernel_initializer,
                                           bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer,
                                           bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer,
                                           kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
                                           kernel_range=kernel_range, bias_range=bias_range,
                                           kernel_quantizer=kernel_quantizer, bias_quantizer=bias_quantizer, **kwargs)

        self.data_quantizer = data_quantizer
        self.data_quantizer_internal = qk.get_quantizer(data_quantizer)
        self.quantizers.append(self.data_quantizer_internal)
        self.data_constraint = qk.get_constraint(self.data_quantizer_internal, data_constraint) if data_quantizer \
            else None

    def call(self, inputs):
        if self.kernel_quantizer:
            quantized_kernel = self.kernel_quantizer_internal(self.kernel)
        else:
            quantized_kernel = self.kernel

        outputs = tf.keras.backend.conv2d(
            inputs,
            quantized_kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)

        if self.data_quantizer:
            outputs = self.data_quantizer_internal(outputs)

        if self.use_bias:
            if self.bias_quantizer:
                quantized_bias = self.bias_quantizer_internal(self.bias)
            else:
                quantized_bias = self.bias

            outputs = tf.keras.backend.bias_add(
                outputs, quantized_bias, data_format=self.data_format)

            if self.data_quantizer:
                outputs = self.data_constraint(outputs)
                # outputs = self.data_quantizer_internal(outputs)

        if self.activation is not None:
            # The activation function should handle quantization by itself, e.g. by specifying the quantized version
            # of the function. Thus, no explicit quantization of the activation output is done here.
            return self.activation(outputs)
        return outputs

    def get_config(self):
        config = {
            "data_quantizer": tf.keras.constraints.serialize(self.data_quantizer_internal),
            "data_constraint": tf.keras.constraints.serialize(self.data_constraint)
        }
        base_config = super(FullyQConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_quantization_config(self):
        config = {
            "data_quantizer":
                str(self.data_quantizer_internal),
        }
        base_config = super(FullyQConv2D, self).get_quantization_config()
        return dict(list(base_config.items()) + list(config.items()))

@tf.keras.utils.register_keras_serializable()
class FullyQDepthwiseConv2D(qk.QDepthwiseConv2D):

    def __init__(self,
                 kernel_size,
                 strides=(1, 1),
                 padding="VALID",
                 depth_multiplier=1,
                 data_format=None,
                 activation=None,
                 use_bias=True,
                 depthwise_initializer="he_normal",
                 bias_initializer="zeros",
                 depthwise_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 depthwise_constraint=None,
                 bias_constraint=None,
                 data_constraint=None,
                 dilation_rate=(1, 1),
                 depthwise_quantizer=None,
                 bias_quantizer=None,
                 data_quantizer=None,
                 depthwise_range=None,
                 bias_range=None,
                 **kwargs):

        super(FullyQDepthwiseConv2D, self).__init__(kernel_size, strides=strides, padding=padding,
                                                    depth_multiplier=depth_multiplier, data_format=data_format,
                                                    activation=activation, use_bias=use_bias,
                                                    depthwise_initializer=depthwise_initializer,
                                                    bias_initializer=bias_initializer,
                                                    depthwise_regularizer=depthwise_regularizer,
                                                    bias_regularizer=bias_regularizer,
                                                    activity_regularizer=activity_regularizer,
                                                    depthwise_constraint=depthwise_constraint,
                                                    bias_constraint=bias_constraint,
                                                    dilation_rate=dilation_rate,
                                                    depthwise_quantizer=depthwise_quantizer,
                                                    bias_quantizer=bias_quantizer, depthwise_range=depthwise_range,
                                                    bias_range=bias_range, **kwargs)

        self.data_quantizer = data_quantizer
        self.data_quantizer_internal = qk.get_quantizer(data_quantizer)
        self.quantizers.append(self.data_quantizer_internal)
        self.data_constraint = qk.get_constraint(self.data_quantizer_internal, data_constraint) if data_quantizer \
            else None

    def call(self, inputs, training=None):
        if self.depthwise_quantizer:
            quantized_depthwise_kernel = (
                self.depthwise_quantizer_internal(self.depthwise_kernel))
        else:
            quantized_depthwise_kernel = self.depthwise_kernel
        outputs = tf.keras.backend.depthwise_conv2d(
            inputs,
            quantized_depthwise_kernel,
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            data_format=self.data_format)

        if self.data_quantizer:
            outputs = self.data_quantizer_internal(outputs)

        if self.use_bias:
            if self.bias_quantizer:
                quantized_bias = self.bias_quantizer_internal(self.bias)
            else:
                quantized_bias = self.bias
            outputs = tf.keras.backend.bias_add(
                outputs, quantized_bias, data_format=self.data_format)

            if self.data_quantizer:
                outputs = self.data_constraint(outputs)
                # outputs = self.data_quantizer_internal(outputs)

        if self.activation is not None:
            # The activation function should handle quantization by itself, e.g. by specifying the quantized version
            # of the function. Thus, no explicit quantization of the activation output is done here.
            return self.activation(outputs)

        return outputs

    def get_config(self):
        config = {
            "data_quantizer": tf.keras.constraints.serialize(self.data_quantizer_internal),
            "data_constraint": tf.keras.constraints.serialize(self.data_constraint)
        }
        base_config = super(FullyQDepthwiseConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_quantization_config(self):
        config = {
            "data_quantizer":
                str(self.data_quantizer_internal),
        }
        base_config = super(FullyQDepthwiseConv2D, self).get_quantization_config()
        return dict(list(base_config.items()) + list(config.items()))

@tf.keras.utils.register_keras_serializable()
class FullyQSeparableConv2D(qk.QSeparableConv2D):

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 depth_multiplier=1,
                 activation=None,
                 use_bias=True,
                 depthwise_initializer='glorot_uniform',
                 pointwise_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 depthwise_regularizer=None,
                 pointwise_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 depthwise_constraint=None,
                 pointwise_constraint=None,
                 bias_constraint=None,
                 data_constraint=None,
                 depthwise_quantizer=None,
                 pointwise_quantizer=None,
                 bias_quantizer=None,
                 data_quantizer=None,
                 **kwargs):

        super(FullyQSeparableConv2D, self).__init__(filters, kernel_size, strides=strides, padding=padding,
                                                    data_format=data_format, dilation_rate=dilation_rate,
                                                    depth_multiplier=depth_multiplier, activation=activation,
                                                    use_bias=use_bias, depthwise_initializer=depthwise_initializer,
                                                    pointwise_initializer=pointwise_initializer,
                                                    bias_initializer=bias_initializer,
                                                    depthwise_regularizer=depthwise_regularizer,
                                                    pointwise_regularizer=pointwise_regularizer,
                                                    bias_regularizer=bias_regularizer,
                                                    activity_regularizer=activity_regularizer,
                                                    depthwise_constraint=depthwise_constraint,
                                                    pointwise_constraint=pointwise_constraint,
                                                    bias_constraint=bias_constraint,
                                                    depthwise_quantizer=depthwise_quantizer,
                                                    pointwise_quantizer=pointwise_quantizer,
                                                    bias_quantizer=bias_quantizer,
                                                    **kwargs)

        self.data_quantizer = data_quantizer
        self.data_quantizer_internal = qk.get_quantizer(data_quantizer)
        self.quantizers.append(self.data_quantizer_internal)
        self.data_constraint = qk.get_constraint(self.data_quantizer_internal, data_constraint) if data_quantizer \
            else None

    def call(self, inputs):
        # Apply the actual ops.
        if self.depthwise_quantizer:
            quantized_depthwise_kernel = self.depthwise_quantizer_internal(
                self.depthwise_kernel)
        else:
            quantized_depthwise_kernel = self.depthwise_kernel

        if self.pointwise_quantizer:
            quantized_pointwise_kernel = self.pointwise_quantizer_internal(
                self.pointwise_kernel)
        else:
            quantized_pointwise_kernel = self.pointwise_kernel

        outputs = tf.keras.backend.separable_conv2d(
            inputs,
            quantized_depthwise_kernel,
            quantized_pointwise_kernel,
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            data_format=self.data_format)

        if self.data_quantizer:
            outputs = self.data_quantizer_internal(outputs)

        if self.use_bias:
            if self.bias_quantizer:
                quantized_bias = self.bias_quantizer_internal(self.bias)
            else:
                quantized_bias = self.bias

            outputs = tf.keras.backend.bias_add(
                outputs,
                quantized_bias,
                data_format=self.data_format)

            if self.data_quantizer:
                outputs = self.data_constraint(outputs)
                # outputs = self.data_quantizer_internal(outputs)

        if self.activation is not None:
            # The activation function should handle quantization by itself, e.g. by specifying the quantized version
            # of the function. Thus, no explicit quantization of the activation output is done here.
            return self.activation(outputs)
        return outputs

    def get_config(self):
        config = {
            "data_quantizer": tf.keras.constraints.serialize(self.data_quantizer_internal),
            "data_constraint": tf.keras.constraints.serialize(self.data_constraint)
        }
        base_config = super(FullyQSeparableConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

@tf.keras.utils.register_keras_serializable()
class FullyQBatchNormalization(qk.QBatchNormalization):
    def __init__(self,
                 axis=-1,
                 momentum=0.99,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 activation=None,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 moving_mean_initializer='zeros',
                 moving_variance_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_quantizer=None,
                 gamma_quantizer=None,
                 mean_quantizer=None,
                 data_quantizer=None,
                 variance_quantizer=None,
                 gamma_constraint=None,
                 beta_constraint=None,
                 data_constraint=None,
                 # use quantized_po2 and enforce quadratic approximation
                 # to get an even exponent for sqrt
                 beta_range=None,
                 gamma_range=None,
                 **kwargs):

        super(FullyQBatchNormalization, self).__init__(axis=axis, momentum=momentum, epsilon=epsilon, center=center,
                                                       scale=scale, activation=activation,
                                                       beta_initializer=beta_initializer,
                                                       gamma_initializer=gamma_initializer,
                                                       moving_mean_initializer=moving_mean_initializer,
                                                       moving_variance_initializer=moving_variance_initializer,
                                                       beta_regularizer=beta_regularizer,
                                                       gamma_regularizer=gamma_regularizer,
                                                       beta_quantizer=beta_quantizer, gamma_quantizer=gamma_quantizer,
                                                       mean_quantizer=mean_quantizer,
                                                       variance_quantizer=variance_quantizer,
                                                       gamma_constraint=gamma_constraint,
                                                       beta_constraint=beta_constraint, beta_range=beta_range,
                                                       gamma_range=gamma_range, **kwargs)

        self.data_quantizer = data_quantizer
        self.data_quantizer_internal = qk.get_quantizer(data_quantizer)
        self.quantizers.append(self.data_quantizer_internal)
        self.data_constraint = qk.get_constraint(self.data_quantizer_internal, data_constraint) if data_quantizer \
            else None

    def call(self, inputs, training=None):
        if self.scale and self.gamma_quantizer:
            quantized_gamma = self.gamma_quantizer_internal(self.gamma)
        else:
            quantized_gamma = self.gamma

        if self.center and self.beta_quantizer:
            quantized_beta = self.beta_quantizer_internal(self.beta)
        else:
            quantized_beta = self.beta

        if self.mean_quantizer:
            quantized_moving_mean = self.mean_quantizer_internal(self.moving_mean)
        else:
            quantized_moving_mean = self.moving_mean

        if self.variance_quantizer:
            quantized_moving_variance = self.variance_quantizer_internal(
                self.moving_variance)
        else:
            quantized_moving_variance = self.moving_variance

        training = self._get_training_value(training)

        # Compute the axes along which to reduce the mean / variance
        input_shape = inputs.shape
        ndims = len(input_shape)
        reduction_axes = [i for i in range(ndims) if i not in self.axis]

        # Broadcasting only necessary for single-axis batch norm where the axis is
        # not the last dimension
        broadcast_shape = [1] * ndims
        broadcast_shape[self.axis[0]] = input_shape.dims[self.axis[0]].value

        def _broadcast(v):
            if (v is not None and len(v.shape) != ndims and
                    reduction_axes != list(range(ndims - 1))):
                return array_ops.reshape(v, broadcast_shape)
            return v

        scale, offset = _broadcast(quantized_gamma), _broadcast(quantized_beta)

        # Determine a boolean value for `training`: could be True, False, or None.
        training_value = tf_utils.smart_constant_value(training)
        if training_value == False:  # pylint: disable=singleton-comparison,g-explicit-bool-comparison
            quantized_mean, quantized_variance = (quantized_moving_mean,
                                                  quantized_moving_variance)
        else:
            # Some of the computations here are not necessary when training==False
            # but not a constant. However, this makes the code simpler.
            keep_dims = len(self.axis) > 1
            mean, variance = self._moments(
                math_ops.cast(inputs, self._param_dtype),
                reduction_axes,
                keep_dims=keep_dims)

            moving_mean = self.moving_mean
            moving_variance = self.moving_variance

            mean = tf_utils.smart_cond(
                training, lambda: mean, lambda: ops.convert_to_tensor(moving_mean))
            variance = tf_utils.smart_cond(
                training,
                lambda: variance,
                lambda: ops.convert_to_tensor(moving_variance))

            new_mean, new_variance = mean, variance

            if self.mean_quantizer:
                quantized_mean = self.mean_quantizer_internal(mean)
            else:
                quantized_mean = mean

            if self.variance_quantizer:
                quantized_variance = self.variance_quantizer_internal(variance)
            else:
                quantized_variance = variance

            if self._support_zero_size_input():
                inputs_size = array_ops.size(inputs)
            else:
                inputs_size = None

            def _do_update(var, value):
                """Compute the updates for mean and variance."""
                return self._assign_moving_average(var, value, self.momentum,
                                                   inputs_size)

            def mean_update():
                true_branch = lambda: _do_update(self.moving_mean, new_mean)
                false_branch = lambda: self.moving_mean
                return tf_utils.smart_cond(training, true_branch, false_branch)

            def variance_update():
                """Update the moving variance."""
                true_branch = lambda: _do_update(self.moving_variance, new_variance)
                false_branch = lambda: self.moving_variance
                return tf_utils.smart_cond(training, true_branch, false_branch)

            self.add_update(mean_update)
            self.add_update(variance_update)

        quantized_mean = math_ops.cast(quantized_mean, inputs.dtype)
        quantized_variance = math_ops.cast(quantized_variance, inputs.dtype)
        if offset is not None:
            offset = math_ops.cast(offset, inputs.dtype)
        if scale is not None:
            scale = math_ops.cast(scale, inputs.dtype)
        # TODO(reedwm): Maybe do math in float32 if given float16 inputs, if doing
        #  math in float16 hurts validation accuracy of popular models like resnet.
        outputs = nn.batch_normalization(inputs,
                                         _broadcast(quantized_mean),
                                         _broadcast(quantized_variance),
                                         offset,
                                         scale,
                                         self.epsilon)
        # If some components of the shape got lost due to adjustments, fix that.
        outputs.set_shape(input_shape)

        if self.data_quantizer:
            outputs = self.data_quantizer_internal(outputs)

        return outputs

    def get_config(self):
        config = {
            "data_quantizer": tf.keras.constraints.serialize(self.data_quantizer_internal),
            "data_constraint": tf.keras.constraints.serialize(self.data_constraint)
        }
        base_config = super(FullyQBatchNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class FullyQAveragePooling2D(qk.QAveragePooling2D):

    def __init__(self,
                 pool_size=(2, 2),
                 strides=None,
                 padding="valid",
                 data_format=None,
                 average_quantizer=None,
                 data_quantizer=None,
                 activation=None,
                 **kwargs):
        super(FullyQAveragePooling2D, self).__init__(pool_size=pool_size, strides=strides, padding=padding,
                                                     data_format=data_format, average_quantizer=average_quantizer,
                                                     activation=activation, **kwargs)

        self.data_quantizer = data_quantizer
        self.data_quantizer_internal = qk.get_quantizer(data_quantizer)
        self.quantizers.append(self.data_quantizer_internal)

    def call(self, inputs):
        x = tf.keras.layers.AveragePooling2D.call(self, inputs)

        if self.average_quantizer:
            if isinstance(self.pool_size, int):
                pool_area = self.pool_size * self.pool_size
            else:
                pool_area = np.prod(self.pool_size)

            # Revertes the division results.
            x = x * pool_area

            # Quantizes the multiplication factor.
            mult_factor = 1.0 / pool_area

            q_mult_factor = self.average_quantizer_internal(mult_factor)
            q_mult_factor = K.cast_to_floatx(q_mult_factor)
            x = x * q_mult_factor

            if self.data_quantizer:
                x = self.data_quantizer_internal(x)

        if self.activation is not None:
            return self.activation(x)
        return x

    def get_config(self):
        config = {
            "data_quantizer": tf.keras.constraints.serialize(self.data_quantizer_internal),
        }
        base_config = super(FullyQAveragePooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_quantization_config(self):
        config = {
            "data_quantizer":
                str(self.data_quantizer_internal),
        }
        base_config = super(FullyQAveragePooling2D, self).get_quantization_config()
        return dict(list(base_config.items()) + list(config.items()))
