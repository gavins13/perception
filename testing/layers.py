from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import glob

import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

import PIL
import imageio
from IPython import display

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Wrapper



class SpectralNormalization(Wrapper):
    """
    Attributes:
       layer: tensorflow keras layers (with kernel attribute)
    """

    def __init__(self, layer, **kwargs):
        super(SpectralNormalization, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        """Build `Layer`"""

        if not self.layer.built:
            self.layer.build(input_shape)

            if not hasattr(self.layer, 'kernel'):
                raise ValueError(
                    '`SpectralNormalization` must wrap a layer that'
                    ' contains a `kernel` for weights')

            self.w = self.layer.kernel
            self.w_shape = self.w.shape.as_list()
            self.u = self.add_variable(
                shape=tuple([1, self.w_shape[-1]]),
                initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
                name='sn_u',
                trainable=False,
                dtype=tf.float32)

        super(SpectralNormalization, self).build()

    @tf.function
    def call(self, inputs):
        """Call `Layer`"""
        # Recompute weights for each forward pass
        self._compute_weights()
        output = self.layer(inputs)
        return output

    def _compute_weights(self):
        """Generate normalized weights.
        This method will update the value of self.layer.kernel with the
        normalized value, so that the layer is ready for call().
        """
        w_reshaped = tf.reshape(self.w, [-1, self.w_shape[-1]])
        eps = 1e-12
        _u = tf.identity(self.u)
        _v = tf.matmul(_u, tf.transpose(w_reshaped))
        _v = _v / tf.maximum(tf.reduce_sum(_v**2)**0.5, eps)
        _u = tf.matmul(_v, w_reshaped)
        _u = _u / tf.maximum(tf.reduce_sum(_u**2)**0.5, eps)

        self.u.assign(_u)
        sigma = tf.matmul(tf.matmul(_v, w_reshaped), tf.transpose(_u))

        #self.layer.kernel = self.w / sigma
        self.layer.kernel.assign(self.w / sigma)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(
            self.layer.compute_output_shape(input_shape).as_list())


class Generator(tf.keras.Model):

  def __init__(self, spectral_norm=True):
    super(Generator, self).__init__()

    def Conv2DTranspose(*args, **kwargs):
        layer = layers.Conv2DTranspose(*args, **kwargs)
        layer = SpectralNormalization(layer) if spectral_norm is True else layer
        return layer

    self.conv1 = Conv2DTranspose(filters=256, kernel_size=(3, 3), strides=(2, 2), use_bias=False)
    self.conv1_bn = layers.BatchNormalization()
    self.conv2 = Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(2, 2), use_bias=False)
    self.conv2_bn = layers.BatchNormalization()
    self.conv3 = Conv2DTranspose(filters=64, kernel_size=(4, 4), strides=(2, 2), use_bias=False)
    self.conv3_bn = layers.BatchNormalization()
    self.conv4 = Conv2DTranspose(filters=3, kernel_size=(4, 4), strides=(2, 2), padding='same')

  def call(self, inputs, training=True):
    """Run the model."""
    conv1 = self.conv1(inputs)
    conv1_bn = self.conv1_bn(conv1, training=training)
    conv1 = tf.nn.relu(conv1_bn)

    conv2 = self.conv2(conv1)
    conv2_bn = self.conv2_bn(conv2, training=training)
    conv2 = tf.nn.relu(conv2_bn)

    conv3 = self.conv3(conv2)
    conv3_bn = self.conv3_bn(conv3, training=training)
    conv3 = tf.nn.relu(conv3_bn)

    conv4 = self.conv4(conv3)
    generated_data = tf.nn.sigmoid(conv4)

    return generated_data


class Discriminator(tf.keras.Model):

  def __init__(self, spectral_norm=True):
    super(Discriminator, self).__init__()
    def Conv2D(*args, **kwargs):
        layer = layers.Conv2D(*args, **kwargs)
        layer = SpectralNormalization(layer) if spectral_norm is True else layer
        return layer

    self.conv1 = Conv2D(64, (4, 4), strides=(2, 2), padding='same')
    self.conv2 = Conv2D(128, (4, 4), strides=(2, 2), use_bias=False)
    self.conv2_bn = layers.BatchNormalization()
    self.conv3 = Conv2D(256, (3, 3), strides=(2, 2), use_bias=False)
    self.conv3_bn = layers.BatchNormalization()

    self.conv4_base = Conv2D(1, (3, 3))

    self.conv4_rot = layers.Dense(4, input_shape=(512, 256*3*3))


  def call(self, inputs, training=True, predict_rotation=False):
    conv1 = tf.nn.leaky_relu(self.conv1(inputs))
    conv2 = self.conv2(conv1)
    conv2_bn = self.conv2_bn(conv2, training=training)
    conv3 = self.conv3(conv2_bn)
    conv3_bn = self.conv3_bn(conv3, training=training)
    if predict_rotation:
      conv3_flattened = tf.reshape(conv3_bn, (tf.shape(conv3_bn)[0], -1))
      rotation_class = self.conv4_rot(conv3_flattened)
      return rotation_class
    else:
      conv4_base = self.conv4_base(conv3_bn)
      discriminator_logits = tf.squeeze(conv4_base, axis=[1, 2])
      return discriminator_logits






  class DeepRepDiscriminator(tf.keras.Model):
    def __init__(self, spectral_norm=True):
      super(Discriminator, self).__init__()
      Conv2D = SpectralNormalization(layers.Conv2D) if spectral_norm is True else layers.Conv2D
      self.conv1 = Conv2D(64, (4, 4), strides=(2, 2), padding='same')
      self.conv2 = Conv2D(128, (4, 4), strides=(2, 2), use_bias=False)
      self.conv2_bn = layers.BatchNormalization()
      self.conv3 = Conv2D(256, (3, 3), strides=(2, 2), use_bias=False)
      self.conv3_bn = layers.BatchNormalization()

      self.conv4_base = Conv2D(1, (3, 3))

      self.conv4_rot = layers.Dense(4, input_shape=(512, 256*3*3))


    def call(self, inputs, training=True, predict_rotation=False):
      conv1 = tf.nn.leaky_relu(self.conv1(inputs))
      conv2 = self.conv2(conv1)
      conv2_bn = self.conv2_bn(conv2, training=training)
      conv3 = self.conv3(conv2_bn)
      conv3_bn = self.conv3_bn(conv3, training=training)
      if predict_rotation:
        conv3_flattened = tf.reshape(conv3_bn, (tf.shape(conv3_bn)[0], -1))
        rotation_class = self.conv4_rot(conv3_flattened)
        return rotation_class
      else:
        conv4_base = self.conv4_base(conv3_bn)
        discriminator_logits = tf.squeeze(conv4_base, axis=[1, 2])
        return discriminator_logits
