import numpy as np
import tensorflow as tf


def GANLoss(logits, is_real=True):
  """Computes standard GAN loss between `logits` and `labels`.

  Args:
    logits (`1-rank Tensor`): logits.
    is_real (`bool`): True means `1` labeling, False means `0` labeling.

  Returns:
    loss (`0-randk Tensor): the standard GAN loss value. (binary_cross_entropy)
  """
  if is_real:
    labels = tf.ones_like(logits)
  else:
    labels = tf.zeros_like(logits)

  return tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(labels,logits), axis=[1])


def rotation_loss(rotation_real, rotation_pred):
    batch_size = rotation_pred.shape[0]
    labels = np.full((batch_size), rotation_real)
    labels = tf.one_hot(labels, 4)
    rotation_loss = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(labels, rotation_pred), axis=[1])
      
    return rotation_loss


def discriminator_loss(real_logits, fake_logits, rotation_real, real_logits_rot):
    # losses of real with label "1"
    real_loss = GANLoss(logits=real_logits, is_real=True)
    # losses of fake with label "0"
    fake_loss = GANLoss(logits=fake_logits, is_real=False)
    rot_loss = rotation_loss(rotation_real, real_logits_rot)

    total_loss = real_loss + fake_loss + rot_loss

    return total_loss


def generator_loss(fake_logits, rotation_real, fake_logits_rot):
  # losses of Generator with label "1" that used to fool the Discriminator
  alpha = 0.2
  return GANLoss(logits=fake_logits, is_real=True) + alpha * rotation_loss(rotation_real, fake_logits_rot)
