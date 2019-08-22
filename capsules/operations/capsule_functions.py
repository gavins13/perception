import tensorflow as tf

def _squash(input_tensor):
  """ See SaraSabur GitHub for this function
  Applies norm nonlinearity (squash) to a capsule layer. Main constraint
  to the passed input tensor: the second dimensions must be vec_dim, i.e.
  the dimension containing the vector entries of a specified capsule

  Args:
    input_tensor: Input tensor. Shape is [batch,  vec_dim, num_channels] for a
      fully connected capsule layer or
      [batch,  vec_dim, num_channels, height, width] for a convolutional
      capsule layer.

  Returns:
    A tensor with same shape as input (rank 3) for output of this layer.
  """

  norm = tf.norm(input_tensor, axis=1, keepdims=True)
  norm_squared = tf.multiply(norm ,norm)
  part_b = tf.divide( input_tensor, norm)
  denom = tf.add(1., norm_squared)
  part_a = tf.divide(norm_squared , denom)
  res = tf.multiply( part_a, part_b  )
  return res
      # returned tensor has shape [batch, vec, num_channels, ((--height, width--)) ]
  '''with tf.name_scope('squashing'):
    norm = tf.norm(input_tensor, axis=1, keep_dims=True)
    norm_squared = norm * norm
    return (input_tensor / norm) * (norm_squared / (1 + norm_squared))
    # returned tensor has shape [batch, vec, num_channels, ((--height, width--)) ]'''
