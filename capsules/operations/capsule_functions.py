import tensorflow as tf

def _squash(input_tensor):
  norm = tf.norm(tensor=input_tensor, axis=1, keepdims=True)
  norm_squared = tf.multiply(norm ,norm)
  part_b = tf.divide( input_tensor, norm)
  denom = tf.add(1., norm_squared)
  part_a = tf.divide(norm_squared , denom)
  res = tf.multiply( part_a, part_b  )
  return res
