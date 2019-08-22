import tensorflow as tf
import numpy as np

def margin_loss(labels, raw_logits, margin=0.4, downweight=0.5):
  """Penalizes deviations from margin for each logit.

  Each wrong logit costs its distance to margin. For negative logits margin is
  0.1 and for positives it is 0.9. First subtract 0.5 from all logits. Now
  margin is 0.4 from each side.

  Args:
    labels: tensor, one hot encoding of ground truth.
    raw_logits: tensor, model predictions in range [0, 1]
    margin: scalar, the margin after subtracting 0.5 from raw_logits.
    downweight: scalar, the factor for negative cost.

  Returns:
    A tensor with cost for each data point of shape [batch_size].
  """

  print(">>>>> Subtract 0.5 from all classifications")
  print(raw_logits.get_shape().as_list())
  print(np.shape(labels))
  logits = raw_logits - 0.5
  print(">>>>> Find Positive Cost")
  positive_cost = labels * tf.cast(tf.less(logits, margin),
                                   tf.float32) * tf.pow(logits - margin, 2)
  print(">>>>> Find Negative Cost")
  negative_cost = (1 - labels) * tf.cast(
      tf.greater(logits, -margin), tf.float32) * tf.pow(logits + margin, 2)
  print(">>>>> Return loss")
  return 0.5 * positive_cost + downweight * 0.5 * negative_cost
