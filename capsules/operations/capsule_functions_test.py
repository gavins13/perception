# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tests for layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf

from capsule_functions import _squash


class LayersTest(tf.test.TestCase):

  def testSquashRankSix(self):
    """Checks the value and shape of the squash output given a rank 6 input."""
    input_tensor = tf.ones((1, 1, 1, 1, 1, 1))
    squashed = _squash(input_tensor)
    self.assertEqual(len(squashed.get_shape()), 6)
    with self.test_session() as sess:
      r_squashed = sess.run(squashed)
    scale = 0.5
    self.assertEqual(np.array(r_squashed).shape, input_tensor.get_shape())
    self.assertAllClose(np.linalg.norm(r_squashed, axis=2), [[[[[scale]]]]])
