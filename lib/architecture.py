from abc import ABC, abstractmethod
import tensorflow as tf

class architecture_base(ABC):
    def __init__(self):
        self.hparams = tf.contrib.training.HParams(
          decay_rate=0.9,
          decay_steps=1000.,
          learning_rate=1.e-5, # 0.001
          maximum_learning_rate = 1.e-7, # 1.e-7
        )

    @abstractmethod
    def build(self, input_images):
        raise NotImplementedError()

    @abstractmethod
    def loss_func(self, input_images, ground_truth, validation_input_images, validation_ground_truth):
        raise NotImplementedError()
