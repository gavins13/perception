from abc import ABC, abstractmethod
import tensorflow as tf


class architecture_base(ABC):
    def __init__(self, evaluate=False):
        self.evaluate = evaluate
        class Config: pass

        self.training = Config()
        self.training.global_step = tf.Variable(initial_value=0, trainable=False, shape=[])

        self.evaluation = Config()
        self.evaluation.__enabled__ = self.evaluate
        self.evaluation.forward_passes = 1

        self.hyperparameters = Config()
        self.config = Config()
        self.config.training = Config()


        self.__config__()
        self.__set__learning__()



    def __set_learning__(self):
        self.config.training.optimizer = tf.keras.optimizers.Adam(self.hyperparameters.learning_rate, epsilon=1.e-8)

    def __end_step__(self):
        with tf.device('/cpu:0'):
            self.training.global_step += 1


    def __build__(self, builder):
        def __wrapper__(self, **args, **kwargs):
            builder()
            self.__end_step__()
        return __wrapper__

    @self.__build__
    @abstractmethod
    def build(self, input_data, ground_truth, **args, **kwargs):
        pass


    @abstractmethod
    def loss_func(self, data):
        pass

    @abstractmethod
    def analyse(self, main_results, full_diagnostics, user_diagnostics,
                save_directory):
        pass
