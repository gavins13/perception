import tensorflow as tf
import numpy as np
import sys

import collections
sys.path.insert(0, 'lib/')
from learning_core import learning_core

JoinedResult = collections.namedtuple('JoinedResult', ('summary', 'train_op', 'total_loss', 'diagnostics', 'full_diagnostics', 'output_weights'))

class multi_gpu_model(learning_core):
    def __init__(self, ArchitectureObject=None, cpu_only=False, eager=False):
        self.ArchitectureObject = None
        self.cpu_only = cpu_only
        self.eager = eager
        if(ArchitectureObject is not None):
            self.strap_architecture(ArchitectureObject)
        else:
            printt(" ... Time to load architecture ... ")

    def run_multi_gpu(self, DataObject, num_gpus=1, data=None, validation_graph=False):
        printt(">>>>Using %d GPUs" % num_gpus)
        if(self.ArchitectureObject is None):
            raise Exception('problem with architecture: not loaded')

        self.extra_data_keys = list(DataObject.extra_data._fields)
        tower_grads = []
        losses = []
        diagnostics = []

        for i in range(num_gpus):
            printt('>>Assignment of data to tower/GPU %d' % i)
            printt('>>>Data Shapes')
            this_gpu_data = {}
            for key in data.keys():
                this_gpu_data[key] = data[key][i]

            tower_output = self.single_tower(i, this_gpu_data, validation_graph=validation_graph)

            print(">>>Grad shapes")
            tower_grads.append(tower_output.grads)
            losses.append(tower_output.loss)
            diagnostics.append(tower_output.diagnostics)

        print('>> Sumarise Beta from all towers')
        summarized_results = self.summarize_towers(tower_grads, losses, diagnostics, tower_output.output_weights)
        print('>> Return Beta from all towers')
        return summarized_results




    def summarize_towers(self, tower_grads, losses, diagnostics, output_weights):
        printt("Start Averaging Gradients...")
        grads = self.average_gradients(tower_grads)
        printt("Start Averaging Diagnostics...")
        diag, full_diag = self.average_diagnostics(diagnostics)
        printt("Apply Averaged Gradients using Optimizer")
        train_op = self._optimizer.apply_gradients(grads,name="ApplyGradients")
        printt("Calculate total, summed loss")
        summed_losses = tf.reduce_sum(input_tensor=losses)
        return JoinedResult(summary, train_op, summed_losses, diag, full_diag, output_weights)
