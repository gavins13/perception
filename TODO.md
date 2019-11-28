class Architecture:
    - specify the end of epoch test to run (if any)
class Biobank:
    - generator function
class LearningCore:
    - optimizers grabbed from architecture
    - gradienttapes
    - training execution
    - testing execution
class Execution:
    - load Config.perception file
    - load Experiments.perception file
    - Checkpointing, restoration, writing the saves to files
    - Keeps track of experiments by writing to a experiments txt file
    - Loads tensorboard during execution using port number specified in the exp config using no GPUs and 
    - 
Config.perception file:
    Specify:
        - default GPUs
        - default CPUs (or calc from GPUs)
        - tensorboard CPUs
        - dfault tensorboard port
        - Default save directory
Files:
- SystemResources: Detect CMD argument
- TensorboardResources: GIF summaries
- 

https://medium.com/@nimatajbakhsh/building-multi-threaded-custom-data-pipelines-for-tensorflow-f76e9b1a32f5
Dataset = tf.data.Dataset
ds = Dataset.from_tensor_slices(['Gen_0', 'Gen_1', 'Gen_2'])
ds = ds.interleave(lambda x: Dataset.from_generator(py_gen, output_types=(tf.string), args=(x,)),
                   cycle_length=3,
                   block_length=1,
                   num_parallel_calls=3)