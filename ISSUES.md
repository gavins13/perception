ISSUES.md

* Issue #1.1: __variables__ and __losses__ are returned in the call but is it better to set them as properties in the object i.e. within __active_vars__?

* Issue #1.2: What is the purpose of having epoch control in execution.py as well as in dataset.py? This redundancy needs fixing. Preferably keep epoch management in execution.py rather than dataset.py (which would use the TF Dataset API and method .repeat() - do not advise this method)


* Issue #2.1: Conversion from dynamic graph to static graph. Achieved using the following code from https://stackoverflow.com/questions/55149026/tensorflow-2-0-do-you-need-a-tf-function-decorator-on-top-of-each-function:
TL;DR: Decorate the training loop with the tf.function decorator and formulate the training loop as a function that can be decorated
'''
@tf.function
def train_step(features, labels):
   with tf.GradientTape() as tape:
        predictions = model(features)
        loss_value = loss(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss_value

for features, labels in dataset:
    lv = train_step(features, label)
    print("loss: ", lv)

 '''

 * Issue #3.1: When using a generator for the dataset, the TF Dataset API function .skip() does not work. Hence, this method needs to be specified in the Perception Dataset API

 * Issue #4.1: 'image' summaries do not work. Please use GIF summaries instead. (see https://github.com/tensorflow/tensorflow/issues/28007 - Issue due to using tf.summary.image within tf.function; tf.function encapsulates the entire loss when perception debug flag is set to False.)
