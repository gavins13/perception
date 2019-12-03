# Code from : https://github.com/ilguyi/gans.tensorflow.v2/blob/master/tf.v2/01.dcgan.ipynb
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import time
import glob
import h5py

import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

import PIL
import imageio
from IPython import display

import tensorflow as tf
from tensorflow.keras import layers
#tf.enable_eager_execution()

tf.get_logger().setLevel('ERROR')
#tf.logging.set_verbosity(tf.logging.INFO)

os.environ["CUDA_VISIBLE_DEVICES"]="0"

from layers import Generator, Discriminator
from loss import discriminator_loss, generator_loss

sys.path.insert(0, '/vol/biomedic/users/kgs13/PhD/projects/perception_tf2/data/biobank/')
from BiobankDataLoader import BioBank

# Training Flags (hyperparameter configuration)
train_dir = 'train/01.dcgan/exp1/'
max_epochs = 5000
save_epochs = 10
print_steps = 100
batch_size = 256
learning_rate_D = 0.0002
learning_rate_G = 0.0002
k = 1 # the number of step of learning D before learning G
num_examples_to_generate = 16
noise_dim = 100
full_save_epochs = 1
full_save_num_images = 49920
second_unpaired = True


# Load training and eval data from tf.keras
(train_data, train_labels), _ = \
    tf.keras.datasets.cifar10.load_data()


train_data = train_data.reshape(-1, 32, 32, 3).astype('float32')
train_data = train_data / 255.
train_labels = np.asarray(train_labels, dtype=np.int32)

train_data = train_data[0:49920]
train_labels = train_labels[0:49920]


tf.random.set_seed(69)
operation_seed = None

# for train
train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
train_dataset = train_dataset.shuffle(buffer_size = 60000)
train_dataset = train_dataset.batch(batch_size = batch_size)

biobank = BioBank()
biobank.use_generator((tf.float32))
biobank.create()
train_dataset = biobank.Datasets[0]

generator = Generator()
discriminator = Discriminator()

# Defun for performance boost
#generator.call = tf.contrib.eager.defun(generator.call)
#discriminator.call = tf.contrib.eager.defun(discriminator.call)

#discriminator_optimizer = tf.train.AdamOptimizer(learning_rate_D, beta1=0.5)
#discriminator_optimizer = tf.train.RMSPropOptimizer(learning_rate_D)
#generator_optimizer = tf.train.AdamOptimizer(learning_rate_G, beta1=0.5)

discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate_D, beta_1=0.5)
if second_unpaired is True:
    discriminator_optimizer_2 = tf.keras.optimizers.Adam(learning_rate_D, beta_1=0.5)
generator_optimizer = tf.keras.optimizers.Adam(learning_rate_G, beta_1=0.5)


'''
Checkpointing
'''
#checkpoint_dir = './training_checkpoints'
checkpoint_dir =  train_dir
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
for_checkpointing = {
    'generator_optimizer': generator_optimizer,
    'discriminator_optimizer': discriminator_optimizer,
    'generator': generator,
    'discriminator': discriminator
    }
if second_unpaired is True:
    for_checkpointing["discriminator_optimizer_2"] =  discriminator_optimizer_2
checkpoint = tf.train.Checkpoint(**for_checkpointing)

'''
Training
'''
# keeping the random vector constant for generation (prediction) so
# it will be easier to see the improvement of the gan.
random_vector_for_generation = tf.random.normal([num_examples_to_generate, 1, 1, noise_dim])

def generate_and_save_images(model, epoch, test_input):
  # make sure the training parameter is set to False because we
  # don't want to train the batchnorm layer when doing inference.
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4, 4))

  for i in range(predictions.shape[0]):
    plt.subplot(4, 4, i+1)
    plt.imshow(predictions[i, :, :, 0], cmap='gray')
    plt.axis('off')

  plt.savefig('images/image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()
  plt.close('all')

def stretch(img, s1, s2):
  old_shape = np.asarray(img.shape)
  stretched = zoom(img, zoom=(s1, s2, 1))
  new_shape = np.asarray(stretched.shape)
  min = (new_shape - old_shape) / 2
  max = new_shape - min
  stretched = stretched[int(min[0]):int(max[0]), int(min[1]):int(max[1])]
  return stretched

def print_or_save_sample_images(sample_data, max_print=num_examples_to_generate, is_save=False, epoch=None, prefix=""):
  print_images = sample_data[:max_print,:]
  print_images = print_images.reshape([max_print, 32, 32, 3])
  print_images = print_images.swapaxes(0, 1)
  print_images = print_images.reshape([32, max_print * 32, 3])

  plt.figure(figsize=(max_print, 1))
  plt.axis('off')
  plt.imshow(print_images, cmap='gray')

  if is_save is True:
    plt.savefig(prefix+'image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()
  plt.close('all')


'''
Training Loop
'''
#global_step = tf.train.get_or_create_global_step()
step = 0
for epoch in range(max_epochs):

  for images in train_dataset:
    images = tf.image.resize(images,[32,32])
    batch_size=images.shape[0]
    start_time = time.time()

    # generating noise from a uniform distribution
    noise = tf.random.normal([batch_size, 1, 1, noise_dim])
    rotation_n = tf.random.uniform([], minval=0, maxval=3, dtype=tf.dtypes.int32, seed=operation_seed)
    if second_unpaired is True:
        noise_2 = tf.random.normal([batch_size, 1, 1, noise_dim])
    rotation = tf.cast(rotation_n, dtype=tf.float32) * np.pi/2.

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape, tf.GradientTape() as disc_rot_tape, tf.GradientTape() as disc_2_tape:
      generated_images = generator(noise, training=True)
      #print('.',images.shape)
      images_rot = tf.image.rot90(images, k=rotation_n)
      generated_images_rot = tf.image.rot90(generated_images, k=rotation_n)

      real_logits = discriminator(images, training=True)
      fake_logits = discriminator(generated_images, training=True)
      real_logits_rot = discriminator(images_rot, training=True, predict_rotation=True)
      fake_logits_rot = discriminator(generated_images_rot, training=True, predict_rotation=True)

      if second_unpaired is True:
          generated_images_2 = generator(noise_2, training=True)
          fake_logits_2 = discriminator(generated_images_2, training=True)
          disc_loss_2 = discriminator_loss(real_logits, fake_logits_2, rotation_n, real_logits_rot) # [] CHECK

      gen_loss = generator_loss(fake_logits, rotation_n, fake_logits_rot)
      disc_loss = discriminator_loss(real_logits, fake_logits, rotation_n, real_logits_rot)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.variables)
    # gradients_of_discriminator_rot = disc_rot_tape.gradient(disc_loss_rot, discriminator.variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.variables))
    # discriminator_rot_optimizer.apply_gradients(zip(gradients_of_discriminator_rot, discriminator.variables))

    if second_unpaired is True:
        gradients_of_discriminator_2 = disc_2_tape.gradient(disc_loss_2, discriminator.variables)
        discriminator_optimizer_2.apply_gradients(zip(gradients_of_discriminator_2, discriminator.variables))

    epochs = step * batch_size / float(len(train_data))
    duration = time.time() - start_time
    if step % print_steps == 0:
      display.clear_output(wait=True)
      examples_per_sec = batch_size / float(duration)
      #print("Epochs: {:.2f} global_step: {} loss_D: {:.3f} loss_G: {:.3f} ({:.2f} examples/sec; {:.3f} sec/batch)".format(
      #          epochs,
      #          step,
      #          np.mean(disc_loss),
      #          np.mean(gen_loss),
      #          examples_per_sec,
      #          duration))
      sample_data = generator(random_vector_for_generation, training=False)
      print_or_save_sample_images(sample_data.numpy())

    step += 1
    print("%d         " % (step), end="\r")

    #if epoch % 1 == 0:
    if step % 20 == 0:
        images = tf.tile(images, [16,1,1,1])
        images_rot = tf.tile(images, [16,1,1,1])
        print("%d  saving" % (step), end="\r")
        display.clear_output(wait=True)
        print("This images are saved at {} epoch".format(epoch+1))
        sample_data = generator(random_vector_for_generation, training=False)
        print_or_save_sample_images(sample_data.numpy(), is_save=True, epoch=epoch+1)
        print_or_save_sample_images(images.numpy(), is_save=True, epoch=epoch+1, prefix="REAL_")
        print_or_save_sample_images(images_rot.numpy(), is_save=True, epoch=epoch+1, prefix="REAL_TRANSFORMED_")

  # saving (checkpoint) the model every save_epochs
  if (epoch + 1) % save_epochs == 0:
    checkpoint.save(file_prefix = checkpoint_prefix)
    

  # Save full batch of training images to calculate FID
  if (epoch + 1) % full_save_epochs == 0:
    for batch in range(full_save_num_images // batch_size):
      random_vector_for_full_save = tf.random.normal([batch_size, 1, 1, noise_dim])
      sample_data = generator(random_vector_for_full_save, training=False)
      sample_blob = sample_data.numpy()
      h5f = h5py.File('imageblob_epoch{}_{}.h5'.format(epoch+1, batch+1), 'w')
      h5f.create_dataset('imageblob', data=sample_blob)
      h5f.close()


'''
Final Epoch
'''

# generating after the final epoch
display.clear_output(wait=True)
sample_data = generator(random_vector_for_generation, training=False)
print_or_save_sample_images(sample_data.numpy(), is_save=True, epoch=max_epochs)


'''
Restore Checkpoint
'''
# restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


'''
Image Epoch Number
'''
def display_image(epoch_no):
  return PIL.Image.open('images/image_at_epoch_{:04d}.png'.format(epoch_no))
display_image(max_epochs)


'''
Generate GIFs
'''
with imageio.get_writer('dcgan.gif', mode='I') as writer:
  filenames = glob.glob('image*.png')
  filenames = sorted(filenames)
  last = -1
  for i,filename in enumerate(filenames):
    frame = 2*(i**0.5)
    if round(frame) > round(last):
      last = frame
    else:
      continue
    image = imageio.imread(filename)
    writer.append_data(image)
  image = imageio.imread(filename)
  writer.append_data(image)

# this is a hack to display the gif inside the notebook
os.system('cp dcgan.gif dcgan.gif.png')
display.Image(filename="dcgan.gif.png")
