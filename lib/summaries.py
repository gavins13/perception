import tempfile
import moviepy.editor as mpy
import os
import os.path as osp
import tensorflow as tf
import numpy as np
from io import StringIO
from .misc import printt
from subprocess import Popen, PIPE

from tensorflow.python.framework import constant_op
from tensorflow.python.ops import summary_op_util

'''
All of the code in this file is for implementing video/GIF summaries.
Credit to danijar@github. Thanks!
https://github.com/tensorflow/tensorboard/issues/39#issuecomment-568917607
'''

def video_summary(name, video, step=None, fps=20):
  if tf.executing_eagerly() is True:
    return video_summary_eager(name, video, step=step, fps=fps)
  else:
    return video_summary_graph(name, video, step=step, fps=fps)

def encode_gif(frames, fps):
  h, w, c = frames[0].shape
  pxfmt = {1: 'gray', 3: 'rgb24'}[c]
  cmd = ' '.join([
      f'ffmpeg -y -f rawvideo -vcodec rawvideo',
      f'-r {fps:.02f} -s {w}x{h} -pix_fmt {pxfmt} -i - -filter_complex',
      f'[0:v]split[x][z];[z]palettegen[y];[x]fifo[x];[x][y]paletteuse',
      f'-r {fps:.02f} -f gif -'])
  proc = Popen(cmd.split(' '), stdin=PIPE, stdout=PIPE, stderr=PIPE)
  for image in frames:
    proc.stdin.write(image.tostring())
  out, err = proc.communicate()
  if proc.returncode:
    raise IOError('\n'.join([' '.join(cmd), err.decode('utf8')]))
  del proc
  return out


def video_summary_graph_(video_shape, frames, fps, name):
    name = name.decode('utf-8')
    summary = tf.compat.v1.Summary()
    B, T, H, W, C = video_shape
    image = tf.compat.v1.Summary.Image(height=B * H, width=T * W, colorspace=C)
    image.encoded_image_string = encode_gif(frames, fps) #tf.numpy_function(encode_gif, inp=[frames, fps, True], Tout=tf.string)
    summary.value.add(tag=name + '/gif', image=image)
    serialised_summary = summary.SerializeToString()
    return serialised_summary


def video_summary_graph(name, video, step=None, fps=20):
  #name = tf.constant(name).numpy().decode('utf-8')
  #video = np.array(video)
  if video.dtype in (tf.float32, tf.float64):
    video = tf.cast(tf.clip_by_value(255 * video, 0, 255), dtype=tf.uint8)
  B, T, H, W, C = video.get_shape().as_list()
  try:
    #frames = video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))

    frames = tf.transpose(video, [1,2,0,3,4])
    frames = tf.reshape(frames, [T, H, B * W, C])
    video_shape = [B, T, H, W, C]
    serialised_summary = tf.numpy_function(video_summary_graph_, [video_shape, frames, fps, name], tf.string)
    tf.summary.experimental.write_raw_pb(serialised_summary, step)
  except (IOError, OSError) as e:
    print('GIF summaries require ffmpeg in $PATH.', e)
    #frames = video.transpose((0, 2, 1, 3, 4)).reshape((1, B * H, T * W, C))
    frames = tf.transpose(video, [0, 2, 1, 3, 4])
    frames = tf.reshape(frames, [1, B * H, T * W, C])
    tf.summary.image(name + '/grid', frames, step)



def video_summary_eager(name, video, step=None, fps=20):
  name = tf.constant(name).numpy().decode('utf-8')
  video = np.array(video)
  if video.dtype in (np.float32, np.float64):
    video = np.clip(255 * video, 0, 255).astype(np.uint8)
  B, T, H, W, C = video.shape
  try:
    frames = video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))
    summary = tf.compat.v1.Summary()
    image = tf.compat.v1.Summary.Image(height=B * H, width=T * W, colorspace=C)
    image.encoded_image_string = encode_gif(frames, fps)
    summary.value.add(tag=name + '/gif', image=image)
    tf.summary.experimental.write_raw_pb(summary.SerializeToString(), step)
  except (IOError, OSError) as e:
    print('GIF summaries require ffmpeg in $PATH.', e)
    frames = video.transpose((0, 2, 1, 3, 4)).reshape((1, B * H, T * W, C))
    tf.summary.image(name + '/grid', frames, step)
