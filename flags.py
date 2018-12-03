import tensorflow as tf
import os

tf.app.flags.DEFINE_string('mode', 'train', 'train / test')
tf.app.flags.DEFINE_string('model_dir', 'model', 'model dir')
tf.app.flags.DEFINE_string('data_dir', 'data', 'data dir')

tf.app.flags.DEFINE_integer('batch_size', 100, 'batch size')

tf.app.flags.DEFINE_string('load', '', 'load model')

tf.app.flags.DEFINE_integer('print_step', 1000, 'printing step')
tf.app.flags.DEFINE_integer('save_step', 10000, 'saving step')
tf.app.flags.DEFINE_integer('max_step', 100000, 'number of steps')

FLAGS = tf.app.flags.FLAGS

FLAGS.units = [512, 256, 128]
#FLAGS.units = [128, 32]

if FLAGS.load != '':
  FLAGS.load = os.path.join(FLAGS.model_dir, 'model-{}'.format(FLAGS.load))

